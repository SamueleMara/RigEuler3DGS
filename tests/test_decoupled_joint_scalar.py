"""Tests for CODEX_DECOUPLED_JOINT_KILLING_AND_SCALAR_TRAJECTORY.

Covers every required test in the spec: §48 scale invariance, §49 alpha*
recovery, §50 wrong joint, §51 acceleration, §52 smooth reversal, §53 all-frame
gradient, §54 Stage-B render gradient chain, §55 endpoint gradient, §56
separation of losses, §57 one-joint invariant -- plus the supporting primitives.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import torch

from rigid_parts import shared_killing_solver as sks
from rigid_parts.decoupled_joint_scalar import (
    alpha_scalar_agreement,
    analytical_frame_scalar,
    charbonnier,
    frame_joint_consistency,
    robust_residual,
    trimmed_chamfer,
    velocity_magnitude_loss,
)
from rigid_parts.robust_inner_killing import (
    integrate_scalar,
    project_field_to_rigid_twist,
    rigid_basis,
    scalar_sign_flip_loss,
    se3_exp_twist,
)

torch.manual_seed(20260922)
DT = torch.float64
SOLVER_SRC = Path(sks.__file__).read_text(encoding="utf-8")
SOLVER_AST = ast.parse(SOLVER_SRC)

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def make_points(m: int = 40) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.rand((m, 3), generator=g, dtype=DT) * 2.0 - 1.0


def rigid_velocity(points: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nij,j->ni", rigid_basis(points), xi)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def test_robust_residual():
    r = torch.tensor([0.0, 0.5, 2.0], dtype=DT)
    check("rho/l2", torch.allclose(robust_residual(r, mode="l2"), r.square()))
    h = robust_residual(r, mode="huber", delta=1.0)
    check("rho/huber quadratic below delta", math.isclose(float(h[1]), 0.125))
    check("rho/huber linear above delta", math.isclose(float(h[2]), 1.5))
    c = robust_residual(r, mode="charbonnier", delta=1.0e-3)
    check("rho/charbonnier zero at zero", abs(float(c[0])) < 1e-12)
    check("rho/charbonnier monotone", bool((c[1:] > c[:-1]).all()))
    check("charbonnier direct", torch.allclose(charbonnier(r, 1e-3), c))
    try:
        robust_residual(r, mode="nope")
        raise SystemExit("unreachable")
    except ValueError:
        check("rho rejects unknown mode", True)


def test_alpha_recovers_amplitude():
    """§49: v_hat_i(t) = a_t psi_i  =>  alpha_t* ~= a_t, residual ~= 0."""
    x = make_points()
    xi = torch.tensor([0.3, -0.7, 0.2, 0.05, -0.02, 0.11], dtype=DT)
    psi = rigid_velocity(x, xi)
    a = torch.tensor([0.1, 0.2, 0.4, 0.8, 0.7, 0.4], dtype=DT)
    v = a[:, None, None] * psi[None]
    w = torch.ones((int(a.numel()), int(x.shape[0])), dtype=DT)

    alpha = analytical_frame_scalar(psi, v, w)
    check("§49 alpha* == a_t", torch.allclose(alpha, a, atol=1e-10),
          f"max err {float((alpha - a).abs().max()):.2e}")
    fit = frame_joint_consistency(psi, v, w, mode="l2")
    check("§49 residual ~ 0", float(fit.loss.detach()) < 1e-18, f"loss {float(fit.loss.detach()):.2e}")
    check("§49 point residual ~ 0", float(fit.point_residual.max()) < 1e-9)
    check("§49 alpha is not a Parameter", not isinstance(fit.alpha, torch.nn.Parameter))


def test_alpha_is_weighted_least_squares():
    x = make_points(12)
    psi = rigid_velocity(x, torch.tensor([0.2, 0.1, -0.4, 0.0, 0.3, 0.1], dtype=DT))
    v = torch.randn((3, 12, 3), dtype=DT)
    w = torch.rand((3, 12), dtype=DT)
    alpha = analytical_frame_scalar(psi, v, w, eps=0.0)
    # d/d alpha  sum_i w ||v - alpha psi||^2 = 0 at the optimum. eps is dropped
    # here so the check tests the formula, not the guard against an empty frame.
    for t in range(3):
        grad = float((-2.0 * w[t] * ((v[t] - alpha[t] * psi) * psi).sum(-1)).sum())
        check(f"§8 stationary at alpha* (t={t})", abs(grad) < 1e-10, f"dL/dalpha {grad:.2e}")


def test_alpha_is_not_per_camera():
    """§12/§36: ONE alpha per (part, frame), regardless of how many observations
    that frame contributes."""
    x = make_points(20)
    psi = rigid_velocity(x, torch.tensor([0.1, 0.4, -0.2, 0.01, 0.0, -0.05], dtype=DT))
    a = torch.tensor([0.5, -0.3], dtype=DT)
    v = a[:, None, None] * psi[None]
    w = torch.rand((2, 20), dtype=DT) + 0.1
    alpha = analytical_frame_scalar(psi, v, w)
    check("§36 one alpha per frame", tuple(alpha.shape) == (2,), str(tuple(alpha.shape)))
    check("§36 alpha independent of the weight split", torch.allclose(alpha, a, atol=1e-10))


def test_scale_invariance():
    """§47/§48: Phi' = a Phi  =>  alpha*' = alpha*/a, alpha*'Phi' = alpha*Phi, L unchanged."""
    x = make_points(30)
    psi = rigid_velocity(x, torch.tensor([0.4, 0.1, -0.3, 0.02, 0.05, 0.0], dtype=DT))
    v = torch.randn((5, 30, 3), dtype=DT) * 0.2 + 0.7 * psi[None]
    w = torch.rand((5, 30), dtype=DT) + 0.05
    base = frame_joint_consistency(psi, v, w, mode="huber", delta=0.3, robust_tau=0.5)
    for a in (0.01, 3.0, 250.0):
        scaled = frame_joint_consistency(a * psi, v, w, mode="huber", delta=0.3, robust_tau=0.5)
        check(f"§48 alpha*' = alpha*/a (a={a})",
              torch.allclose(scaled.alpha, base.alpha / a, rtol=1e-9, atol=1e-12))
        check(f"§48 alpha*'Phi' = alpha*Phi (a={a})",
              torch.allclose(scaled.alpha[:, None, None] * (a * psi)[None],
                             base.alpha[:, None, None] * psi[None], atol=1e-10))
        check(f"§48 L_frame-joint unchanged (a={a})",
              torch.allclose(scaled.loss, base.loss, rtol=1e-9, atol=1e-14),
              f"{float(base.loss):.6e} vs {float(scaled.loss):.6e}")


def fit_free_field(v, w, x, *, steps=400, lr=0.05, mode="l2"):
    """Optimize a free per-point field, project it to ONE rigid twist, and drive
    §10's all-frame consistency. This is the solver's Stage A with the local-cell
    parameterization replaced by the most permissive field possible, so a
    non-zero floor cannot be blamed on a weak field."""
    phi = torch.nn.Parameter(0.1 * torch.randn_like(x))
    opt = torch.optim.Adam([phi], lr=lr)
    fit = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        proj = project_field_to_rigid_twist(x, phi, torch.ones(x.shape[0], dtype=DT), reg=1e-10)
        fit = frame_joint_consistency(proj["rigid_velocity"], v, w, mode=mode)
        fit.loss.backward()
        opt.step()
    proj = project_field_to_rigid_twist(x, phi.detach(), torch.ones(x.shape[0], dtype=DT), reg=1e-10)
    return fit, proj, phi


def test_wrong_joint_has_a_positive_floor():
    """§50: frame 0 rotates about z, frame 1 translates along x. No scalar pair
    reconciles them, so L_frame-joint must stay > 0 even at the optimal alpha*."""
    x = make_points(30)
    v_rot = rigid_velocity(x, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=DT))
    v_tra = rigid_velocity(x, torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=DT))
    w = torch.ones((2, 30), dtype=DT)

    bad, _, _ = fit_free_field(torch.stack([v_rot, v_tra]), w, x)
    good, _, _ = fit_free_field(torch.stack([v_rot, 0.37 * v_rot]), w, x)
    check("§50 inconsistent frames keep L > 0", float(bad.loss.detach()) > 1e-3,
          f"L_bad {float(bad.loss.detach()):.4e}")
    check("§50 consistent frames reach ~0", float(good.loss.detach()) < 1e-6,
          f"L_good {float(good.loss.detach()):.4e}")
    check("§50 the gap is unambiguous", float(bad.loss.detach()) > 1e3 * float(good.loss.detach()))


def test_acceleration_does_not_bias_the_joint():
    """§51: one fixed joint, strongly nonlinear speed. Stage A must recover the
    same axis it recovers from a constant-speed sequence."""
    x = make_points(40)
    xi = torch.tensor([0.0, 0.0, 1.0, 0.2, -0.1, 0.0], dtype=DT)
    psi = rigid_velocity(x, xi)
    a = torch.tensor([0.1, 0.2, 0.4, 0.8, 0.7, 0.4], dtype=DT)
    w = torch.ones((int(a.numel()), 40), dtype=DT)

    fast, proj_a, _ = fit_free_field(a[:, None, None] * psi[None], w, x)
    flat, proj_b, _ = fit_free_field(torch.ones_like(a)[:, None, None] * psi[None], w, x)

    def axis(p):
        return p["omega"] / p["omega"].norm().clamp_min(1e-12)

    cos = float((axis(proj_a) * axis(proj_b)).sum().abs())
    true_cos = float((axis(proj_a) * (xi[:3] / xi[:3].norm())).sum().abs())
    check("§51 accelerating sequence gives the same axis", cos > 1 - 1e-6, f"cos {cos:.9f}")
    check("§51 axis matches the generating joint", true_cos > 1 - 1e-6, f"cos {true_cos:.9f}")
    check("§51 residual stays ~0 despite acceleration", float(fast.loss.detach()) < 1e-6,
          f"{float(fast.loss.detach()):.2e} vs flat {float(flat.loss.detach()):.2e}")
    check("§51 alpha* tracks the speed profile up to gauge",
          float(torch.corrcoef(torch.stack([fast.alpha.detach(), a]))[0, 1]) > 0.999)


def test_smooth_reversal():
    """§52: Stage A still sees ONE joint through a sign reversal, and Stage B's
    flip penalty stays small for a smooth zero crossing."""
    x = make_points(40)
    psi = rigid_velocity(x, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.3], dtype=DT))
    a = torch.tensor([0.4, 0.2, 0.05, 0.0, -0.05, -0.2], dtype=DT)
    w = torch.ones((int(a.numel()), 40), dtype=DT)

    fit, proj, _ = fit_free_field(a[:, None, None] * psi[None], w, x)
    check("§52 one joint survives the reversal", float(fit.loss.detach()) < 1e-6, f"{float(fit.loss.detach()):.2e}")
    alpha = fit.alpha.detach()
    check("§52 alpha* reverses direction", float(alpha[0] * alpha[-1]) < 0,
          f"{float(alpha[0]):.4f} -> {float(alpha[-1]):.4f}")
    # The crossing is exactly through zero, so no adjacent PRODUCT is negative --
    # which is precisely the case §28's flip penalty is meant to leave free.
    check("§52 the crossing passes through zero",
          float(alpha.abs().min()) < 1e-9 * float(alpha.abs().max()),
          f"min|alpha| {float(alpha.abs().min()):.2e}")

    smooth = float(scalar_sign_flip_loss(a))
    abrupt = float(scalar_sign_flip_loss(torch.tensor([0.4, 0.4, 0.4, -0.4, -0.4, -0.4], dtype=DT)))
    check("§52 smooth reversal is nearly free", smooth < 0.01, f"{smooth:.4e}")
    check("§52 abrupt reversal is penalised", abrupt > 20 * max(smooth, 1e-9),
          f"smooth {smooth:.4e} abrupt {abrupt:.4e}")


def test_all_frame_gradient():
    """§53: the SAME field parameter must receive gradient from every frame, and
    dropping a frame must change that gradient."""
    x = make_points(25)
    xi = torch.tensor([0.1, 0.5, -0.2, 0.05, 0.0, 0.2], dtype=DT)
    psi_true = rigid_velocity(x, xi)
    a = torch.tensor([0.2, 0.5, -0.3, 0.9], dtype=DT)
    v = a[:, None, None] * psi_true[None] + 0.02 * torch.randn((4, 25, 3), dtype=DT)
    w = torch.ones((4, 25), dtype=DT)

    def grad_of(frames):
        torch.manual_seed(3)           # same starting field for every subset
        phi = torch.nn.Parameter(0.3 * torch.randn(x.shape, dtype=DT))
        proj = project_field_to_rigid_twist(x, phi, torch.ones(25, dtype=DT), reg=1e-10)
        fit = frame_joint_consistency(proj["rigid_velocity"], v[frames], w[frames], mode="l2")
        fit.loss.backward()
        return phi.grad.clone()

    full = grad_of([0, 1, 2, 3])
    check("§53 gradient reaches the field", float(full.abs().max()) > 0)
    check("§53 gradient is finite", bool(torch.isfinite(full).all()))
    for drop in range(4):
        kept = [f for f in range(4) if f != drop]
        partial = grad_of(kept)
        delta = float((full - partial).abs().max())
        check(f"§53 removing frame {drop} changes the gradient", delta > 1e-9, f"d {delta:.3e}")

    # Every frame's own term is non-zero, i.e. the sum is not carried by one frame.
    singles = [float(grad_of([f]).abs().max()) for f in range(4)]
    check("§53 every frame contributes", all(s > 0 for s in singles),
          " ".join(f"{s:.2e}" for s in singles))


def test_se3_exp_gradient_is_finite_at_zero_and_tiny_angle():
    """The real cause of the Stage-B NaN. q(t_0) = 0 always, and a near-prismatic
    joint keeps |omega * q| tiny for every frame, so d||w||/dw = 0/0 poisoned the
    gradient of s(t) even though the loss itself was finite."""
    u = torch.tensor([0.0, 0.3, 0.0], dtype=DT)
    for name, omega in (("zero omega", torch.zeros(3, dtype=DT)),
                        ("near-prismatic omega", torch.tensor([3e-4, 0.0, 0.0], dtype=DT)),
                        ("ordinary omega", torch.tensor([0.0, 0.0, 0.7], dtype=DT))):
        q = torch.nn.Parameter(torch.tensor([0.0, 1e-9, 1e-5, 0.05, 0.5, 2.0], dtype=DT))
        T = se3_exp_twist(omega, u, q)
        check(f"forward is finite ({name})", bool(torch.isfinite(T).all()))
        T.sum().backward()
        check(f"gradient is finite ({name})", bool(torch.isfinite(q.grad).all()),
              f"{int((~torch.isfinite(q.grad)).sum())} non-finite of {q.numel()}")
    # q = 0 must give exactly the identity, with a finite gradient.
    q0 = torch.nn.Parameter(torch.zeros(1, dtype=DT))
    T0 = se3_exp_twist(torch.tensor([0.0, 0.0, 1.0], dtype=DT), u, q0)
    check("q=0 gives the identity", torch.allclose(T0[0], torch.eye(4, dtype=DT), atol=1e-14))
    T0.sum().backward()
    check("q=0 gradient is finite", bool(torch.isfinite(q0.grad).all()))

    # Both branches must match torch.matrix_exp of the 4x4 twist -- an
    # independent reference -- on either side of the theta = 0.1 switch.
    omega = torch.tensor([0.3, -0.4, 0.8], dtype=DT)
    qs = torch.tensor([0.0, 1e-6, 1e-3, 0.05, 0.1, 0.11, 0.5, 2.0, 6.0], dtype=DT)
    got = se3_exp_twist(omega, u, qs)
    worst = 0.0
    for i, qv in enumerate(qs):
        gen = torch.zeros((4, 4), dtype=DT)
        gen[:3, :3] = torch.tensor([[0.0, -omega[2], omega[1]],
                                    [omega[2], 0.0, -omega[0]],
                                    [-omega[1], omega[0], 0.0]], dtype=DT) * qv
        gen[:3, 3] = u * qv
        worst = max(worst, float((got[i] - torch.matrix_exp(gen)).abs().max()))
    check("matches torch.matrix_exp across the switch", worst < 1e-9, f"max err {worst:.2e}")
    for i, qv in enumerate(qs):
        R = got[i, :3, :3]
        check(f"stays a rotation at q={float(qv):g}",
              torch.allclose(R @ R.T, torch.eye(3, dtype=DT), atol=1e-12)
              and abs(float(torch.det(R)) - 1.0) < 1e-12)


def test_matrix_to_quaternion_gradient_is_finite():
    """The second 0/0 in the Stage-B chain. sqrt(1 + m00 - m11 - m22) is sqrt(0)
    at the identity: finite forward, INFINITE derivative. Every frame of a
    near-prismatic joint is near-identity, so this poisoned s(t)'s gradient even
    though the render loss itself was finite."""
    from utils.rigid_transform_utils import matrix_to_quaternion, quaternion_to_matrix

    axis = torch.tensor([0.3, -0.4, 0.8], dtype=DT)
    axis = axis / axis.norm()
    worst = 0.0
    for angle in (0.0, 1e-9, 1e-4, 0.01, 0.5, 1.5, 3.0, 3.14, math.pi):
        R = se3_exp_twist(axis * angle, torch.zeros(3, dtype=DT),
                          torch.ones(1, dtype=DT))[:, :3, :3]
        worst = max(worst, float((R - quaternion_to_matrix(matrix_to_quaternion(R))).abs().max()))
    check("round-trips exactly, identity through 180 deg", worst < 1e-14, f"max err {worst:.2e}")

    torch.manual_seed(1)
    rand_worst = 0.0
    for _ in range(200):
        v = torch.randn(3, dtype=DT)
        v = v / v.norm() * torch.rand(1, dtype=DT) * 6.283
        R = se3_exp_twist(v, torch.zeros(3, dtype=DT), torch.ones(1, dtype=DT))[:, :3, :3]
        rand_worst = max(rand_worst, float(
            (R - quaternion_to_matrix(matrix_to_quaternion(R))).abs().max()))
    check("round-trips over 200 random rotations", rand_worst < 1e-14, f"max err {rand_worst:.2e}")

    for label, omega in (("zero omega", torch.zeros(3, dtype=DT)),
                         ("near-prismatic", torch.tensor([3e-4, 0.0, 0.0], dtype=DT)),
                         ("ordinary", torch.tensor([0.0, 0.0, 0.9], dtype=DT))):
        q = torch.nn.Parameter(torch.tensor([0.0, 1e-6, 0.3, 0.7], dtype=DT))
        R = se3_exp_twist(omega, torch.tensor([0.0, 0.3, 0.0], dtype=DT), q)[:, :3, :3]
        matrix_to_quaternion(R).sum().backward()
        check(f"gradient is finite ({label})", bool(torch.isfinite(q.grad).all()),
              f"{int((~torch.isfinite(q.grad)).sum())} non-finite of {q.numel()}")

    check("the branch choice carries no gradient",
          "candidates.detach().argmax" in
          Path(inspect.getfile(matrix_to_quaternion)).read_text(encoding="utf-8"))


def test_stage_b_transport_gradient():
    """§54: s(t) -> q(t) -> T(t) -> transported Gaussians must give finite,
    non-zero gradients on MULTIPLE s(t) entries."""
    x = make_points(20)
    omega = torch.tensor([0.0, 0.0, 0.8], dtype=DT)
    u = torch.tensor([0.1, 0.0, 0.0], dtype=DT)
    frames = 6
    dt = torch.full((frames - 1,), 0.2, dtype=DT)
    speed = torch.nn.Parameter(torch.linspace(0.2, 0.9, frames, dtype=DT))

    q = integrate_scalar(speed, dt)
    T = se3_exp_twist(omega, u, q)
    moved = torch.einsum("tij,nj->tni", T[:, :3, :3], x) + T[:, :3, 3][:, None]
    loss = (moved - x[None]).square().mean()
    loss.backward()
    g = speed.grad
    check("§54 gradients are finite", bool(torch.isfinite(g).all()))
    check("§54 grad(s[0]) != 0", abs(float(g[0])) > 0, f"{float(g[0]):.3e}")
    check("§54 grad(s[mid]) != 0", abs(float(g[frames // 2])) > 0, f"{float(g[frames // 2]):.3e}")
    check("§54 grad(s[-1]) != 0", abs(float(g[-1])) > 0, f"{float(g[-1]):.3e}")
    check("§54 more than one s(t) receives gradient", int((g.abs() > 0).sum()) >= 3,
          f"{int((g.abs() > 0).sum())}/{frames}")


def test_endpoint_gradient_reaches_every_scalar():
    """§46/§55: the final-PCD loss must backpropagate to earlier s(t) via q(T)."""
    x = make_points(30)
    omega = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    u = torch.tensor([0.0, 0.05, 0.0], dtype=DT)
    frames = 7
    dt = torch.full((frames - 1,), 0.25, dtype=DT)
    speed = torch.nn.Parameter(torch.full((frames,), 0.3, dtype=DT))

    truth = se3_exp_twist(omega, u, torch.tensor([0.9], dtype=DT))[0]
    target = x @ truth[:3, :3].T + truth[:3, 3]

    q = integrate_scalar(speed, dt)
    T = se3_exp_twist(omega, u, q[-1:])[0]
    pred = x @ T[:3, :3].transpose(0, 1) + T[:3, 3]
    out = trimmed_chamfer(pred, target, forward_trim=1.0, backward_trim=1.0, backward_budget=1.0)
    out["loss"].backward()
    g = speed.grad
    check("§55 grad(s[0]) != 0", abs(float(g[0])) > 1e-12, f"{float(g[0]):.3e}")
    check("§55 grad(s[mid]) != 0", abs(float(g[frames // 2])) > 1e-12)
    check("§55 grad(s[-1]) != 0", abs(float(g[-1])) > 1e-12)
    check("§55 all scalars receive gradient", int((g.abs() > 1e-12).sum()) == frames,
          f"{int((g.abs() > 1e-12).sum())}/{frames}")
    # §46: interior scalars carry twice the trapezoid weight of the endpoints.
    check("§46 trapezoid weighting", abs(float(g[1]) / float(g[0]) - 2.0) < 1e-6,
          f"ratio {float(g[1]) / float(g[0]):.6f}")


def test_endpoint_optimization_converges():
    """§55 end to end: descending the endpoint loss actually reaches q(T)."""
    x = make_points(40)
    omega = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    u = torch.tensor([0.0, 0.0, 0.0], dtype=DT)
    frames, step = 6, 0.2
    dt = torch.full((frames - 1,), step, dtype=DT)
    q_true = 0.8
    truth = se3_exp_twist(omega, u, torch.tensor([q_true], dtype=DT))[0]
    target = x @ truth[:3, :3].T + truth[:3, 3]

    speed = torch.nn.Parameter(torch.full((frames,), 0.1, dtype=DT))
    opt = torch.optim.Adam([speed], lr=0.05)
    for _ in range(600):
        opt.zero_grad(set_to_none=True)
        q = integrate_scalar(speed, dt)
        T = se3_exp_twist(omega, u, q[-1:])[0]
        pred = x @ T[:3, :3].transpose(0, 1) + T[:3, 3]
        trimmed_chamfer(pred, target, forward_trim=1.0, backward_trim=1.0,
                        backward_budget=1.0)["loss"].backward()
        opt.step()
    reached = float(integrate_scalar(speed.detach(), dt)[-1])
    check("§55 q(T) converges to the endpoint", abs(reached - q_true) < 5e-3,
          f"q(T) {reached:.5f} vs {q_true}")


def test_trimmed_chamfer_rejects_outliers():
    """§25: a few stray predicted points must not dominate the endpoint."""
    g = torch.Generator().manual_seed(11)
    target = torch.rand((300, 3), generator=g, dtype=DT)
    pred = target[:100].clone()
    clean = trimmed_chamfer(pred, target, forward_trim=0.9, backward_trim=0.9)
    dirty = pred.clone()
    dirty[:5] += 50.0
    poisoned = trimmed_chamfer(dirty, target, forward_trim=0.9, backward_trim=0.9)
    untrimmed = trimmed_chamfer(dirty, target, forward_trim=1.0, backward_trim=1.0)
    check("§25 exact match gives forward ~ 0", float(clean["forward"]) < 1e-12,
          f"{float(clean['forward']):.2e}")
    exact = trimmed_chamfer(pred, target, forward_trim=1.0, backward_trim=1.0,
                            backward_budget=1.0)
    check("§25 a budget matched to the part gives loss ~ 0", float(exact["loss"]) < 1e-12,
          f"{float(exact['loss']):.2e}")
    check("§25 an oversized budget admits unowned final points",
          float(clean["backward"]) > float(exact["backward"]),
          f"budget 1.5 -> {float(clean['backward']):.2e}, budget 1.0 -> "
          f"{float(exact['backward']):.2e}")
    check("§25 trimming survives 5% outliers", float(poisoned["forward"]) < 1e-12,
          f"{float(poisoned['forward']):.2e}")
    check("§25 untrimmed does not", float(untrimmed["forward"]) > 1.0,
          f"{float(untrimmed['forward']):.2e}")
    check("§25 bidirectional", float(clean["backward"]) >= 0.0
          and "backward" in poisoned)
    check("§25 backward budget limits the candidate set",
          int(clean["backward_count"]) <= int(round(0.9 * 1.5 * 100)) + 1,
          f"{int(clean['backward_count'])}")
    check("§25 empty inputs are safe",
          float(trimmed_chamfer(pred[:0], target)["loss"]) == 0.0)


def test_velocity_magnitude_needs_the_learned_scalar():
    """§24: unlike Stage A this term is NOT scale/speed invariant -- that is what
    makes it determine the trajectory."""
    x = make_points(20)
    psi = rigid_velocity(x, torch.tensor([0.2, 0.0, 0.5, 0.0, 0.1, 0.0], dtype=DT))
    a = torch.tensor([0.3, 0.6, 0.9], dtype=DT)
    v = a[:, None, None] * psi[None]
    w = torch.ones((3, 20), dtype=DT)
    exact = velocity_magnitude_loss(psi, v, w, a, mode="l2")
    wrong = velocity_magnitude_loss(psi, v, w, a * 2.0, mode="l2")
    check("§24 correct s_mid gives ~0", float(exact) < 1e-18, f"{float(exact):.2e}")
    check("§24 wrong s_mid is penalised", float(wrong) > 1e-4, f"{float(wrong):.2e}")

    s = torch.nn.Parameter(torch.zeros(3, dtype=DT))
    opt = torch.optim.Adam([s], lr=0.1)
    for _ in range(800):
        opt.zero_grad(set_to_none=True)
        velocity_magnitude_loss(psi, v, w, s, mode="l2").backward()
        opt.step()
    check("§24 recovers the amplitude profile", torch.allclose(s.detach(), a, atol=1e-3),
          f"{[round(float(q), 4) for q in s.detach()]} vs {[float(q) for q in a]}")


def test_alpha_vs_s_agreement_diagnostic():
    """§34: reported, never enforced."""
    a = torch.tensor([0.1, 0.3, 0.6, 0.4], dtype=DT)
    same = alpha_scalar_agreement(a, a * 2.0)
    check("§34 corr = 1 under pure rescaling", abs(same["corr"] - 1.0) < 1e-9)
    check("§34 sign agreement = 1", abs(same["sign_agreement"] - 1.0) < 1e-12)
    check("§34 scale ratio reported", abs(same["scale_ratio"] - 2.0) < 1e-9)
    opposed = alpha_scalar_agreement(a, -a)
    check("§34 anti-correlation detected", opposed["corr"] < -0.999)
    check("§34 sign disagreement detected", opposed["sign_agreement"] == 0.0)


def test_robust_weight_protects_the_fit():
    """§38: surviving false dynamic points must be downweighted, not deleted."""
    x = make_points(60)
    psi = rigid_velocity(x, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=DT))
    a = torch.tensor([0.5, 0.7], dtype=DT)
    v = a[:, None, None] * psi[None]
    v[:, :6] += 30.0                                  # 10% gross outliers
    w = torch.ones((2, 60), dtype=DT)

    plain = frame_joint_consistency(psi, v, w, mode="l2", irls_steps=1)
    robust = frame_joint_consistency(psi, v, w, mode="l2", irls_steps=4, robust_tau=0.1)
    err_plain = float((plain.alpha - a).abs().max())
    err_robust = float((robust.alpha - a).abs().max())
    check("§38 IRLS recovers alpha* despite outliers", err_robust < 0.05, f"{err_robust:.4f}")
    check("§38 IRLS beats the unweighted fit", err_robust < 0.1 * err_plain,
          f"plain {err_plain:.4f} robust {err_robust:.4f}")
    check("§38 outliers are downweighted, not removed",
          bool((robust.robust_weight[:, :6] > 0).all())
          and float(robust.robust_weight[:, :6].max()) < 0.1 * float(
              robust.robust_weight[:, 6:].min()))
    check("§38 robust weight is detached", not robust.robust_weight.requires_grad)


def test_joint_residual_per_point():
    """§39: E_k,i = median_t ||v_hat - alpha* psi||, saved and not written back."""
    x = make_points(30)
    psi = rigid_velocity(x, torch.tensor([0.0, 0.5, 0.0, 0.0, 0.0, 0.1], dtype=DT))
    v = torch.stack([0.4 * psi, 0.9 * psi, 0.2 * psi])
    v[:, 3] += 5.0
    w = torch.ones((3, 30), dtype=DT)
    others = [i for i in range(30) if i != 3]
    fit = frame_joint_consistency(psi, v, w, mode="l2")
    check("§39 per-point residual shape", tuple(fit.point_residual.shape) == (30,))
    worst = int(fit.point_residual.argmax())
    check("§39 the inconsistent point stands out", worst == 3, f"argmax {worst}")
    check("§39 it stands out by an order of magnitude",
          float(fit.point_residual[3]) > 10 * float(fit.point_residual[others].max()),
          f"{float(fit.point_residual[3]):.3f} vs {float(fit.point_residual[others].max()):.3f}")
    # A single gross outlier still drags the unweighted alpha*, which is exactly
    # why §38's IRLS exists: with it the consistent points return to zero.
    robust = frame_joint_consistency(psi, v, w, mode="l2", irls_steps=4, robust_tau=0.05)
    check("§39 IRLS restores ~0 for consistent points",
          float(robust.point_residual[others].max())
          < 1.0e-3 * float(fit.point_residual[others].max()),
          f"plain {float(fit.point_residual[others].max()):.2e} "
          f"robust {float(robust.point_residual[others].max()):.2e}")
    check("§39 diagnostic is detached", not fit.point_residual.requires_grad)


# ---------------------------------------------------------------------------
# §56 / §57 architecture invariants, checked against the real solver source
# ---------------------------------------------------------------------------
def test_zero_field_is_a_stationary_point_and_is_seeded_away_from():
    """The flip side of §47/§48: because L_frame-joint sees only the DIRECTION of
    Phi, Phi = 0 has identically zero gradient. A zeroed field would never move
    and every part would report identity, so the solver must seed it from real
    observed motion."""
    x = make_points(20)
    psi_true = rigid_velocity(x, torch.tensor([0.0, 0.0, 1.0, 0.1, 0.0, 0.0], dtype=DT))
    v = torch.stack([0.5 * psi_true, 0.9 * psi_true])
    w = torch.ones((2, 20), dtype=DT)

    phi = torch.nn.Parameter(torch.zeros_like(x))
    proj = project_field_to_rigid_twist(x, phi, torch.ones(20, dtype=DT), reg=1e-10)
    frame_joint_consistency(proj["rigid_velocity"], v, w, mode="l2").loss.backward()
    check("zero field gives zero gradient", float(phi.grad.abs().max()) == 0.0,
          f"|grad| {float(phi.grad.abs().max()):.2e}")

    phi2 = torch.nn.Parameter(1e-6 * torch.randn(x.shape, dtype=DT))
    proj2 = project_field_to_rigid_twist(x, phi2, torch.ones(20, dtype=DT), reg=1e-14)
    frame_joint_consistency(proj2["rigid_velocity"], v, w, mode="l2").loss.backward()
    check("any non-zero field does give gradient", float(phi2.grad.abs().max()) > 0.0)

    src = inspect.getsource(sks.solve_shared_killing_sequence)
    check("the solver seeds the field from observed motion",
          "seeded_from[part] = t_star" in src and "project_field_to_rigid_twist(" in src)
    check("the seed is only used without a warm start",
          "if warm_start_field:\n                continue" in src)
    check("the seed is recorded", "seeded_from_interval" in src)
    # The magnitude must be re-anchored on EVERY outer, warm start included: the
    # gauge's `s *= rms` compensation does not survive into the next solve
    # because `speed` is rebuilt as ones while `control` is carried over.
    check("the field magnitude is re-anchored every outer",
          "control[part] *= target_rms[part] / float(rms_now)" in src)
    check("the anchor uses the gauge's own measure",
          src.index("def _field_rms") < src.index("control[part] *= target_rms[part]")
          and "rms = _field_rms(projections, part)" in src)
    check("the anchor is reported", "anchored the field magnitude" in src)




# ---------------------------------------------------------------------------
# Structural contract
#
# CODEX_MERGED_ROBUST_SUPPORT_AND_DETERMINISTIC_LOSS_ROUTING supersedes this
# spec's two-stage layout (its §0: "if an older implementation/spec conflicts
# with this document, this document wins"). Three assertions that used to live
# here are now WRONG under the merged contract and were removed rather than
# weakened:
#
#   - Stage A excludes render/depth/final-PCD. Reversed by merged §37/§38/§41:
#     those losses are now MANDATORY in the geometry block, and a geometry step
#     driven only by strain/rigidity is explicitly not valid identification.
#   - The A-E ablation ladder. Gone with it -- there is no stage in which the
#     informative losses are off.
#   - Stage-B loss normalisation. Removed; L_S is §54's plain weighted sum.
#
# The surviving structural assertions (two disjoint blocks, derived xi, no
# prohibited trainables, deterministic routing, per-loss gradient audit) live in
# tests/test_merged_support_and_routing.py, which owns them for the merged
# contract. What stays here is the mathematics: alpha*, scale invariance, the
# wrong-joint floor, SE(3) and quaternion gradients, trimmed Chamfer, IRLS.
# ---------------------------------------------------------------------------
def test_parameter_blocks_are_still_disjoint():
    """The one structural fact this spec and the merged one agree on."""
    control = torch.nn.Parameter(torch.zeros(2, 4, 3))
    speed = torch.nn.Parameter(torch.ones(2, 9))
    check("geometry block owns only the field",
          [id(t) for t in sks.geometry_parameters(control, speed)] == [id(control)])
    check("scalar block owns only s(t)",
          [id(t) for t in sks.scalar_parameters(control, speed)] == [id(speed)])
    check("the earlier spec's names still resolve",
          sks.stage_a_parameters is sks.geometry_parameters
          and sks.stage_b_parameters is sks.scalar_parameters)


def test_solver_signature_is_still_a_drop_in():
    legacy = inspect.signature(
        __import__("rigid_parts.killing_inner_solver", fromlist=["x"]).solve_nested_killing_sequence)
    new = inspect.signature(sks.solve_shared_killing_sequence)
    missing = set(legacy.parameters) - set(new.parameters)
    check("drop-in: every legacy keyword is accepted", not missing, str(missing))
    extra = set(new.parameters) - set(legacy.parameters)
    check("drop-in: the new keywords are the diagnostic/endpoint ones",
          extra == {"final_xyz", "gt_moving", "soft_part_probs", "soft_part_probs_name"},
          str(extra))
    for name in ("final_xyz", "gt_moving", "soft_part_probs"):
        check(f"drop-in: {name} is optional", new.parameters[name].default is None)
    check("drop-in: the handoff source name defaults to the one-hot warning",
          "one-hot" in str(new.parameters["soft_part_probs_name"].default))


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
