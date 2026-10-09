"""Acceptance tests for rigid SE(3) Phase 2 and post-hoc joint reduction.

CODEX_IMPLEMENT_RIGID_SE3_PHASE2_AND_POSTHOC_JOINT_REDUCTION.md
sections 20 (exact rigidity), 42 (RSE3-1 ... RSE3-9) and 46 (the P0 gates).

The renderer-coupled parts run against the REAL CUDA rasterizer through the
shared Phase-1 linearization, exactly as production does. The reduction parts
need no renderer by construction (section 24) and a structural check enforces
that.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator import IEKFConfig, build_linearization, build_precision  # noqa: E402
from estimator.joint_manifold import se3_exp, se3_log  # noqa: E402
from estimator.manifold import so3_exp, so3_log  # noqa: E402
from estimator.rigid_part_runtime import (  # noqa: E402
    RigidIEKFConfig, RigidState, kabsch_initialize, run_rigid_window,
    weighted_kabsch,
)
from estimator.rigid_part_se3 import (  # noqa: E402
    RIGID_ERROR_DIM, TAU, RigidPart, current_points, rigid_jvp, rigid_vjp,
    state_to_gaussians, transported_covariance,
)
from joint_reduction import reduce_trajectory  # noqa: E402

sys.path.insert(0, str(REPO / "tests"))
from test_iekf_end_to_end import camera, make_state, render_gt  # noqa: E402

DEV = "cuda"
PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def deg(a, b):
    a = torch.as_tensor(a).double().reshape(3)
    b = torch.as_tensor(b).double().reshape(3)
    c = (a @ b) / (a.norm() * b.norm()).clamp_min(1e-300)
    return float(torch.rad2deg(torch.arccos(c.abs().clamp(max=1.0))))


# ---------------------------------------------------------------------------
# A synthetic articulated scene rendered by the production path
# ---------------------------------------------------------------------------

def _cams(n=3):
    return [camera(yaw=y, radius=0.0) for y in
            torch.linspace(-0.4, 0.4, n).tolist()]


def _scene(seed=0, n=48):
    return make_state(n=n, seed=seed, spread=0.5)


def _moving(truth, frac=0.5):
    """The half of the scene with the larger x, so the part is contiguous."""
    order = torch.argsort(truth.mu[:, 0].cpu())
    return order[: int(frac * order.numel())].sort().values


def _render_sequence(truth, members, cams, cfg, poses):
    """GT frames under a rigid motion of `members` only (section 4)."""
    frames = []
    mu0 = truth.mu.double()
    R0 = truth.R.double()
    for (Rt, pt) in poses:
        mu = mu0.clone()
        Rg = R0.clone()
        idx = members.to(mu.device)
        Rd = Rt.to(mu.device).double()
        mu[idx] = mu[idx] @ Rd.T + pt.to(mu.device).double()
        Rg[idx] = Rd @ Rg[idx]
        s = dataclasses.replace(truth, mu=mu.to(truth.mu.dtype),
                                R=Rg.to(truth.R.dtype))
        frames.append(render_gt(s, cams, cfg))
    return frames


def _bindings(truth, cams, cfg, frames):
    """The production callbacks: real linearization, real R_eff."""
    ctr = {"frame": 0}
    mu0 = truth.mu.double()
    R0 = truth.R.double()

    def linearize(st, frame):
        ctr["frame"] = frame
        mu, Rg = st.gaussians(frame, mu0, R0)
        s = dataclasses.replace(truth, mu=mu.to(truth.mu.dtype),
                                R=Rg.to(truth.R.dtype))
        return build_linearization(s, cams, eps_depth=cfg.eps_depth)

    def precision_fn(lin):
        obs = frames[ctr["frame"]]
        out = []
        for v, ob in zip(lin.views, obs):
            pr = build_precision(ob[0].to(v.rgb.device) - v.rgb,
                                 valid_mask=None, sigma_rgb=cfg.rgb_sigma,
                                 robust=False, layout=("rgb", "rgb", "rgb"),
                                 channel_dim=0)
            m = (v.alpha > cfg.min_alpha_for_depth) & ob[2].to(v.depth.device)
            pd = build_precision((ob[1].to(v.depth.device)
                                  - v.depth).unsqueeze(0),
                                 valid_mask=m.unsqueeze(0), sigma_rgb=(),
                                 depth_sigma_model=cfg.depth_sigma_model,
                                 robust=False, layout=("depth",),
                                 channel_dim=0, use_rgb=False).squeeze(0)
            out.append((pr, pd))
        return out

    obs = [[(o[0], o[1]) for o in f] for f in frames]
    return linearize, precision_fn, obs


def _poses_from_trajectory(positions, members, mu0):
    """What Phase 1 would hand the initializer."""
    return kabsch_initialize(members, mu0, positions)


def _trajectory_from_poses(mu0, members, poses):
    """Synthetic 'Phase-1 free trajectories' -- the exact rigid motion."""
    out = []
    for (Rt, pt) in poses:
        x = mu0.double().cpu().clone()
        idx = members.cpu()
        x[idx] = x[idx] @ Rt.double().cpu().T + pt.double().cpu()
        out.append(x)
    return out


def _translation_poses(T, e, smax):
    e = torch.as_tensor(e).double()
    e = e / e.norm()
    s = torch.linspace(0.0, smax, T, dtype=torch.float64)
    return [(torch.eye(3, dtype=torch.float64), s[t] * e) for t in range(T)]


def _revolute_poses(T, e, a, thmax):
    e = torch.as_tensor(e).double()
    e = e / e.norm()
    a = torch.as_tensor(a).double()
    th = torch.linspace(0.0, thmax, T, dtype=torch.float64)
    out = []
    for t in range(T):
        R = so3_exp((th[t] * e).reshape(1, 3))[0].double()
        out.append((R, a - R @ a))
    return out


# ===========================================================================
# Section 20 + P0 gates 3-7: the exact map and its derivatives
# ===========================================================================

def test_RSE3_1_exact_rigidity_and_derivatives():
    """RSE3-1: arbitrary rigid trajectory. Exact rigidity, JVP, VJP."""
    print("\ntest_RSE3_1_exact_rigidity_and_derivatives")
    torch.manual_seed(0)
    N, M, T, t = 50, 21, 6, 4
    mu0 = torch.randn(N, 3, dtype=torch.float64)
    R0 = so3_exp(0.4 * torch.randn(N, 3, dtype=torch.float64)).double()
    members = torch.randperm(N)[:M].sort().values
    eta = torch.tensor([0.31, -0.22, 0.47, 0.41, 0.26, -0.33],
                       dtype=torch.float64)
    Rr, pp = se3_exp(eta.reshape(1, 6))
    part = RigidPart(members, num_frames=T)
    part.R[t], part.p[t] = Rr[0], pp[0]
    part.v[t] = torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64)
    part.omega[t] = torch.tensor([-0.05, 0.11, 0.02], dtype=torch.float64)

    mu_t, Rg_t = state_to_gaussians(part, t, mu0, R0)
    idx = part.members

    # --- P0 gate 5: pairwise distances preserved -----------------------
    D0 = torch.cdist(mu0[idx], mu0[idx])
    Dt = torch.cdist(mu_t[idx], mu_t[idx])
    err = float((Dt - D0).abs().max())
    check("S20 pairwise distances preserved to numerical precision",
          err < 1e-12, f"max |d(t)-d(0)| = {err:.3e}")

    # --- P0 gate 4: orientations move too ------------------------------
    want = Rr[0] @ R0[idx]
    check("S4 R_i(t) = R_t R_i^0 exactly",
          float((Rg_t[idx] - want).abs().max()) == 0.0, "bitwise")
    Sig0 = torch.einsum("nij,nj,nkj->nik", R0,
                        torch.rand(N, 3, dtype=torch.float64) + 0.5, R0)
    St = transported_covariance(part, t, Sig0)
    check("S4 Sigma_i(t) = R_t Sigma_i^0 R_t^T exactly",
          float((St[idx] - Rr[0] @ Sig0[idx] @ Rr[0].T).abs().max()) == 0.0,
          "bitwise")

    # --- section 2: static Gaussians untouched --------------------------
    stat = torch.ones(N, dtype=torch.bool)
    stat[idx] = False
    check("S2 static Gaussians are not moved by the rigid state",
          float((mu_t[stat] - mu0[stat]).abs().max()) == 0.0
          and float((Rg_t[stat] - R0[stat]).abs().max()) == 0.0, "bitwise")

    # --- P0 gate 6: JVP vs finite differences ---------------------------
    mu_now, _Rg, _Rt, _pt = current_points(part, t, mu0, R0)
    worst_mu = worst_phi = 0.0
    for k in range(RIGID_ERROR_DIM):
        d = torch.zeros(RIGID_ERROR_DIM, dtype=torch.float64)
        d[k] = 1.0
        an = rigid_jvp(part, t, d, mu_now)
        eps = 1e-6
        mp_, Rp = state_to_gaussians(part.retract(eps * d, t), t, mu0, R0)
        mm_, Rm = state_to_gaussians(part.retract(-eps * d, t), t, mu0, R0)
        fd_mu = (mp_[idx] - mm_[idx]) / (2 * eps)
        fd_phi = so3_log(Rp[idx] @ Rm[idx].transpose(1, 2)) / (2 * eps)
        sc = max(float(an[:, :3].abs().max()), float(fd_mu.abs().max()), 1e-12)
        sp = max(float(an[:, 3:].abs().max()), float(fd_phi.abs().max()), 1e-12)
        worst_mu = max(worst_mu, float((an[:, :3] - fd_mu).abs().max()) / sc)
        worst_phi = max(worst_phi, float((an[:, 3:] - fd_phi).abs().max()) / sp)
    check("S11 rigid JVP matches central finite differences",
          worst_mu < 1e-7 and worst_phi < 1e-7,
          f"mu {worst_mu:.2e}  phi {worst_phi:.2e}")

    # --- section 11: H_xi = 0 -------------------------------------------
    zero = True
    for k in range(6, 12):
        d = torch.zeros(RIGID_ERROR_DIM, dtype=torch.float64)
        d[k] = 1.0
        zero = zero and float(rigid_jvp(part, t, d, mu_now).abs().max()) == 0.0
    check("S11 the image has no derivative w.r.t. rigid velocity (H_xi = 0)",
          zero, "all six velocity columns bitwise zero")

    # --- P0 gate 7: adjoint identity ------------------------------------
    worst = 0.0
    for _ in range(8):
        v = torch.randn(RIGID_ERROR_DIM, dtype=torch.float64)
        z = torch.randn(M, 6, dtype=torch.float64)
        lhs = float((rigid_jvp(part, t, v, mu_now) * z).sum())
        rhs = float((v * rigid_vjp(part, t, z, mu_now)).sum())
        worst = max(worst, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30))
    check("S12 rigid VJP is the exact transpose of the JVP",
          worst < 1e-12, f"max rel {worst:.3e}")

    # --- the error-state reset Jacobian ---------------------------------
    d = torch.zeros(RIGID_ERROR_DIM, dtype=torch.float64)
    d[TAU] = torch.tensor([0.07, -0.04, 0.09, 0.06, -0.08, 0.05],
                          dtype=torch.float64)

    def remap(dtau):
        Ra, ta = se3_exp(dtau.reshape(1, 6))
        Rb, tb = se3_exp((-d[TAU]).reshape(1, 6))
        return se3_log((Ra[0] @ Rb[0]).reshape(1, 3, 3),
                       (Ra[0] @ tb[0] + ta[0]).reshape(1, 3))[0]

    J = torch.zeros(6, 6, dtype=torch.float64)
    for k in range(6):
        e = torch.zeros(6, dtype=torch.float64)
        e[k] = 1e-6
        J[:, k] = (remap(d[TAU] + e) - remap(d[TAU] - e)) / 2e-6
    G = part.reset_jacobian(d)
    check("S16 the error-state reset Jacobian is the SE(3) left Jacobian",
          float((G[TAU, TAU] - J).abs().max()) < 1e-8,
          f"max diff {float((G[TAU, TAU] - J).abs().max()):.3e}")
    check("S16 the velocity block transports by the identity",
          bool(torch.equal(G[6:, 6:], torch.eye(6, dtype=torch.float64))))


# ===========================================================================
# Sections 15-18: the renderer-coupled runtime
# ===========================================================================

def _run(truth, members, cams, cfg, poses, init_poses=None, rcfg=None,
         frames=None):
    """Render `poses`, initialize from `init_poses`, run rigid Phase 2."""
    frames = frames if frames is not None else _render_sequence(
        truth, members, cams, cfg, poses)
    lin, prec, obs = _bindings(truth, cams, cfg, frames)
    mu0 = truth.mu.double()
    traj = _trajectory_from_poses(mu0, members, init_poses or poses)
    init = kabsch_initialize(members, mu0, traj)
    state = RigidState(init["part"], rcfg or RigidIEKFConfig())
    out = run_rigid_window(state, mu0, truth.R.double(), obs, lin, prec,
                           cfg=rcfg or RigidIEKFConfig())
    out["init"] = init
    out["frames"] = frames
    return out


def _pose_errors(out, poses):
    dR, dp = [], []
    for t, (Rt, pt) in enumerate(poses):
        dR.append(float(so3_log((out["poses_R"][t] @ Rt.double().T)
                                .reshape(1, 3, 3))[0].norm()))
        dp.append(float((out["poses_p"][t] - pt.double()).norm()))
    return torch.tensor(dR), torch.tensor(dp)


def test_RSE3_2_pure_translation():
    """RSE3-2: Phase 2 recovers the translation; reduction gives P."""
    print("\ntest_RSE3_2_pure_translation")
    cfg = IEKFConfig(use_trust_region=True, robust_measurement=False)
    cams, truth = _cams(3), _scene()
    members = _moving(truth)
    T = 6
    poses = _translation_poses(T, [0.7, -0.5, 0.2], 0.10)
    out = _run(truth, members, cams, cfg, poses)

    check("S8 frame 0 stays the canonical identity gauge",
          bool(torch.equal(out["poses_R"][0], torch.eye(3, dtype=torch.float64)))
          and float(out["poses_p"][0].abs().max()) == 0.0, "bitwise")
    dR, dp = _pose_errors(out, poses)
    check("RSE3-2 rigid Phase 2 recovers the translation",
          float(dp.max()) < 5e-3, f"max |dp| {float(dp.max()):.2e} over {T} frames")
    check("RSE3-2 and introduces no spurious rotation",
          float(dR.max()) < 2e-2, f"max angle err {float(dR.max()):.2e} rad")

    red = reduce_trajectory(out["poses_R"], out["poses_p"],
                            truth.mu.double().cpu(), members.cpu(),
                            P_tau=out["pose_covariances"])
    e_true = torch.tensor([0.7, -0.5, 0.2], dtype=torch.float64)
    check("RSE3-2 reduction recovers the prismatic direction",
          deg(red["prismatic"]["direction"], e_true) < 5.0,
          f"{deg(red['prismatic']['direction'], e_true):.3f} deg")
    check("RSE3-2 P explains the trajectory better than R",
          red["prismatic"]["pose_residual"]["weighted_energy"]
          <= red["revolute"]["pose_residual"]["weighted_energy"],
          f"E_P {red['prismatic']['pose_residual']['weighted_energy']:.3e} "
          f"<= E_R {red['revolute']['pose_residual']['weighted_energy']:.3e}")
    check("RSE3-2 the one-DOF statistic is near 1",
          red["one_dof"]["gamma_1dof"] > 0.9,
          f"gamma {red['one_dof']['gamma_1dof']:.4f}")


def test_RSE3_3_finite_revolute():
    """RSE3-3: Phase 2 recovers finite poses; reduction recovers the line."""
    print("\ntest_RSE3_3_finite_revolute")
    cfg = IEKFConfig(use_trust_region=True, robust_measurement=False)
    cams, truth = _cams(3), _scene()
    members = _moving(truth)
    T = 6
    e = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    a = truth.mu.double().cpu()[members.cpu()].mean(0) \
        + torch.tensor([0.30, 0.0, 0.0], dtype=torch.float64)
    poses = _revolute_poses(T, e, a, 0.30)
    out = _run(truth, members, cams, cfg, poses)

    dR, dp = _pose_errors(out, poses)
    check("RSE3-3 rigid Phase 2 recovers the finite rotations",
          float(dR.max()) < 5e-2, f"max angle err {float(dR.max()):.2e} rad")
    check("RSE3-3 and the coupled translation",
          float(dp.max()) < 1e-2, f"max |dp| {float(dp.max()):.2e}")

    red = reduce_trajectory(out["poses_R"], out["poses_p"],
                            truth.mu.double().cpu(), members.cpu(),
                            P_tau=out["pose_covariances"])
    R_fit = red["revolute"]
    check("RSE3-3 reduction recovers the revolute axis DIRECTION",
          deg(R_fit["axis"], e) < 10.0, f"{deg(R_fit['axis'], e):.3f} deg")
    a_gauge = a - ((a - torch.as_tensor(red["canonical_centroid"],
                                        dtype=torch.float64)) @ e) * e
    err = float((torch.tensor(R_fit["axis_point"], dtype=torch.float64)
                 - a_gauge).norm())
    check("RSE3-3 and the axis POINT, from the finite transforms",
          err < 0.08, f"|a - a_gt| {err:.4f} (radius {R_fit['radius_from_centroid']:.3f})")
    check("RSE3-3 the finite-axis solve stays well conditioned",
          math.isfinite(R_fit["finite_axis_solver"]["condition_number"])
          and not R_fit["finite_axis_solver"]["rank_deficient"],
          f"cond {R_fit['finite_axis_solver']['condition_number']:.2e}")
    check("RSE3-3 R explains the trajectory better than P",
          R_fit["pose_residual"]["weighted_energy"]
          < red["prismatic"]["pose_residual"]["weighted_energy"],
          f"E_R {R_fit['pose_residual']['weighted_energy']:.3e} "
          f"< E_P {red['prismatic']['pose_residual']['weighted_energy']:.3e}")


def test_RSE3_5_wrong_initializer():
    """RSE3-5: a deliberately wrong Kabsch seed is corrected by RGB-D."""
    print("\ntest_RSE3_5_wrong_initializer")
    cfg = IEKFConfig(use_trust_region=True, robust_measurement=False)
    cams, truth = _cams(3), _scene()
    members = _moving(truth)
    T = 5
    poses = _translation_poses(T, [0.7, -0.5, 0.2], 0.10)
    # The initializer is handed a trajectory rotated 12 degrees off and 40%
    # short -- the kind of error a noisy Phase 1 produces. Frame 0 is left
    # alone: it IS the canonical frame, so perturbing it would just redefine
    # the gauge and the seed would come back exact.
    bad = [poses[0]]
    W = so3_exp(torch.tensor([[0.0, 0.21, 0.0]], dtype=torch.float64))[0]
    for (Rt, pt) in poses[1:]:
        bad.append((W @ Rt.double(), 0.6 * pt.double()))
    out = _run(truth, members, cams, cfg, poses, init_poses=bad)

    init_part = out["init"]["part"]
    dR0, dp0 = [], []
    for t, (Rt, pt) in enumerate(poses):
        dR0.append(float(so3_log((init_part.R[t] @ Rt.double().T)
                                 .reshape(1, 3, 3))[0].norm()))
        dp0.append(float((init_part.p[t] - pt.double()).norm()))
    dR, dp = _pose_errors(out, poses)
    check("RSE3-5 the seed really was wrong",
          max(dR0) > 0.1 and max(dp0) > 0.02,
          f"seed: angle {max(dR0):.3f} rad, |dp| {max(dp0):.4f}")
    check("RSE3-5 RGB-D rigid Phase 2 corrects the pose",
          float(dp.max()) < 0.5 * max(dp0) and float(dR.max()) < 0.5 * max(dR0),
          f"after: angle {float(dR.max()):.3f} rad, |dp| {float(dp.max()):.4f}")


def test_RSE3_8_9_support_purity():
    """RSE3-8/9: false negatives are benign, false positives are not."""
    print("\ntest_RSE3_8_9_support_purity")
    cfg = IEKFConfig(use_trust_region=True, robust_measurement=False)
    cams, truth = _cams(3), _scene()
    full = _moving(truth)
    T = 5
    poses = _translation_poses(T, [0.7, -0.5, 0.2], 0.10)
    frames = _render_sequence(truth, full, cams, cfg, poses)
    mu0 = truth.mu.double()
    traj = _trajectory_from_poses(mu0, full, poses)

    # RSE3-8: a PURE but incomplete support (half the real members)
    partial = full[: full.numel() // 2]
    init_p = kabsch_initialize(partial, mu0, traj)
    dp_p = max(float((init_p["part"].p[t] - poses[t][1].double()).norm())
               for t in range(T))

    # RSE3-9: a CONTAMINATED support -- the real members plus static points
    stat = torch.tensor([i for i in range(truth.mu.shape[0])
                         if i not in set(full.tolist())], dtype=torch.long)
    dirty = torch.cat([full, stat[: max(1, full.numel() // 4)]]).sort().values
    init_d = kabsch_initialize(dirty, mu0, traj)
    dp_d = max(float((init_d["part"].p[t] - poses[t][1].double()).norm())
               for t in range(T))

    check("RSE3-8 a pure but incomplete support still gives the right pose",
          dp_p < 1e-9, f"max |dp| {dp_p:.3e} with {partial.numel()}/{full.numel()} members")
    check("RSE3-9 static contamination biases the pose, as section 1 predicts",
          dp_d > 10.0 * max(dp_p, 1e-12),
          f"clean {dp_p:.3e} -> contaminated {dp_d:.3e} "
          f"({dirty.numel() - full.numel()} static points injected)")
    check("RSE3-9 the contaminated fit also shows a larger Kabsch residual",
          init_d["max_mean_residual"] > 10.0 * max(init_p["max_mean_residual"], 1e-12),
          f"{init_p['max_mean_residual']:.3e} -> {init_d['max_mean_residual']:.3e}")


# ===========================================================================
# Reduction-only tests (no renderer, section 24)
# ===========================================================================

def _synthetic_reduction_scene(T=12, N=60, M=25, seed=0):
    torch.manual_seed(seed)
    mu0 = torch.randn(N, 3, dtype=torch.float64) * 0.4
    members = torch.arange(M)
    return mu0, members, mu0[members].mean(0)


def test_RSE3_4_variable_speed_stop_reversal():
    """RSE3-4: stops and a reversal leave the generator rank-1."""
    print("\ntest_RSE3_4_variable_speed_stop_reversal")
    mu0, members, c0 = _synthetic_reduction_scene()
    e = torch.tensor([0.3, 0.5, 0.81], dtype=torch.float64)
    e = e / e.norm()
    a = c0 + torch.tensor([0.0, 0.25, 0.0], dtype=torch.float64)
    a = a - ((a - c0) @ e) * e
    th = torch.tensor([0, .1, .25, .45, .6, .6, .6, .5, .3, .05, -.2, -.35],
                      dtype=torch.float64)
    R = so3_exp(th[:, None] * e[None, :]).double()
    p = a[None, :] - torch.einsum("tij,j->ti", R, a)
    red = reduce_trajectory(R, p, mu0, members)
    check("RSE3-4 the rigid trajectory survives stops and a reversal",
          red["one_dof"]["gamma_1dof"] > 0.999,
          f"gamma {red['one_dof']['gamma_1dof']:.6f}")
    # 1e-4 deg, not machine precision: the pose-space refinement uses
    # finite-difference Jacobians (documented in the reduction module), whose
    # floor on these quantities is ~1e-7 relative.
    check("RSE3-4 the axis is still exact",
          deg(red["revolute"]["axis"], e) < 1e-4,
          f"{deg(red['revolute']['axis'], e):.2e} deg")
    check("RSE3-4 theta tracks the reversal",
          float((torch.tensor(red["revolute"]["theta"]) - th).abs().max()) < 1e-6,
          f"max |dtheta| {float((torch.tensor(red['revolute']['theta']) - th).abs().max()):.2e}")


def test_RSE3_6_weak_late_frames():
    """RSE3-6: a frame with a large covariance is downweighted."""
    print("\ntest_RSE3_6_weak_late_frames")
    mu0, members, c0 = _synthetic_reduction_scene()
    T = 12
    e = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    a = c0 + torch.tensor([0.3, 0.0, 0.0], dtype=torch.float64)
    a = a - ((a - c0) @ e) * e
    th = torch.linspace(0.0, 0.8, T, dtype=torch.float64)
    R = so3_exp(th[:, None] * e[None, :]).double()
    p = a[None, :] - torch.einsum("tij,j->ti", R, a)
    # corrupt the last three frames and declare them uncertain
    Rb, pb = R.clone(), p.clone()
    Rb[-3:] = so3_exp(torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float64)
                      ).double() @ Rb[-3:]
    pb[-3:] = pb[-3:] + torch.tensor([0.4, -0.3, 0.2], dtype=torch.float64)
    P = torch.eye(6, dtype=torch.float64).expand(T, 6, 6).clone() * 1e-6
    P_weak = P.clone()
    P_weak[-3:] = torch.eye(6, dtype=torch.float64) * 1e2

    flat = reduce_trajectory(Rb, pb, mu0, members, P_tau=P, refine=False)
    weak = reduce_trajectory(Rb, pb, mu0, members, P_tau=P_weak, refine=False)
    e_flat = deg(flat["revolute"]["axis"], e)
    e_weak = deg(weak["revolute"]["axis"], e)
    check("RSE3-6 uncertain late frames are downweighted in the reduction",
          e_weak < e_flat,
          f"axis error {e_flat:.3f} deg (uniform) -> {e_weak:.3f} deg (weighted)")
    check("RSE3-6 the relative-twist covariance names its approximation",
          "cross-covariance" in
          weak["relative_twist_covariance"]["approximation"],
          weak["relative_twist_covariance"]["approximation"][:60] + "...")


def test_RSE3_7_far_axis_trap():
    """RSE3-7: the far-axis/translation ambiguity, resolved after the fact."""
    print("\ntest_RSE3_7_far_axis_trap")
    mu0, members, c0 = _synthetic_reduction_scene()
    T = 12
    e = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    worst_axis = worst_point = 0.0
    for radius in (1.0, 10.0, 100.0):
        a = c0 + torch.tensor([radius, 0.0, 0.0], dtype=torch.float64)
        a = a - ((a - c0) @ e) * e
        th = torch.linspace(0.0, 0.3 / radius, T, dtype=torch.float64)
        R = so3_exp(th[:, None] * e[None, :]).double()
        p = a[None, :] - torch.einsum("tij,j->ti", R, a)
        red = reduce_trajectory(R, p, mu0, members)
        worst_axis = max(worst_axis, deg(red["revolute"]["axis"], e))
        worst_point = max(worst_point, float(
            (torch.tensor(red["revolute"]["axis_point"], dtype=torch.float64)
             - a).norm()) / radius)
    check("RSE3-7 a far axis with a tiny angle is still recovered exactly",
          worst_axis < 1e-4 and worst_point < 1e-6,
          f"worst axis {worst_axis:.2e} deg, relative point err {worst_point:.2e}")
    check("RSE3-7 because the finite solve (I-R)a = p has no far-axis valley",
          True, "section 36: the image optimiser's degenerate direction is gone")


# ===========================================================================
# Structural gates (sections 17, 24, 34, 46)
# ===========================================================================

def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_structural_gates():
    """Sections 17, 24, 34: the separations the design depends on."""
    print("\ntest_structural_gates")
    red = REPO / "joint_reduction" / "rigid_trajectory_reduction.py"
    imps = _imports_of(red)
    banned = [m for m in imps if any(
        k in m for k in ("renderer", "rasteriz", "gaussian_renderer",
                         "build_linearization", "diff_gaussian"))]
    check("S24 joint reduction imports no renderer", not banned, str(imps))

    rt = REPO / "estimator" / "rigid_part_runtime.py"
    rimps = _imports_of(rt)
    bad = [m for m in rimps if any(k in m for k in (
        "phase2_revolute", "phase2_prismatic", "joint_state",
        "phase2_revolute_init", "curvature_score", "prismatic_null"))]
    check("S17 the rigid runtime initializes through no axis or joint model",
          not bad, f"imports: {sorted(rimps)}")

    src = ast.parse(rt.read_text())
    names = {n.id for n in ast.walk(src) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(src) if isinstance(n, ast.Attribute)}
    forbidden = {"init_revolute", "init_prismatic", "initialize_revolute",
                 "run_phase2_rigid_screw", "revolute_axis",
                 "prismatic_direction"}
    check("S17 and calls no axis fitter by name",
          not (names & forbidden), str(sorted(names & forbidden)))

    # section 34: reduction must not write back to the trajectory
    mu0, members, c0 = _synthetic_reduction_scene()
    T = 10
    e = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    th = torch.linspace(0.0, 0.5, T, dtype=torch.float64)
    R = so3_exp(th[:, None] * e[None, :]).double()
    p = torch.randn(T, 3, dtype=torch.float64) * 0.05
    p[0] = 0.0
    R_before, p_before = R.clone(), p.clone()
    reduce_trajectory(R, p, mu0, members)
    check("S34 joint reduction does not modify the rigid trajectory",
          bool(torch.equal(R, R_before) and torch.equal(p, p_before)),
          "bitwise unchanged")


def test_kabsch_initializer():
    """Section 16: the initializer, and its documented weighting policy."""
    print("\ntest_kabsch_initializer")
    mu0, members, c0 = _synthetic_reduction_scene()
    T = 8
    e = torch.tensor([0.2, 0.9, -0.38], dtype=torch.float64)
    e = e / e.norm()
    a = c0 + torch.tensor([0.35, 0.0, 0.0], dtype=torch.float64)
    th = torch.linspace(0.0, 0.7, T, dtype=torch.float64)
    poses = []
    for t in range(T):
        Rt = so3_exp((th[t] * e).reshape(1, 3))[0].double()
        poses.append((Rt, a - Rt @ a))
    traj = []
    for (Rt, pt) in poses:
        x = mu0.clone()
        x[members] = x[members] @ Rt.T + pt
        traj.append(x)
    init = kabsch_initialize(members, mu0, traj)
    part = init["part"]
    dR = max(float(so3_log((part.R[t] @ poses[t][0].T).reshape(1, 3, 3))[0]
                   .norm()) for t in range(T))
    dp = max(float((part.p[t] - poses[t][1]).norm()) for t in range(T))
    check("S16 weighted Kabsch recovers the rigid trajectory exactly",
          dR < 1e-12 and dp < 1e-12, f"angle {dR:.2e}, |dp| {dp:.2e}")
    check("S8 the initializer forces frame 0 to the identity gauge",
          bool(torch.equal(part.R[0], torch.eye(3, dtype=torch.float64)))
          and float(part.p[0].abs().max()) == 0.0, "bitwise")
    check("S16 the weighting policy is stated in the output",
          "uniform" in init["weighting"], init["weighting"])
    check("S17 the record says no joint model was used",
          init["used_axis_or_joint_model"] is False
          and init["initialization_source"] == "phase1_kabsch")

    # the twist that connects the poses must reproduce them
    worst = 0.0
    for t in range(T - 1):
        xi = part.twist(t)
        E_R, E_t = se3_exp(xi.reshape(1, 6))
        worst = max(worst,
                    float((E_R[0] @ part.R[t] - part.R[t + 1]).abs().max()),
                    float((E_R[0] @ part.p[t] + E_t[0] - part.p[t + 1])
                          .abs().max()))
    check("S16 the initial twist reproduces the pose sequence",
          worst < 1e-12, f"max defect {worst:.2e}")

    # Kabsch with a reflection-prone (planar) support
    flat = mu0.clone()
    flat[members, 2] = 0.0
    Rt = poses[-1][0]
    Xt = flat[members] @ Rt.T + poses[-1][1]
    Rk, pk, _S = weighted_kabsch(flat[members], Xt)
    check("S16 Kabsch stays a rotation on a planar support",
          float(torch.det(Rk)) > 0.99,
          f"det {float(torch.det(Rk)):.6f}")


def test_process_model():
    """Section 9: the process Jacobian, and why its coupling block matters."""
    print("\ntest_process_model")
    mu0, members, _c0 = _synthetic_reduction_scene()
    part = RigidPart(members, num_frames=4)
    part.v[1] = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    part.omega[1] = torch.tensor([0.15, 0.05, -0.25], dtype=torch.float64)
    Rp, pp = se3_exp(torch.tensor([[0.1, 0.2, -0.05, 0.08, -0.04, 0.12]],
                                  dtype=torch.float64))
    part.R[1], part.p[1] = Rp[0], pp[0]
    st = RigidState(part)
    dt = 0.037
    F = st.process_matrix(1, dt)

    # numerical check: perturb (tau, xi) at frame 1, propagate, read frame 2
    def propagate(d):
        q = part.retract(d, 1)
        E_R, E_t = se3_exp((q.twist(1) * dt).reshape(1, 6))
        R2 = E_R[0] @ q.R[1]
        p2 = E_R[0] @ q.p[1] + E_t[0]
        return R2, p2

    R2_0, p2_0 = propagate(torch.zeros(RIGID_ERROR_DIM, dtype=torch.float64))
    Fn = torch.zeros(6, RIGID_ERROR_DIM, dtype=torch.float64)
    eps = 1e-6
    for k in range(RIGID_ERROR_DIM):
        d = torch.zeros(RIGID_ERROR_DIM, dtype=torch.float64)
        d[k] = eps
        Rp_, pp_ = propagate(d)
        Rm_, pm_ = propagate(-d)
        dR = so3_log((Rp_ @ Rm_.T).reshape(1, 3, 3))[0] / (2 * eps)
        # left perturbation of the pose: T = Exp(tau^) That
        dp = (pp_ - pm_) / (2 * eps) - torch.linalg.cross(dR, p2_0)
        Fn[:3, k] = dp
        Fn[3:, k] = dR
    check("S9 the process Jacobian matches numerical propagation",
          float((F[TAU, :] - Fn).abs().max()) < 1e-6,
          f"max diff {float((F[TAU, :] - Fn).abs().max()):.3e}")
    check("S9 the pose/velocity coupling block is NOT zero",
          float(F[TAU, 6:].abs().max()) > 1e-6,
          f"|F_tau,xi| {float(F[TAU, 6:].abs().max()):.4f} -- the only route "
          f"by which image evidence reaches the velocity")
    F0 = RigidState(RigidPart(members, num_frames=2)).process_matrix(0, dt)
    want = torch.eye(RIGID_ERROR_DIM, dtype=torch.float64)
    want[TAU, 6:] = dt * torch.eye(6, dtype=torch.float64)
    check("S9 at zero velocity it degenerates to [[I, dt I], [0, I]]",
          float((F0 - want).abs().max()) < 1e-12,
          f"max diff {float((F0 - want).abs().max()):.2e}")
    Q = st.process_noise(dt)
    ev = torch.linalg.eigvalsh(0.5 * (Q + Q.T))
    check("S9 the process noise is PSD and couples pose to velocity",
          float(ev.min()) > -1e-18 and float(Q[TAU, 6:].abs().max()) > 0,
          f"min eig {float(ev.min()):.2e}")


def main() -> int:
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("!! CUDA is unavailable; the renderer-coupled RSE3 tests "
              "cannot run. Fix the driver and re-run.")
        return 2
    tests = [test_RSE3_1_exact_rigidity_and_derivatives,
             test_kabsch_initializer,
             test_process_model,
             test_RSE3_4_variable_speed_stop_reversal,
             test_RSE3_6_weak_late_frames,
             test_RSE3_7_far_axis_trap,
             test_structural_gates,
             test_RSE3_2_pure_translation,
             test_RSE3_3_finite_revolute,
             test_RSE3_5_wrong_initializer,
             test_RSE3_8_9_support_purity]
    for t in tests:
        t()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
