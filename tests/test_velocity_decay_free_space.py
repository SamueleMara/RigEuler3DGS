#!/usr/bin/env python
"""Mean-reverting velocity (estimator/velocity_decay.py) and the free-space support
refinement (estimator/free_space_support.py). CPU only.

  [D1] the discretization equals the integral of the continuous OU covariance;
       tau = 0 is the constant-velocity triple exactly
  [D2] Phase 1: tau = 0 reproduces the old predict / F / Q bitwise; tau > 0 decays
       v, omega by phi and integrates over h; F is the Jacobian of the prediction
  [D3] Phase 2 rigid state: tau = 0 bitwise as before; with no measurement the pose
       stops after ~ v0 tau instead of travelling v0 t
  [D4] Phase 3 rate model: the same, on (q, qdot)
  [F1] the projection agrees with tools/cross_camera_registration.backproject
  [F2] closing door in front of a static interior: the hidden interior that the
       coherence support wrongly holds is dropped, the half of the door it missed
       is added, the wall stays static

    python tests/test_velocity_decay_free_space.py
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from tests.wrong_camera_fixtures import check_factory, finish  # noqa: E402


def ou_integral(dt, tau, n=20001):
    from scipy.linalg import expm
    A = np.array([[0.0, 1.0], [0.0, -1.0 / tau if tau > 0 else 0.0]])
    s = np.linspace(0.0, dt, n)
    vals = np.array([expm(A * (dt - x)) @ np.array([[0.0], [1.0]]) @ np.array([[0.0, 1.0]]) @ expm(A * (dt - x)).T
                     for x in s])
    return expm(A * dt), np.trapz(vals, s, axis=0)


def test_terms(check):
    from estimator.velocity_decay import terms
    worst_F, worst_Q = 0.0, 0.0
    for dt, tau in [(0.04, 0.5), (0.04, 2.0), (0.04, 30.0), (0.5, 0.3), (0.04, 0.0)]:
        phi, h, cxx, cxv, cvv = terms(dt, tau)
        F, Q = ou_integral(dt, tau)
        worst_F = max(worst_F, np.abs(F - [[1, h], [0, phi]]).max())
        worst_Q = max(worst_Q, np.abs(Q - [[cxx, cxv], [cxv, cvv]]).max() / np.abs(Q).max())
    check("[D1] F and Q equal the continuous OU integral", worst_F < 1e-12 and worst_Q < 1e-7,
          f"F err {worst_F:.1e}, Q rel err {worst_Q:.1e}")
    check("[D1] tau = 0 is the constant-velocity triple exactly",
          terms(0.04, 0.0) == (1.0, 0.04, 0.04 ** 3 / 3.0, 0.04 ** 2 / 2.0, 0.04))


def toy_state(n=7, seed=0):
    from estimator.manifold import so3_exp
    from estimator.state import GaussianMotionState
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)  # noqa: E731
    return GaussianMotionState(mu=r(n, 3), R=so3_exp(r(n, 3)), v=r(n, 3), omega=0.3 * r(n, 3),
                               colors=torch.rand(n, 3, dtype=torch.float64), opacity=torch.ones(n, 1, dtype=torch.float64),
                               scales=0.01 * torch.ones(n, 3, dtype=torch.float64))


def test_phase1(check):
    from estimator.covariance import make_process_noise
    from estimator.manifold import so3_exp, so3_log
    from estimator.process import error_state_transition, process_predict
    from estimator.velocity_decay import terms
    st, dt = toy_state(), 0.04
    a, b = process_predict(st, dt), process_predict(st, dt, tau=0.0)
    same = all(torch.equal(getattr(a, k), getattr(b, k)) for k in ("mu", "R", "v", "omega"))
    same &= torch.equal(error_state_transition(st, dt), error_state_transition(st, dt, tau=0.0))
    same &= torch.equal(make_process_noise(7, dt, 1e-3, 1e-3, 0.1, 0.1, dtype=torch.float64),
                        make_process_noise(7, dt, 1e-3, 1e-3, 0.1, 0.1, dtype=torch.float64, tau=0.0))
    check("[D2] tau = 0: predict, F and Q bitwise unchanged", same)
    tau = 0.5
    phi, h, cxx, cxv, cvv = terms(dt, tau)
    p = process_predict(st, dt, tau=tau)
    ok = torch.allclose(p.v, phi * st.v) and torch.allclose(p.omega, phi * st.omega) \
        and torch.allclose(p.mu, st.mu + h * st.v) and torch.allclose(p.R, so3_exp(h * st.omega) @ st.R)
    check("[D2] tau > 0: v, omega decay by phi, the pose integrates over h", ok, f"phi {phi:.4f}, h/dt {h / dt:.4f}")
    Q = make_process_noise(7, dt, 1e-3, 2e-3, 0.1, 0.2, dtype=torch.float64, tau=tau)
    ok = math.isclose(float(Q[0, 0, 0]), 1e-6 * dt + 0.01 * cxx, rel_tol=1e-12) \
        and math.isclose(float(Q[0, 3, 9]), 0.04 * cxv, rel_tol=1e-12) \
        and math.isclose(float(Q[0, 6, 6]), 0.01 * cvv, rel_tol=1e-12)
    check("[D2] Q blocks are s^2 c_ab plus the unchanged pose random walk", ok)
    # F = d predict / d (left error), by central differences on one Gaussian
    F = error_state_transition(st, dt, tau=tau)[0]
    eps, J = 1e-6, torch.zeros(12, 12, dtype=torch.float64)
    base = process_predict(st, dt, tau=tau)

    def err(x):
        return torch.cat([x.mu[0] - base.mu[0], so3_log(x.R[0:1] @ base.R[0:1].transpose(1, 2))[0],
                          x.v[0] - base.v[0], x.omega[0] - base.omega[0]])

    import dataclasses
    for k in range(12):
        cols = []
        for sgn in (1, -1):
            d = torch.zeros(12, dtype=torch.float64)
            d[k] = sgn * eps
            mu, R, v, w = st.mu.clone(), st.R.clone(), st.v.clone(), st.omega.clone()
            mu[0] += d[:3]
            R[0] = so3_exp(d[3:6].reshape(1, 3))[0] @ R[0]
            v[0] += d[6:9]
            w[0] += d[9:]
            cols.append(err(process_predict(dataclasses.replace(st, mu=mu, R=R, v=v, omega=w), dt, tau=tau)))
        J[:, k] = (cols[0] - cols[1]) / (2 * eps)
    e = float((J - F).abs().max())
    check("[D2] F is the Jacobian of the decaying prediction", e < 1e-6, f"max err {e:.1e}")


def test_rigid(check):
    from estimator.rigid_part_runtime import RigidIEKFConfig, RigidState
    from estimator.rigid_part_se3 import RigidPart
    T, dt, tau = 200, 0.04, 0.5
    out = {}
    for name, t in (("cv", 0.0), ("ou", tau)):
        part = RigidPart([0, 1, 2], num_frames=T)
        part.v[0] = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        st = RigidState(part, RigidIEKFConfig(velocity_tau=t))
        if name == "cv":
            F0, Q0 = st.process_matrix(0, dt), st.process_noise(dt)
        for k in range(T - 1):
            st.predict(k, dt)
        out[name] = (float(st.parts[0].p[-1].norm()), st)
    # the pre-decay formulas, verbatim: Ad_E, dt J_l(dt xi), identity on xi; the dt^3/3, dt^2/2, dt triple
    from estimator.joint_manifold import se3_adjoint, se3_exp, se3_left_jacobian
    xi = torch.tensor([0.1, 0, 0, 0, 0, 0], dtype=torch.float64) * dt
    E_R, E_t = se3_exp(xi.reshape(1, 6))
    ref = torch.eye(12, dtype=torch.float64)
    ref[:6, :6] = se3_adjoint(E_R, E_t)[0]
    ref[:6, 6:] = dt * se3_left_jacobian(xi.reshape(1, 6))[0]
    qd = torch.tensor([dt ** 3 / 3.0] * 6 + [dt] * 6, dtype=torch.float64)
    check("[D3] tau = 0: F and Q are the constant-velocity matrices bitwise",
          torch.equal(F0, ref) and torch.equal(Q0.diagonal(), qd) and torch.equal(Q0[:6, 6:].diagonal(),
                                                                                  torch.full((6,), dt ** 2 / 2.0, dtype=torch.float64)))
    cv, ou = out["cv"][0], out["ou"][0]
    check("[D3] no measurement: constant velocity travels v0 t, mean-reverting stops near v0 tau",
          abs(cv - 0.1 * dt * (T - 1)) < 1e-9 and abs(ou - 0.1 * tau * (1 - math.exp(-dt * (T - 1) / tau))) < 1e-9,
          f"cv {cv:.3f} m, ou {ou:.4f} m (v0 tau = {0.1 * tau:.3f})")


def test_phase3(check):
    from estimator.joint_process import predict, process_matrix, process_noise
    from estimator.velocity_decay import terms
    T, dt, tau = 100, 0.04, 0.5
    res = {}
    for t in (0.0, tau):
        part = types.SimpleNamespace(q=torch.zeros(T, dtype=torch.float64), qdot=torch.zeros(T, dtype=torch.float64),
                                     num_frames=T)
        part.qdot[0] = 1.0
        st = types.SimpleNamespace(dim=2, num_parts=1, error_dim=2, parts=[part],
                                   covariance=torch.eye(2, dtype=torch.float64))
        for k in range(T - 1):
            predict(st, k, dt, 1.0, tau=t)
        res[t] = (float(part.q[-1]), st)
    st = res[0.0][1]
    F, Q = process_matrix(st, dt), process_noise(st, dt, 1.0)
    check("[D4] tau = 0: the rate model is unchanged",
          torch.equal(F, torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float64))
          and torch.equal(Q, torch.tensor([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]], dtype=torch.float64)))
    phi, h, *_ = terms(dt, tau)
    F = process_matrix(st, dt, tau)
    check("[D4] tau > 0: F = [[1, h], [0, phi]] and q stops near qdot0 tau",
          torch.allclose(F, torch.tensor([[1.0, h], [0.0, phi]], dtype=torch.float64))
          and abs(res[tau][0] - tau * (1 - math.exp(-dt * (T - 1) / tau))) < 1e-9,
          f"q_end cv {res[0.0][0]:.2f}, ou {res[tau][0]:.3f}")


def camera(H=48, W=64, tx=0.5, ty=0.375, R=None, t=None):
    V = np.eye(4)
    if R is not None:
        V[:3, :3], V[3, :3] = R.T, t                    # row-vector: p_cam = p_w @ V
    return types.SimpleNamespace(viewmatrix=torch.tensor(V), tanfovx=tx, tanfovy=ty, image_height=H, image_width=W)


def test_projection(check):
    from cross_camera_registration import backproject
    from scipy.spatial.transform import Rotation

    from estimator.free_space_support import CONSISTENT, HIDDEN, VIOLATED, classify
    rng = np.random.default_rng(0)
    cam = camera(R=Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix(), t=np.array([0.05, -0.02, 1.5]))
    D = rng.uniform(0.5, 2.0, (48, 64))
    valid = np.ones_like(D, bool)
    P = torch.tensor(backproject(cam, D, valid))
    Dt, Vt = torch.tensor(D), torch.tensor(valid)
    c0 = classify(P, cam, Dt, Vt, 0.01)
    # move along the camera ray: closer -> violated, farther -> hidden
    Vinv = np.linalg.inv(cam.viewmatrix.numpy())
    org = torch.tensor(Vinv[3, :3])
    ray = (P - org) / (P - org).norm(dim=1, keepdim=True)
    zdir = torch.tensor(cam.viewmatrix.numpy()[:3, 2])           # camera z axis in world (row-vector V)
    step = 0.05 / (ray @ zdir).abs().unsqueeze(1)                # 5 cm in camera depth
    c1, c2 = classify(P - step * ray, cam, Dt, Vt, 0.01), classify(P + step * ray, cam, Dt, Vt, 0.01)
    check("[F1] back-projected depth is consistent; 5 cm closer violated, 5 cm farther hidden",
          bool((c0 == CONSISTENT).all() and (c1 == VIOLATED).all() and (c2 == HIDDEN).all()),
          f"{float((c0 == CONSISTENT).float().mean()):.3f} {float((c1 == VIOLATED).float().mean()):.3f} "
          f"{float((c2 == HIDDEN).float().mean()):.3f}")


def door_scene(T=40, H=60, W=80):
    """Camera at the origin looking +z. Wall z = 3; static interior face z = 2 (|x|, |y| <= 0.3);
    door hinged at (-0.35, 0, 1.5) about +y, 0.7 wide, |y| <= 0.3, swinging from 60 deg open
    (toward the camera) to closed (z = 1.5, covering the interior)."""
    cam = camera(H=H, W=W, tx=0.5, ty=0.375)
    hinge = np.array([-0.35, 0.0, 1.5])
    thetas = np.radians(np.linspace(60.0, 0.0, T))
    u = (2 * np.arange(W) + 1) / W - 1
    v = (2 * np.arange(H) + 1) / H - 1
    dx, dy = np.meshgrid(u * cam.tanfovx, v * cam.tanfovy)
    depths = []
    for th in thetas:
        e = np.array([np.cos(th), 0.0, -np.sin(th)])            # along the door from the hinge
        n = np.array([np.sin(th), 0.0, np.cos(th)])             # door normal
        d = np.stack([dx, dy, np.ones_like(dx)], -1)
        z = np.full((H, W), 3.0)
        ti = 2.0                                                # interior plane z = 2
        inside = (np.abs(dx * ti) <= 0.3) & (np.abs(dy * ti) <= 0.3)
        z = np.where(inside, 2.0, z)
        t = (hinge @ n) / (d @ n)
        hit = d * t[..., None]
        s = (hit - hinge) @ e
        on = (t > 0) & (s >= 0) & (s <= 0.7) & (np.abs(hit[..., 1]) <= 0.3)
        z = np.where(on & (t < z), t, z)                        # z of the hit = t (d_z = 1)
        depths.append(z)
    # Gaussians: door at the canonical (open) pose, interior face, wall patch
    s, y = np.meshgrid(np.linspace(0.02, 0.68, 12), np.linspace(-0.28, 0.28, 8))
    e0 = np.array([np.cos(thetas[0]), 0.0, -np.sin(thetas[0])])
    door = hinge + s.reshape(-1, 1) * e0 + np.c_[np.zeros(s.size), y.reshape(-1), np.zeros(s.size)]
    ix, iy = np.meshgrid(np.linspace(-0.28, 0.28, 10), np.linspace(-0.28, 0.28, 8))
    interior = np.c_[ix.reshape(-1), iy.reshape(-1), np.full(ix.size, 2.0)]
    wx, wy = np.meshgrid(np.linspace(-1.2, 1.2, 10), np.linspace(-0.9, 0.9, 6))
    wall = np.c_[wx.reshape(-1), wy.reshape(-1), np.full(wx.size, 3.0)]
    X = np.concatenate([door, interior, wall])
    R, p = [], []
    for th in thetas:
        a = th - thetas[0]                                      # closing: rotation about +y by -(60 - th)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        R.append(Ry)
        p.append(hinge - Ry @ hinge)
    obs = [[(None, torch.tensor(z), torch.ones_like(torch.tensor(z), dtype=torch.bool))] for z in depths]
    return cam, X, torch.tensor(np.stack(R)), torch.tensor(np.stack(p)), obs, (len(door), len(interior), len(wall))


def test_door(check):
    from estimator.free_space_support import assign, evidence
    cam, X, R, p, obs, (nd, ni, nw) = door_scene()
    door = torch.zeros(len(X), dtype=torch.bool)
    door[:nd] = True
    interior = torch.zeros_like(door)
    interior[nd:nd + ni] = True
    wall = ~(door | interior)
    # the coherence support: the hidden interior plus the hinge half of the door
    s_coord = torch.tensor(np.linspace(0.02, 0.68, 12)).repeat(8)
    previous = interior | (door & (torch.cat([s_coord, torch.zeros(ni + nw, dtype=torch.float64)]) < 0.35))
    ev = evidence(torch.tensor(X), R, p, obs, [cam], tau=0.02)
    dec = assign(ev, previous, min_members=10)
    m, lab = dec["members"], dec["label"]
    decided_door = door & (lab != 2)                            # the hinge column barely moves
    check("[F2] every door Gaussian that moves is in the refined support",
          bool(m[decided_door].all()), f"{int(m[decided_door].sum())}/{int(decided_door.sum())}")
    check("[F2] the half of the door the coherence support missed was added",
          int((m & door & ~previous).sum()) >= 0.9 * int((door & ~previous & (lab != 2)).sum()))
    check("[F2] the hidden interior is dropped (no evidence of moving -> static)",
          not bool(m[interior].any()), f"{int(m[interior].sum())}/{ni} kept; labels {lab[interior].unique().tolist()}")
    check("[F2] the wall stays static", not bool(m[wall].any()))
    check("[F2] the refinement was accepted", dec["accepted"])


def main() -> int:
    check, st = check_factory()
    print("[D] mean-reverting velocity")
    test_terms(check)
    test_phase1(check)
    test_rigid(check)
    test_phase3(check)
    print("\n[F] free-space support refinement")
    test_projection(check)
    test_door(check)
    return finish(st)


if __name__ == "__main__":
    raise SystemExit(main())
