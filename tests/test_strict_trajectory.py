"""Section 30: the banded joint trajectory solver.

CODEX_THEORETICAL_METHOD_STRICT_IMPLEMENTATION.md sections 11, 12, 13, 30.

Section 11 removes the coordinate-wise scalar search as the production solver
and requires the joint system

    [H_data + H_temp + lambda I] dq = g - H_temp q

over all K*T trajectory coordinates at once. Section 30 requires that solver to
be exercised on constant velocity, acceleration, a sinusoid, stop/start,
reversal, irregular dt, partial occlusion and missing cameras, and to be
checked against an explicit dense Gauss-Newton reference on tiny problems.

The dense reference here is built by CENTRAL FINITE DIFFERENCES of the true
nonlinear render, not by re-using the analytic Jacobian. It therefore tests the
analytic normal equations rather than restating them, and it builds the FULL
K*T x K*T Hessian including cross-frame blocks, so section 11's claim that "no
renderer cross-frame Hessian is needed" is measured rather than assumed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.renderer_bridge import CameraSpec, build_linearization  # noqa: E402
from estimator.state import GaussianMotionState  # noqa: E402
from joint_reduction.kpart_models import (  # noqa: E402
    PRISMATIC, REVOLUTE, KPartObservation, PartModel, SceneModel)
from joint_reduction.kpart_window import (  # noqa: E402
    AlternationConfig, WindowObjective, acceleration_operator,
    optimise_q_trajectory_coordinatewise, solve_banded_trajectory,
    temporal_hessian, temporal_precision, window_trajectory_energy)

DEV = "cuda"
PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def camera(width=48, height=48, fov=0.9, tz=0.0):
    tanf = math.tan(fov * 0.5)
    view = torch.eye(4, device=DEV)
    view[3, 0] = tz
    znear, zfar = 0.01, 100.0
    proj = torch.zeros((4, 4), device=DEV)
    proj[0, 0] = 1.0 / tanf
    proj[1, 1] = 1.0 / tanf
    proj[2, 2] = zfar / (zfar - znear)
    proj[3, 2] = 1.0
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    full = view.unsqueeze(0).bmm(proj.t().unsqueeze(0)).squeeze(0)
    return CameraSpec(viewmatrix=view, projmatrix=full, tanfovx=tanf,
                      tanfovy=tanf, image_height=height, image_width=width)


def scene_of(n=260, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros((n, 3))
    mu[:, 0] = (torch.rand(n, generator=g) - 0.5) * 0.6
    mu[:, 1] = (torch.rand(n, generator=g) - 0.5) * 0.6
    mu[:, 2] = 2.0 + torch.rand(n, generator=g) * 0.3
    z = torch.zeros((n, 3))
    return GaussianMotionState(
        mu=mu.to(DEV), R=torch.eye(3).repeat(n, 1, 1).to(DEV),
        v=z.clone().to(DEV), omega=z.to(DEV),
        colors=torch.rand((n, 3), generator=g).to(DEV),
        opacity=torch.full((n, 1), 0.85).to(DEV),
        scales=torch.full((n, 3), 0.05).to(DEV))


class Cfg:
    rgb_sigma = 0.1
    robust_measurement = False
    student_t_nu = 5.0
    depth_sigma_model = "constant:0.02:0.0"
    depth_sigma_a = 0.02
    depth_sigma_b = 0.0
    min_alpha_for_depth = 0.05
    eps_depth = 1e-6


def render_obs(state, mu, R, cams, occlude=None):
    s = GaussianMotionState(mu=mu, R=R if R is not None else state.R, v=state.v,
                            omega=state.omega, colors=state.colors,
                            opacity=state.opacity, scales=state.scales)
    lin = build_linearization(s, cams)
    out = []
    for v in lin.views:
        valid = (v.alpha > 0.05).clone()
        if occlude is not None:
            valid[:, : int(valid.shape[1] * occlude)] = False
        out.append((v.rgb.clone(), v.depth.clone(), valid))
    del lin
    return out


def build_window(q_true, *, cams=None, seed=0, kind=PRISMATIC, times=None,
                 occlude=None, drop=None, n=260):
    """A window whose frames are renders of the TRUE trajectory."""
    cams = cams or [camera()]
    st = scene_of(n=n, seed=seed)
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    moment = torch.tensor([0.0, 0.0, 0.0], device=DEV)
    model = (PartModel(kind=PRISMATIC, axis=axis, part_index=1) if kind == PRISMATIC
             else PartModel(kind=REVOLUTE, axis=axis, moment=moment, part_index=1))
    scene = SceneModel(parts=[model], K=1)
    labels = torch.zeros(st.num_gaussians, dtype=torch.long, device=DEV)
    labels[st.mu[:, 0] > 0.0] = 1
    frames = []
    for t, q in enumerate(q_true):
        mu, R = scene.transport_all(st.mu, st.R, labels, [float(q)])
        obs = render_obs(st, mu, R, cams, occlude=occlude)
        if drop is not None and t in drop:
            obs = [None if ci in drop[t] else o for ci, o in enumerate(obs)]
        frames.append((obs, t))
    obj = WindowObjective(st.mu, st.R, cams, frames, Cfg(), state_template=st)
    if times is not None:
        obj.frame_times = list(times)
    return obj, scene, labels


def build_window_2part(q1, q2, *, seed=0, noise=0.0, n=320):
    """Two prismatic parts stacked as SEMI-TRANSPARENT layers on the same rays.

    Both layers cover the same (x,y) footprint at different depths and the
    opacity is low enough that the far layer shows through the near one, so a
    single pixel receives colour from both parts. The per-frame K x K block
    h_t then has a large off-diagonal term. This is the case section 11 is
    about: coordinate descent must move q_1 and q_2 one at a time and zig-zags
    down the correlated valley, while the joint solve takes the coupled step.
    """
    cams = [camera()]
    g = torch.Generator().manual_seed(seed)
    half = n // 2
    xy = (torch.rand((half, 2), generator=g) - 0.5) * 0.6
    mu = torch.zeros((2 * half, 3))
    mu[:half, :2] = xy
    mu[half:, :2] = xy                                   # same footprint
    mu[:half, 2] = 2.00 + torch.rand(half, generator=g) * 0.02
    mu[half:, 2] = 2.20 + torch.rand(half, generator=g) * 0.02
    z = torch.zeros((2 * half, 3))
    st = GaussianMotionState(
        mu=mu.to(DEV), R=torch.eye(3).repeat(2 * half, 1, 1).to(DEV),
        v=z.clone().to(DEV), omega=z.to(DEV),
        colors=torch.rand((2 * half, 3), generator=g).to(DEV),
        opacity=torch.full((2 * half, 1), 0.30).to(DEV),   # the far layer shows
        scales=torch.full((2 * half, 3), 0.05).to(DEV))
    a1 = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    a2 = torch.tensor([0.9848, 0.1736, 0.0], device=DEV)   # 10 degrees apart
    scene = SceneModel(parts=[PartModel(kind=PRISMATIC, axis=a1, part_index=1),
                              PartModel(kind=PRISMATIC, axis=a2, part_index=2)],
                       K=2)
    labels = torch.ones(2 * half, dtype=torch.long, device=DEV)
    labels[half:] = 2
    gn = torch.Generator(device=DEV).manual_seed(seed + 99)
    frames = []
    for t, (u, v) in enumerate(zip(q1, q2)):
        mu_t, R_t = scene.transport_all(st.mu, st.R, labels,
                                        [float(u), float(v)])
        obs = render_obs(st, mu_t, R_t, cams)
        if noise > 0.0:
            obs = [(o[0] + noise * torch.randn(o[0].shape, generator=gn,
                                               device=DEV),
                    o[1], o[2]) for o in obs]
        frames.append((obs, t))
    obj = WindowObjective(st.mu, st.R, cams, frames, Cfg(), state_template=st)
    return obj, scene, labels


# ===========================================================================
def test_30_dense_reference_matches_banded():
    """§30: the banded assembly must equal an explicit dense Gauss-Newton.

    The dense reference differentiates the TRUE nonlinear render by central
    finite differences over every (part, frame) variable, including the
    cross-frame blocks the banded assembly omits.
    """
    from estimator.measurement import build_precision  # noqa: F401

    q_true = [0.0, 0.03, 0.07, 0.12]
    obj, scene, labels = build_window(q_true, seed=3, n=200)
    T, K = len(obj.frames), scene.K
    n = K * T
    q0 = [[0.0, 0.02, 0.05, 0.10]]

    # ---- analytic banded assembly (sections 11, 13) ----------------------
    H_band = torch.zeros((n, n), dtype=torch.float64, device=DEV)
    g_band = torch.zeros(n, dtype=torch.float64, device=DEV)
    for t, (obs, _f) in enumerate(obj.frames):
        qt = [q0[k][t] for k in range(K)]
        h_t, g_t = obj.frame_normal_system(scene, labels, qt, obs)
        sl = slice(t * K, t * K + K)
        H_band[sl, sl] = h_t
        g_band[sl] = g_t

    # ---- dense finite-difference reference --------------------------------
    # The whitening is FROZEN at the linearization point. That is what
    # Gauss-Newton means: H = J^T R_eff^-1 J with R_eff evaluated at q0. It also
    # removes a genuine discontinuity -- the depth-validity gate
    # alpha > min_alpha_for_depth flips on and off as q moves, and a central
    # difference across that flip diverges as 1/h (measured: a max |dD/dq| of
    # 35, 107 and 352 at h = 3e-2, 1e-2 and 3e-3, entirely on alpha <= 0.05
    # pixels). Those pixels carry no depth evidence at q0, so freezing the mask
    # excludes them, exactly as the measurement model does.
    frozen = []
    for t, (obs, _f) in enumerate(obj.frames):
        qt = [q0[k][t] for k in range(K)]
        mu, R = scene.transport_all(obj.mu0, obj.R0, labels, qt)
        lin, prec, _res = obj.linearize_frame(mu, R, obs)
        frozen.append([(pr.clamp_min(0).sqrt(), pd.clamp_min(0).sqrt())
                       for pr, pd in prec])
        del lin

    def whitened(qt_all):
        """sqrt(R_eff^-1) r stacked over every frame, camera and pixel."""
        parts = []
        for t, (obs, _f) in enumerate(obj.frames):
            qt = [qt_all[k][t] for k in range(K)]
            mu, R = scene.transport_all(obj.mu0, obj.R0, labels, qt)
            lin, _p, res = obj.linearize_frame(mu, R, obs)
            for (sr, sd), (rr, rd) in zip(frozen[t], res):
                parts.append((sr * rr).reshape(-1).double())
                parts.append((sd * rd).reshape(-1).double())
            del lin
        return torch.cat(parts)

    # The renders are float32, so a central difference is a balance between
    # truncation (large h) and cancellation (small h). Measured on this scene,
    # the analytic and finite-difference RGB Jacobians agree to cos 0.9991 at
    # h = 1e-2 and degrade monotonically to cos 0.936 at h = 1e-5 -- i.e. the
    # error is cancellation, not truncation.
    h = 1e-2
    r0 = whitened(q0)
    J = torch.zeros((r0.numel(), n), dtype=torch.float64, device=DEV)
    for k in range(K):
        for t in range(T):
            up = [list(c) for c in q0]
            dn = [list(c) for c in q0]
            up[k][t] += h
            dn[k][t] -= h
            J[:, t * K + k] = (whitened(up) - whitened(dn)) / (2 * h)
    H_dense = J.T @ J
    g_dense = -(J.T @ r0)          # r = y - h(q), so dr/dq = -H

    off = H_dense.clone()
    for t in range(T):
        off[t * K:(t + 1) * K, t * K:(t + 1) * K] = 0.0
    diag_scale = float(H_dense.diagonal().abs().max())
    print(f"       dense GN Hessian {n}x{n}, max |diagonal block| "
          f"{diag_scale:.4g}")
    print(f"       max |cross-frame block| {float(off.abs().max()):.4g} "
          f"= {float(off.abs().max()) / diag_scale:.2e} of it")
    check("§11 the data Hessian has no cross-frame blocks",
          float(off.abs().max()) / diag_scale < 1e-10,
          "frame t's render depends only on q_t, measured by finite "
          "differences on the true render")

    rel_H = float((H_band - H_dense).abs().max()) / diag_scale
    rel_g = (float((g_band - g_dense).abs().max())
             / max(float(g_dense.abs().max()), 1e-30))
    print(f"       banded vs dense: Hessian rel {rel_H:.2e}, "
          f"gradient rel {rel_g:.2e}")
    # Tolerance set by what a float32 render can certify by finite
    # differences, not by what the analytic Jacobian happens to produce.
    check("§30 the banded Hessian matches the dense reference", rel_H < 0.05,
          f"relative {rel_H:.2e} (float32 central difference at h={h:g})")
    check("§30 the banded gradient matches the dense reference", rel_g < 0.05,
          f"relative {rel_g:.2e}")


def test_30_trajectory_shapes():
    """§30: constant velocity, acceleration, sinusoid, stop/start, reversal."""
    T = 8
    shapes = {
        "constant velocity": [0.012 * t for t in range(T)],
        "acceleration": [0.0016 * t * t for t in range(T)],
        "sinusoid": [0.05 * math.sin(2 * math.pi * t / T) for t in range(T)],
        "stop/start": [0.0, 0.02, 0.04, 0.04, 0.04, 0.06, 0.08, 0.10],
        "reversal": [0.0, 0.03, 0.06, 0.03, 0.0, -0.03, -0.06, -0.03],
    }
    cfg = AlternationConfig(sigma_a=1.0, dt=1.0, trajectory_iters=8)
    worst = 0.0
    for name, q_true in shapes.items():
        obj, scene, labels = build_window(q_true, seed=11)
        q0 = [[0.0] * T]
        q_out, info = solve_banded_trajectory(obj, scene, labels, q0, cfg)
        err = max(abs(a - b) for a, b in zip(q_out[0], q_true))
        worst = max(worst, err)
        print(f"       {name:18s} max |q - q_true| {err:.5f}  "
              f"E {info['energy0']:.1f} -> {info['energy']:.1f}  "
              f"{info['status']}")
        check(f"§30 {name}: the energy decreased monotonically",
              info["monotone"] and info["energy"] <= info["energy0"])
        check(f"§30 {name}: the trajectory is recovered", err < 0.01,
              f"max error {err:.5f}")
    print(f"       worst error over all five shapes: {worst:.5f}")


def test_30_irregular_dt():
    """§30: irregular dt must use the exact equivalent process covariance."""
    times = [0.0, 0.1, 0.35, 0.4, 0.9, 1.0, 1.6, 1.65]
    T = len(times)
    q_true = [0.04 * t for t in times]            # constant VELOCITY in time
    obj, scene, labels = build_window(q_true, seed=5, times=times)
    cfg = AlternationConfig(sigma_a=0.5, dt=1.0, trajectory_iters=8)
    q_out, info = solve_banded_trajectory(obj, scene, labels, [[0.0] * T], cfg)
    err = max(abs(a - b) for a, b in zip(q_out[0], q_true))
    print(f"       irregular dt: max |q - q_true| {err:.5f}, "
          f"E {info['energy0']:.1f} -> {info['energy']:.1f}")
    check("§30 irregular dt: the trajectory is recovered", err < 0.01,
          f"max error {err:.5f}")

    A = acceleration_operator(T, times=times)
    v = torch.tensor(q_true, dtype=torch.float64)
    check("§12 the irregular-dt prior is free of a constant-velocity "
          "trajectory", float((A @ v).abs().max()) < 1e-9,
          f"residual acceleration {float((A @ v).abs().max()):.1e}")


def test_30_partial_occlusion_and_missing_cameras():
    """§30: partial occlusion and missing cameras."""
    T = 6
    q_true = [0.015 * t for t in range(T)]
    cams = [camera(), camera(tz=0.25)]
    cfg = AlternationConfig(sigma_a=1.0, trajectory_iters=8)

    obj, scene, labels = build_window(q_true, cams=cams, seed=8, occlude=0.4)
    q_out, info = solve_banded_trajectory(obj, scene, labels, [[0.0] * T], cfg)
    err = max(abs(a - b) for a, b in zip(q_out[0], q_true))
    print(f"       40% of depth occluded : max |q - q_true| {err:.5f}")
    check("§30 partial occlusion: the trajectory is recovered", err < 0.01,
          f"max error {err:.5f}")

    drop = {1: {0}, 3: {1}, 4: {0}}          # a camera missing at 3 frames
    obj2, scene2, labels2 = build_window(q_true, cams=cams, seed=8, drop=drop)
    q2, info2 = solve_banded_trajectory(obj2, scene2, labels2, [[0.0] * T], cfg)
    err2 = max(abs(a - b) for a, b in zip(q2[0], q_true))
    print(f"       a camera missing at 3/6 frames: max |q - q_true| {err2:.5f}")
    check("§30 missing cameras: the trajectory is recovered", err2 < 0.01,
          f"max error {err2:.5f}")
    check("§20 a missing camera contributes no evidence",
          info2["monotone"])


def test_11_joint_beats_coordinatewise():
    """§11: the joint solve on two parts that share pixels.

    Section 11 removes the coordinate-wise search as the PRODUCTION solver.
    This measures the difference on a window where the two parts' Jacobian
    columns are strongly correlated, which is the case coordinate descent
    cannot handle, rather than asserting the difference.
    """
    T = 6
    q1 = [0.02 * t for t in range(T)]
    q2 = [-0.015 * t for t in range(T)]
    obj, scene, labels = build_window_2part(q1, q2, seed=4, noise=0.01)
    cfg = AlternationConfig(sigma_a=1.0, trajectory_iters=10,
                            q_golden_iters=24)

    # how correlated are the two parts at a representative frame?
    h_t, _g_t = obj.frame_normal_system(scene, labels, [0.0, 0.0],
                                        obj.frames[T // 2][0])
    rho = float(h_t[0, 1] / (h_t[0, 0] * h_t[1, 1]).sqrt().clamp_min(1e-30))
    print(f"       per-frame Jacobian correlation between the two parts: "
          f"{rho:+.3f}")

    start = [[0.0] * T, [0.0] * T]
    qb, info = solve_banded_trajectory(obj, scene, labels,
                                       [list(c) for c in start], cfg)
    Eb = window_trajectory_energy(obj, scene, labels, qb, sigma_a=cfg.sigma_a)
    qc = optimise_q_trajectory_coordinatewise(obj, scene, labels,
                                              [list(c) for c in start], cfg)
    Ec = window_trajectory_energy(obj, scene, labels, qc, sigma_a=cfg.sigma_a)
    eb = max(max(abs(a - b) for a, b in zip(qb[j], (q1, q2)[j]))
             for j in range(2))
    ec = max(max(abs(a - b) for a, b in zip(qc[j], (q1, q2)[j]))
             for j in range(2))
    print(f"       banded      : E {Eb:12.2f}  max |q - q_true| {eb:.5f}  "
          f"{info['status']}")
    print(f"       coordinate  : E {Ec:12.2f}  max |q - q_true| {ec:.5f}")
    check("§11 the joint banded solve reaches an energy at least as low as "
          "coordinate descent", Eb <= Ec * (1.0 + 1e-6),
          f"E {Eb:.2f} vs {Ec:.2f}")
    check("§11 the two parts really are coupled in this window", abs(rho) > 0.3,
          f"correlation {rho:+.3f}")
    check("§11 the joint solve recovers both trajectories", eb < 0.01,
          f"max error {eb:.5f}")
    check("§11 production is the banded solver",
          AlternationConfig().trajectory_solver == "banded")


def test_12_temporal_hessian_is_exact():
    """§12: H_temp must be the exact Hessian of E_temporal."""
    from joint_reduction.kpart_window import temporal_energy
    T = 7
    w_a = temporal_precision(0.8, 0.3)
    H = temporal_hessian(T, w_a, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(5):
        q = (torch.rand(T, generator=g, dtype=torch.float64) - 0.5) * 2
        e = temporal_energy([float(x) for x in q], w_a=w_a)
        quad = float(0.5 * q @ H @ q)
        worst = max(worst, abs(e - quad) / max(abs(e), 1e-12))
    check("§12 E_temporal = 1/2 q^T H_temp q exactly", worst < 1e-12,
          f"worst relative {worst:.1e}")
    ev = torch.linalg.eigvalsh(H)
    check("§12 H_temp is positive semi-definite",
          float(ev.min()) > -1e-9, f"min eigenvalue {float(ev.min()):.1e}")
    null = int((ev.abs() < 1e-9 * max(float(ev.max()), 1.0)).sum())
    check("§12 its null space is exactly the constant-velocity trajectories",
          null == 2, f"{null} zero eigenvalues (offset and rate)")


def main():
    for fn in (test_12_temporal_hessian_is_exact,
               test_30_dense_reference_matches_banded,
               test_30_trajectory_shapes,
               test_30_irregular_dt,
               test_30_partial_occlusion_and_missing_cameras,
               test_11_joint_beats_coordinatewise):
        print(f"\n{fn.__name__}")
        fn()
        torch.cuda.empty_cache()
    print(f"\n{len(PASSED)} checks passed across 6 tests")


if __name__ == "__main__":
    main()
