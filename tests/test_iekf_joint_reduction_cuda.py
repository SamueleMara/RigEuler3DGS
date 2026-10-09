"""Reduced joint observation operators against the real CUDA renderer.

CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md section 35: T17 (rigid reduction
consistency), T18 (prismatic reduction), T19 (revolute reduction).

The point of these is that the SAME renderer Jacobian machinery works after the
state is reduced (section 2), and that H_pq really is sum_i H^mu_pi e (section
29.1) / sum_i [H^mu_pi (e x (mu_i - a)) + H^phi_pi e] (section 29.2).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator import CameraSpec, GaussianMotionState, build_linearization  # noqa: E402
from joint_reduction.renderer_reduction import (  # noqa: E402
    RenderJointModel,
    joint_apply_H,
    joint_apply_HT,
    joint_normal_system,
    joint_update,
)

PASSED: list[str] = []
DEV = "cuda"


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def camera(width=64, height=64, fov=0.8):
    tanf = math.tan(fov * 0.5)
    view = torch.eye(4, device=DEV)
    znear, zfar = 0.01, 100.0
    proj = torch.zeros((4, 4), device=DEV)
    proj[0, 0] = 1.0 / tanf
    proj[1, 1] = 1.0 / tanf
    proj[2, 2] = zfar / (zfar - znear)
    proj[3, 2] = 1.0
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    full = view.unsqueeze(0).bmm(proj.t().unsqueeze(0)).squeeze(0)
    return CameraSpec(viewmatrix=view, projmatrix=full, tanfovx=tanf, tanfovy=tanf,
                      image_height=height, image_width=width)


def scene(n=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros((n, 3))
    mu[:, 0] = (torch.rand(n, generator=g) - 0.5) * 0.3
    mu[:, 1] = (torch.rand(n, generator=g) - 0.5) * 0.3
    mu[:, 2] = 2.0 + torch.rand(n, generator=g) * 0.3
    R = torch.eye(3).repeat(n, 1, 1)
    scales = torch.full((n, 3), 0.09)
    scales[:, 0] = 0.14
    scales[:, 2] = 0.05
    z = torch.zeros((n, 3))
    return GaussianMotionState(
        mu=mu.to(DEV), R=R.to(DEV), v=z.to(DEV), omega=z.to(DEV),
        colors=torch.rand((n, 3), generator=g).to(DEV),
        opacity=torch.full((n, 1), 0.8).to(DEV), scales=scales.to(DEV))


def h_of(state, cams):
    lin = build_linearization(state, cams, eps_depth=1e-6)
    return lin, [(v.rgb, v.depth, v.alpha) for v in lin.views]


def fd_joint(state, cams, model, q_eps):
    """Central difference of the render with respect to the joint coordinate."""
    mu_p, R_p = model.transport(state.mu, state.R, q_eps)
    mu_m, R_m = model.transport(state.mu, state.R, -q_eps)
    sp = GaussianMotionState(mu=mu_p, R=R_p, v=state.v, omega=state.omega,
                             colors=state.colors, opacity=state.opacity,
                             scales=state.scales)
    sm = GaussianMotionState(mu=mu_m, R=R_m, v=state.v, omega=state.omega,
                             colors=state.colors, opacity=state.opacity,
                             scales=state.scales)
    _, pp = h_of(sp, cams)
    _, pm = h_of(sm, cams)
    return [((a[0] - b[0]) / (2 * q_eps), (a[1] - b[1]) / (2 * q_eps),
             a[2], b[2]) for a, b in zip(pp, pm)]


# A joint perturbation moves EVERY member Gaussian at once, so the band of
# pixels sitting on some splat's hard radius / alpha<1/255 cull is far wider
# than for a single Gaussian. Those pixels jump by O(1) however small dq is
# (section 26's piecewise smoothness), so the comparison is made well inside
# the splats. Measured on this scene the error at threshold 0.05 is 5e-2 and at
# 0.30 it is 2.5e-5 -- the difference is entirely the fringe, not the Jacobian.
INTERIOR_ALPHA = 0.30


def interior_rel(fd_rgb, an_rgb, a0, ap, am, thr=INTERIOR_ALPHA):
    m = ((a0 > thr) & (ap > thr) & (am > thr)).unsqueeze(0).expand_as(fd_rgb)
    if int(m.sum()) == 0:
        return float("nan")
    return float((fd_rgb - an_rgb)[m].norm() / fd_rgb[m].norm().clamp_min(1e-20))


def test_T18_prismatic_reduction():
    """T18: known prismatic axis -- H_pq matches the renderer, and q is recovered."""
    cams = [camera()]
    st = scene(n=5, seed=3)
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    model = RenderJointModel(kind="prismatic", axis=axis)
    lin, prim = h_of(st, cams)
    a0 = prim[0][2]

    dq = torch.tensor([1.0], device=DEV)
    hv = joint_apply_H(lin, model, st.mu, dq)
    best = float("inf")
    for eps in (3e-3, 1e-3, 3e-4):
        fd = fd_joint(st, cams, model, eps)
        rel = interior_rel(fd[0][0], hv[0][0], a0, fd[0][2], fd[0][3])
        best = min(best, rel)
        print(f"       eps={eps:.0e}  interior rel {rel:.3e}")
    check("T18 H_pq matches the renderer's response to dq", best < 2e-3,
          f"best rel {best:.3e}")

    # Adjoint identity in the reduced coordinate.
    torch.manual_seed(5)
    z = [(torch.randn_like(hv[0][0]), torch.randn_like(hv[0][1]))]
    lhs = float((hv[0][0] * z[0][0]).sum() + (hv[0][1] * z[0][1]).sum())
    rhs = float(joint_apply_HT(lin, model, st.mu, z)[0])
    check("T18 reduced adjoint identity", abs(lhs - rhs) / max(abs(lhs), 1e-20) < 1e-4,
          f"{lhs:.6e} vs {rhs:.6e}")

    # Recover a known q from a rendered measurement (section 30's tiny solve).
    q_true = 0.05
    mu_t, R_t = model.transport(st.mu, st.R, q_true)
    target = GaussianMotionState(mu=mu_t, R=R_t, v=st.v, omega=st.omega,
                                 colors=st.colors, opacity=st.opacity, scales=st.scales)
    _, gt = h_of(target, cams)
    q = 0.0
    for _ in range(12):
        mu_q, R_q = model.transport(st.mu, st.R, q)
        cur = GaussianMotionState(mu=mu_q, R=R_q, v=st.v, omega=st.omega,
                                  colors=st.colors, opacity=st.opacity, scales=st.scales)
        lin_q, pr = h_of(cur, cams)
        res = [(gt[0][0] - pr[0][0], gt[0][1] - pr[0][1])]
        prec = [(torch.ones_like(res[0][0]),
                 (pr[0][2] > 0.05).to(res[0][1].dtype))]
        dq_step, info = joint_update(lin_q, model, mu_q, res, prec)
        q += float(dq_step[0])
    check("T18 the joint coordinate q is recovered", abs(q - q_true) < 5e-3,
          f"q={q:.5f} vs {q_true}")
    check("T18 the reduced normal system is 1x1", info["information"].shape == (1, 1))


def test_T19_revolute_reduction():
    """T19: known revolute axis and point -- H_pq matches, centers AND
    orientations are updated (section 29.2)."""
    cams = [camera()]
    st = scene(n=5, seed=11)
    axis = torch.tensor([0.0, 1.0, 0.0], device=DEV)
    point = torch.tensor([0.0, 0.0, 2.15], device=DEV)
    model = RenderJointModel(kind="revolute", axis=axis, point=point)
    lin, prim = h_of(st, cams)
    a0 = prim[0][2]

    dq = torch.tensor([1.0], device=DEV)
    hv = joint_apply_H(lin, model, st.mu, dq)
    best = float("inf")
    for eps in (3e-3, 1e-3, 3e-4):
        fd = fd_joint(st, cams, model, eps)
        rel = interior_rel(fd[0][0], hv[0][0], a0, fd[0][2], fd[0][3])
        best = min(best, rel)
        print(f"       eps={eps:.0e}  interior rel {rel:.3e}")
    check("T19 H_pq matches the renderer's response to dq", best < 5e-3,
          f"best rel {best:.3e}")

    torch.manual_seed(13)
    z = [(torch.randn_like(hv[0][0]), torch.randn_like(hv[0][1]))]
    lhs = float((hv[0][0] * z[0][0]).sum() + (hv[0][1] * z[0][1]).sum())
    rhs = float(joint_apply_HT(lin, model, st.mu, z)[0])
    check("T19 reduced adjoint identity", abs(lhs - rhs) / max(abs(lhs), 1e-20) < 1e-4,
          f"{lhs:.6e} vs {rhs:.6e}")

    # The revolute transport must move orientations too, unlike prismatic.
    mu_q, R_q = model.transport(st.mu, st.R, 0.2)
    check("T19 revolute transport rotates the Gaussians",
          float((R_q - st.R).abs().max()) > 1e-3,
          f"max |dR| {float((R_q - st.R).abs().max()):.4f}")
    check("T19 revolute transport moves the centers",
          float((mu_q - st.mu).norm(dim=1).mean()) > 1e-3)

    q_true = 0.08
    mu_t, R_t = model.transport(st.mu, st.R, q_true)
    target = GaussianMotionState(mu=mu_t, R=R_t, v=st.v, omega=st.omega,
                                 colors=st.colors, opacity=st.opacity, scales=st.scales)
    _, gt = h_of(target, cams)
    q = 0.0
    for _ in range(12):
        mu_q, R_q = model.transport(st.mu, st.R, q)
        cur = GaussianMotionState(mu=mu_q, R=R_q, v=st.v, omega=st.omega,
                                  colors=st.colors, opacity=st.opacity, scales=st.scales)
        lin_q, pr = h_of(cur, cams)
        res = [(gt[0][0] - pr[0][0], gt[0][1] - pr[0][1])]
        prec = [(torch.ones_like(res[0][0]), (pr[0][2] > 0.05).to(res[0][1].dtype))]
        dq_step, info = joint_update(lin_q, model, mu_q, res, prec)
        q += float(dq_step[0])
    check("T19 the joint coordinate q is recovered", abs(q - q_true) < 1e-2,
          f"q={q:.5f} vs {q_true}")


def test_T17_rigid_reduction_consistency():
    """T17: a reduced update must move the scene the same way the equivalent
    full per-Gaussian pose update does."""
    cams = [camera()]
    st = scene(n=6, seed=21)
    axis = torch.tensor([0.0, 0.0, 1.0], device=DEV)
    model = RenderJointModel(kind="prismatic", axis=axis)
    lin, _ = h_of(st, cams)

    dq = torch.tensor([0.7], device=DEV)
    reduced = joint_apply_H(lin, model, st.mu, dq)
    # The equivalent full-state direction is exactly G dq.
    g = model.pose_tangent(st.mu)
    pose = (g @ dq.reshape(1, -1, 1).expand(g.shape[0], -1, 1)).squeeze(-1).float()
    full = lin.apply_H(pose)
    rel = float((reduced[0][0] - full[0][0]).abs().max()
                / full[0][0].abs().max().clamp_min(1e-20))
    check("T17 the reduced operator equals the full operator on G dq", rel < 1e-6,
          f"rel {rel:.3e}")

    # And membership masking really excludes non-members.
    member = torch.zeros(6, dtype=torch.bool, device=DEV)
    member[:3] = True
    masked = RenderJointModel(kind="prismatic", axis=axis, member=member)
    gm = masked.pose_tangent(st.mu)
    check("T17 non-member Gaussians have exactly zero joint tangent",
          float(gm[3:].abs().max()) == 0.0)
    check("T17 member Gaussians keep their tangent",
          float(gm[:3].abs().max()) > 0.0)
    mu_q, _ = masked.transport(st.mu, st.R, 0.3)
    check("T17 transport moves only members",
          float((mu_q[3:] - st.mu[3:]).abs().max()) == 0.0
          and float((mu_q[:3] - st.mu[:3]).abs().max()) > 1e-3)


def test_screw_is_rejected():
    """Section 0 rule 13: screw/helical is out of scope and must be refused."""
    try:
        RenderJointModel(kind="screw", axis=torch.tensor([0.0, 0.0, 1.0], device=DEV))
        raise SystemExit("unreachable")
    except ValueError:
        check("a screw joint is rejected, not silently accepted", True)
    try:
        RenderJointModel(kind="revolute", axis=torch.tensor([0.0, 0.0, 1.0], device=DEV))
        raise SystemExit("unreachable")
    except ValueError:
        check("a revolute joint without an axis point is rejected", True)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the reduced-joint tests")
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
