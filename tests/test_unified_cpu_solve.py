"""`solve_on_cpu`: the full-window vectors in host memory, same arithmetic.

At 45,920 Gaussians x 60 frames x 12 every full-window float64 vector is
264 MB, and PCG plus the preconditioner hold about eight of them. On an 8 GiB
card shared with other jobs the real Washer joint run went OOM at 3.4 GiB.
`IEKSConfig.solve_on_cpu` keeps those vectors on the host and sends one
frame's tangent to the GPU per JVP/VJP. This file checks it is the same
solver: operator action, right-hand side and an accepted trajectory agree
with the GPU solve to within a MEASURED resident-vs-resident floor (the
rasterizer's Fisher path is not bitwise reproducible), and the anchors still
hold.

    python tests/test_unified_cpu_solve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CHECKS = 0
FAILURES: list[str] = []


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")
    else:
        print(f"  ok    {label}  {detail}")


def main() -> int:
    if not torch.cuda.is_available():
        print("cpu solve NOT_RUN: CUDA unavailable")
        return 0
    from bidirectional_iekf.ieks import IEKSConfig, Trajectory, UnifiedFreeIEKS
    from bidirectional_iekf.observations import FreeRGBDObservation
    from estimator.manifold import so3_exp
    from estimator.renderer_bridge import build_linearization
    from estimator.state import GaussianMotionState
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

    N, T, dt = 6, 5, 0.1
    A, B = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
    cams = make_cameras(n_cam=2, H=80, W=80, radius=2.2)

    def make(mu, v):
        return GaussianMotionState(
            mu=mu.float().cuda(), R=so3_exp(torch.zeros(N, 3)).cuda(),
            v=v.float().cuda(), omega=torch.zeros(N, 3).cuda(),
            colors=torch.rand(N, 3, generator=torch.Generator().manual_seed(1)).cuda(),
            opacity=torch.full((N, 1), 0.8).cuda(),
            scales=torch.tensor([[.12, .08, .06]]).expand(N, 3).contiguous().cuda())

    mu0 = torch.tensor([[-.2, .1, 0], [.1, -.1, .02], [0, .2, .05],
                        [-.05, .03, .1], [.15, .12, -.08], [-.1, -.15, .04]])
    vt = torch.tensor([[.05, 0, 0]]).expand(N, 3).contiguous()
    truth = [make(mu0 + vt * dt * t, vt) for t in range(T)]
    obs = [[(r.clone(), d.clone(), torch.ones_like(d, dtype=torch.bool))
            for r, d in build_linearization(truth[t], cams).predicted()] for t in range(T)]
    Q = torch.diag(torch.tensor([1e4] * 6 + [1e2] * 6, dtype=torch.float64)
                   ).expand(N, 12, 12).contiguous()
    g = torch.Generator().manual_seed(2)
    warm = Trajectory(states=[make(truth[t].mu.cpu() + 0.01 * torch.randn(N, 3, generator=g),
                                   torch.zeros(N, 3)) for t in range(T)],
                      frame_ids=list(range(T)), dt=dt)

    def ieks(cpu, **kw):
        return UnifiedFreeIEKS(
            FreeRGBDObservation(cams, obs, frame_ids=list(range(T))), Q,
            config=IEKSConfig(source_pose_policy="fixed_conditional",
                              anchor_groups=((0, A), (T - 1, B)), solve_on_cpu=cpu,
                              stream_frames=True, process_factors_on_cpu=True,
                              diagonal_preconditioner=True, verbose=False, **kw))

    def rel(a, b):
        return float((a.cpu() - b.cpu()).norm()) / max(float(a.norm()), 1e-30)

    op_g, _ = ieks(False).build_system(warm, 1e-4)
    op_g2, _ = ieks(False).build_system(warm, 1e-4)
    op_c, _ = ieks(True).build_system(warm, 1e-4)
    u = torch.randn(T, N, 12, generator=g, dtype=torch.float64)
    floor_mv = rel(op_g.information_matvec(u.cuda()), op_g2.information_matvec(u.cuda()))
    floor_b = rel(op_g.rhs(), op_g2.rhs())
    print(f"       GPU-vs-GPU floor: matvec {floor_mv:.3e}, rhs {floor_b:.3e}")
    check("the CPU solve's vectors really live on the host",
          op_c.zeros().device.type == "cpu" and op_g.zeros().device.type == "cuda")
    d_mv = rel(op_g.information_matvec(u.cuda()), op_c.information_matvec(u))
    d_b = rel(op_g.rhs(), op_c.rhs())
    check("same operator action", d_mv <= max(4 * floor_mv, 1e-12),
          f"{d_mv:.3e} against a {floor_mv:.3e} floor")
    check("same right-hand side", d_b <= max(4 * floor_b, 1e-12),
          f"{d_b:.3e} against a {floor_b:.3e} floor")

    rg = ieks(False, max_outer=8, damping=1e-4, pcg_max_iters=80, pcg_tol=1e-10).run(warm)
    rg2 = ieks(False, max_outer=8, damping=1e-4, pcg_max_iters=80, pcg_tol=1e-10).run(warm)
    rc = ieks(True, max_outer=8, damping=1e-4, pcg_max_iters=80, pcg_tol=1e-10).run(warm)

    def worst(a, b):
        return max(float((a["trajectory"].states[t].mu - b["trajectory"].states[t].mu)
                         .abs().max()) for t in range(T))
    ctrl, dcpu = worst(rg, rg2), worst(rg, rc)
    print(f"       GPU {rg['status']} E {rg['energy']['E_total']:.4e}; CPU {rc['status']} "
          f"E {rc['energy']['E_total']:.4e}; states differ {dcpu:.3e} (GPU control {ctrl:.3e})")
    check("same accepted trajectory, within the GPU control",
          dcpu <= max(4 * ctrl, 1e-6), f"{dcpu:.3e} vs {ctrl:.3e}")
    st = rc["trajectory"].states
    check("anchors still exact under the CPU solve",
          float((st[0].mu[A] - warm.states[0].mu[A]).abs().max()) == 0.0
          and float((st[T - 1].mu[B] - warm.states[T - 1].mu[B]).abs().max()) == 0.0)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
