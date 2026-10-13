"""The raw-union run's two contracts: per-source anchors and one shared render.

Section 5.3 of CODEX_PRIORITY_REAL_BIDIRECTIONAL_PHASE1_MERGE_AND_VISUALIZE.md.

  * the initial group's pose is fixed at its source frame 0, and the final
    group's at its source frame T-1 -- each ONLY for its own rows. Freezing a
    whole frame, which `anchor_frames` does, would clamp the opposite source
    at a time it was never reconstructed at;
  * the rates at an anchored frame are NOT eliminated;
  * PCG keeps the eliminated coordinates at exactly zero, preconditioner
    included -- its 12x12 blocks mix pose and rate, so without care a zero
    residual on an anchor can come back nonzero;
  * the two groups are rendered TOGETHER, so the image operator couples
    them: a tangent on one source changes the other source's rows wherever
    they are co-visible.

    python tests/test_unified_union_anchors.py
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


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")
    else:
        print(f"  ok    {label}  {detail}")


def cuda_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import diff_gaussian_rasterization  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    if not cuda_ready():
        print("union anchors NOT_RUN: CUDA or diff_gaussian_rasterization unavailable.")
        return 0

    from bidirectional_iekf.ieks import IEKSConfig, Trajectory, UnifiedFreeIEKS
    from bidirectional_iekf.observations import FreeRGBDObservation
    from estimator.manifold import so3_exp
    from estimator.renderer_bridge import build_linearization
    from estimator.state import GaussianMotionState
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

    N, T, dt = 4, 5, 0.1
    A = torch.tensor([0, 1])            # initial source rows
    B = torch.tensor([2, 3])            # final source rows
    cameras = make_cameras(n_cam=2, H=80, W=80, radius=2.2)

    def make(mu, v):
        return GaussianMotionState(
            mu=mu.float().cuda(), R=so3_exp(torch.zeros(N, 3)).cuda(),
            v=v.float().cuda(), omega=torch.zeros(N, 3).cuda(),
            colors=torch.tensor([[.9, .3, .2], [.8, .4, .3],
                                 [.2, .4, .9], [.3, .6, .8]]).cuda(),
            opacity=torch.full((N, 1), 0.7).cuda(),
            scales=torch.tensor([[.13, .08, .06]]).expand(N, 3).contiguous().cuda())

    # A and B interleaved and overlapping in projection, so they are
    # co-visible and the renderer composites them in the same pixels.
    mu0 = torch.tensor([[-0.12, 0.10, 0.00], [0.10, -0.10, 0.02],
                        [-0.02, 0.04, 0.12], [0.05, -0.02, -0.10]])
    v_true = torch.tensor([[0.05, 0.0, 0.0]]).expand(N, 3).contiguous()
    truth = [make(mu0 + v_true * dt * t, v_true) for t in range(T)]
    obs = [[(r.clone(), d.clone(), torch.ones_like(d, dtype=torch.bool))
            for r, d in build_linearization(truth[t], cameras).predicted()]
           for t in range(T)]
    Q = torch.diag(torch.tensor([1e4] * 6 + [1e2] * 6, dtype=torch.float64)
                   ).expand(N, 12, 12).contiguous()
    groups = ((0, A), (T - 1, B))

    def solver(**kw):
        return UnifiedFreeIEKS(
            FreeRGBDObservation(cameras, obs, frame_ids=list(range(T))), Q,
            config=IEKSConfig(source_pose_policy="fixed_conditional",
                              anchor_groups=groups, verbose=False, **kw))

    g = torch.Generator().manual_seed(3)
    warm = Trajectory(states=[make(truth[t].mu.cpu() + 0.01 * torch.randn(
        N, 3, generator=g), torch.zeros(N, 3)) for t in range(T)],
        frame_ids=list(range(T)), dt=dt)

    print("[1] the elimination is per group, in the operator and the RHS")
    op, _ = solver().build_system(warm, 1e-4)
    u = torch.randn(T, N, 12, generator=g, dtype=torch.float64).cuda()
    Au, b = op.information_matvec(u), op.rhs()
    check("A's pose at frame 0 is eliminated",
          float(Au[0, A, 0:6].abs().max()) == 0.0 and float(b[0, A, 0:6].abs().max()) == 0.0)
    check("B's pose at frame T-1 is eliminated",
          float(Au[T - 1, B, 0:6].abs().max()) == 0.0
          and float(b[T - 1, B, 0:6].abs().max()) == 0.0)
    check("B's pose at frame 0 is NOT -- it was never reconstructed there",
          float(Au[0, B, 0:6].abs().max()) > 0.0,
          f"max |A u| = {float(Au[0, B, 0:6].abs().max()):.3e}")
    check("nor A's at frame T-1",
          float(Au[T - 1, A, 0:6].abs().max()) > 0.0,
          f"max |A u| = {float(Au[T - 1, A, 0:6].abs().max()):.3e}")
    check("the anchored rows' RATES stay free",
          float(Au[0, A, 6:12].abs().max()) > 0.0
          and float(Au[T - 1, B, 6:12].abs().max()) > 0.0)

    print("\n[2] one shared render couples the two sources")
    from bidirectional_iekf.ieks import Trajectory as _T  # noqa: F401
    plain = UnifiedFreeIEKS(FreeRGBDObservation(cameras, obs, frame_ids=list(range(T))),
                            torch.zeros_like(Q), config=IEKSConfig(verbose=False))
    op0, _ = plain.build_system(warm, 0.0)
    only_a = torch.zeros(T, N, 12, dtype=torch.float64).cuda()
    only_a[2, A, 0:3] = torch.randn(len(A), 3, generator=g, dtype=torch.float64).cuda()
    cross = op0.information_matvec(only_a)
    check("a tangent on source A produces information on source B's rows",
          float(cross[2, B, 0:6].abs().max()) > 0.0,
          f"max |(H'R^-1 H u_A)_B| = {float(cross[2, B, 0:6].abs().max()):.3e}, "
          "with the process term switched off so only the image couples them")

    print("\n[3] a real solve never moves an anchored pose")
    res = solver(max_outer=10, damping=1e-4, pcg_max_iters=120,
                 pcg_tol=1e-10).run(warm)
    st = res["trajectory"].states
    dA0 = float((st[0].mu[A] - warm.states[0].mu[A]).abs().max())
    dB = float((st[T - 1].mu[B] - warm.states[T - 1].mu[B]).abs().max())
    RA0 = float((st[0].R[A] - warm.states[0].R[A]).abs().max())
    RB = float((st[T - 1].R[B] - warm.states[T - 1].R[B]).abs().max())
    print(f"       {res['status']}, E {res['trace'][0]['E_total']:.3e} -> "
          f"{res['energy']['E_total']:.3e}")
    check("A's frame-0 pose is unchanged, centre and rotation",
          dA0 == 0.0 and RA0 == 0.0, f"{dA0:.3e} / {RA0:.3e}")
    check("B's frame-(T-1) pose is unchanged, centre and rotation",
          dB == 0.0 and RB == 0.0, f"{dB:.3e} / {RB:.3e}")
    moved_b0 = float((st[0].mu[B] - warm.states[0].mu[B]).abs().max())
    moved_aT = float((st[T - 1].mu[A] - warm.states[T - 1].mu[A]).abs().max())
    check("while the unanchored poses do move",
          moved_b0 > 0.0 and moved_aT > 0.0,
          f"B at frame 0 moved {moved_b0:.3e}, A at frame T-1 {moved_aT:.3e}")
    check("and the objective went down",
          res["energy"]["E_total"] < res["trace"][0]["E_total"])

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
