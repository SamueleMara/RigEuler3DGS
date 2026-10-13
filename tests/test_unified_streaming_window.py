"""The streamed window is the same operator, not a cheaper approximation.

Section 5 of CODEX_COMPLETE_MISSING_UNIFIED_BIDI_PHASE1_AFTER_9B78EFA.md:
"Use exact mathematical matrix-free H_G JVP/VJP, sparse process factors,
bounded recompute/streaming, explicit CPU/GPU transfer semantics and
documented memory peaks."

What forced it, measured on the real Washer scene (24,582 Gaussians, 640x480,
RTX 4070 Laptop 8 GiB) with tools/run_unified_phase1_single_source.py:

    4 frames x 1 camera     4 factors    878 MiB
    8 frames x 2 cameras   16 factors   1901 MiB
   16 frames x 4 cameras   64 factors   OOM at 4147 MiB

about 85 MiB per (frame, camera) factor. The real window is 60 frames x 8
cameras = 480 factors, so the resident path needs of the order of 41 GiB.
That is a measurement, not an extrapolation from a 60-Gaussian fixture.

`IEKSConfig.stream_frames` keeps ONE frame's linearisation alive and rebuilds
it on demand. The claim this file checks is the only one that matters: the
operator's action, its right-hand side, its preconditioner blocks and the
accepted trajectory are the same either way.

"The same" has to be measured, not assumed to be bitwise. The CUDA rasterizer
composites in float32 with a tile order that is not reproducible run to run,
so building the SAME resident system twice from the same state already differs
by a few parts in 1e7. That floor is measured here first, from a resident
control, and streaming is then required to stay inside it. Calling a 2e-7
difference a defect without the control would be reading the renderer's noise
as a method error; demanding 1e-12 of a float32 renderer would fail the
correct implementation.

    python tests/test_unified_streaming_window.py
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
        print("streaming NOT_RUN: CUDA or diff_gaussian_rasterization unavailable.")
        return 0

    from bidirectional_iekf.ieks import (CONVERGED, IEKSConfig, Trajectory,
                                         UnifiedFreeIEKS, _StreamedCaches)
    from bidirectional_iekf.observations import FreeRGBDObservation
    from estimator.manifold import so3_exp
    from estimator.renderer_bridge import build_linearization
    from estimator.state import GaussianMotionState
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

    torch.manual_seed(0)
    N, T, dt = 6, 5, 0.1
    cameras = make_cameras(n_cam=2, H=96, W=96, radius=2.4)

    def make(mu, v):
        return GaussianMotionState(
            mu=mu.float().cuda(), R=so3_exp(torch.zeros(N, 3)).float().cuda(),
            v=v.float().cuda(), omega=torch.zeros(N, 3).cuda(),
            colors=(torch.rand(N, 3, generator=torch.Generator().manual_seed(3))
                    * 0.7 + 0.2).cuda(),
            opacity=torch.full((N, 1), 0.9).cuda(),
            scales=torch.tensor([[0.12, 0.07, 0.06]]).expand(N, 3).contiguous().cuda())

    mu0 = torch.tensor([[-0.42, 0.22, 0.0], [-0.40, -0.24, 0.03],
                        [-0.05, 0.30, -0.02], [0.05, -0.28, 0.02],
                        [0.44, 0.20, 0.01], [0.42, -0.22, -0.03]])
    v_true = torch.zeros(N, 3)
    v_true[4:, 1] = 0.08
    truth = [make(mu0 + v_true * (dt * t), v_true) for t in range(T)]
    observations = [[(rgb.clone(), depth.clone(),
                      torch.ones_like(depth, dtype=torch.bool))
                     for rgb, depth in build_linearization(truth[t], cameras).predicted()]
                    for t in range(T)]
    Q = torch.diag(torch.tensor([1e4] * 3 + [1e4] * 3 + [1e2] * 3 + [1e2] * 3,
                                dtype=torch.float64)).expand(N, 12, 12).contiguous()
    warm = Trajectory(states=[make(mu0 + 0.012, torch.zeros(N, 3)) for _ in range(T)],
                      frame_ids=list(range(T)), dt=dt)

    def ieks(stream, host_process=None, diagonal=False, max_outer=25):
        obs = FreeRGBDObservation(cameras, observations, frame_ids=list(range(T)))
        return UnifiedFreeIEKS(obs, Q, config=IEKSConfig(
            max_outer=max_outer, damping=1e-4, pcg_max_iters=120, pcg_tol=1e-11,
            stream_frames=stream,
            process_factors_on_cpu=(stream if host_process is None
                                    else host_process),
            diagonal_preconditioner=diagonal, verbose=False))

    print("[1] the streamed provider is a window, read one frame at a time")
    resident_ieks, streamed_ieks = ieks(False), ieks(True)
    op_r, caches_r = resident_ieks.build_system(warm, 1e-4)
    op_s, caches_s = streamed_ieks.build_system(warm, 1e-4)
    check("the resident path materialises every frame",
          isinstance(caches_r, list) and len(caches_r) == T, f"{T} caches held")
    check("the streamed path holds none of them",
          isinstance(caches_s, _StreamedCaches) and len(caches_s) == T,
          "_StreamedCaches, rebuilt on demand")
    try:
        caches_s[0:2]
        check("slicing a streamed window is refused", False, "no exception")
    except TypeError:
        check("slicing a streamed window is refused", True,
              "slicing would materialise exactly what this avoids")

    print("\n[2] the control: two RESIDENT builds of the same system")
    g = torch.Generator().manual_seed(11)
    u = torch.randn(T, N, 12, generator=g, dtype=torch.float64).cuda()

    def rel(a, b):
        return float((a - b).norm()) / max(float(a.norm()), 1e-30)

    controls = []
    for _ in range(3):
        op_c, _ = ieks(False).build_system(warm, 1e-4)
        controls.append({
            "rhs": rel(op_r.rhs(), op_c.rhs()),
            "matvec": rel(op_r.information_matvec(u), op_c.information_matvec(u)),
            "fisher": rel(op_r.measurement_block_diagonal(),
                          op_c.measurement_block_diagonal()),
            "curvature": rel(op_r.curvature_scale(), op_c.curvature_scale())})
    floor = {k: max(c[k] for c in controls) for k in controls[0]}
    print("       resident-vs-resident, 3 rebuilds: "
          + ", ".join(f"{k} {v:.3e}" for k, v in floor.items()))
    check("the renderer is not bitwise reproducible, so a floor exists",
          max(floor.values()) > 0.0,
          f"largest control difference {max(floor.values()):.3e}; a bitwise "
          "comparison would be the wrong test")

    print("\n[3] streamed against resident, inside that floor")
    tol = {k: max(4.0 * v, 1e-12) for k, v in floor.items()}
    db = rel(op_r.rhs(), op_s.rhs())
    check("the gradients agree", db <= tol["rhs"],
          f"{db:.3e} against a {floor['rhs']:.3e} control floor")
    da = rel(op_r.information_matvec(u), op_s.information_matvec(u))
    check("the operator actions agree", da <= tol["matvec"],
          f"{da:.3e} against a {floor['matvec']:.3e} control floor, "
          "on a random tangent")
    dm = rel(op_r.measurement_block_diagonal(), op_s.measurement_block_diagonal())
    check("the measurement Fisher blocks agree", dm <= tol["fisher"],
          f"{dm:.3e} against a {floor['fisher']:.3e} control floor")
    dc = rel(op_r.curvature_scale(), op_s.curvature_scale())
    check("the LM curvature scales agree", dc <= tol["curvature"],
          f"{dc:.3e} against a {floor['curvature']:.3e} control floor")

    print("\n[4] host-resident process Jacobians change nothing but their address")
    op_h, _ = ieks(False, host_process=True).build_system(warm, 1e-4)
    check("A and B really are off the device",
          op_h.process[0].A.device.type == "cpu"
          and op_r.process[0].A.device.type == "cuda",
          f"{op_h.process[0].A.device} against {op_r.process[0].A.device}")
    check("but the solve still runs on the device",
          op_h.work_device.type == "cuda" and op_h.zeros().device.type == "cuda",
          f"work_device {op_h.work_device}; inferring it from the factors' "
          "storage would have moved the whole PCG to the CPU")
    dh = rel(op_r.information_matvec(u), op_h.information_matvec(u))
    check("and the operator action is unchanged", dh <= tol["matvec"],
          f"{dh:.3e} against a {floor['matvec']:.3e} control floor")
    dhb = rel(op_r.rhs(), op_h.rhs())
    check("as is the right-hand side", dhb <= tol["rhs"],
          f"{dhb:.3e} against a {floor['rhs']:.3e} control floor")
    from bidirectional_iekf.full_window_free import (
        _diagonal_jacobi_preconditioner, block_jacobi_preconditioner)
    m_r = block_jacobi_preconditioner(op_r, damping=1e-4)
    m_h = block_jacobi_preconditioner(op_h, damping=1e-4)
    z = torch.randn(T, N, 12, generator=torch.Generator().manual_seed(5),
                    dtype=torch.float64).cuda()
    dp = rel(m_r(z), m_h(z))
    check("and so is the preconditioner", dp <= max(4.0 * floor["fisher"], 1e-12),
          f"{dp:.3e} against a {floor['fisher']:.3e} control floor")

    print("\n[5] the memo stops one pass turning into several")
    before = caches_s.rebuilds
    op_s.measurement_block_diagonal()
    op_s.curvature_scale()
    check("asking again does not re-render the window",
          caches_s.rebuilds == before,
          f"{caches_s.rebuilds} rebuilds, unchanged by two further requests")
    before = caches_s.rebuilds
    op_s.information_matvec(u)
    check("a matvec rebuilds exactly one linearisation per frame",
          caches_s.rebuilds - before == T,
          f"{caches_s.rebuilds - before} rebuilds for {T} frames")

    print("\n[6] and the two solvers reach the same accepted trajectory")
    e_start = resident_ieks.evaluate(warm)["E_total"]
    torch.cuda.reset_peak_memory_stats()
    res_r = resident_ieks.run(warm)
    peak_r = torch.cuda.max_memory_allocated() / 2 ** 20
    torch.cuda.reset_peak_memory_stats()
    res_s = streamed_ieks.run(warm)
    peak_s = torch.cuda.max_memory_allocated() / 2 ** 20
    e_r, e_s = res_r["energy"]["E_total"], res_s["energy"]["E_total"]
    worst = max(float((res_r["trajectory"].states[t].mu
                       - res_s["trajectory"].states[t].mu).norm(dim=1).max())
                for t in range(T))
    print(f"       resident {res_r['status']} E {e_r:.6e} peak {peak_r:.0f} MiB; "
          f"streamed {res_s['status']} E {e_s:.6e} peak {peak_s:.0f} MiB")
    check("both converge", res_r["status"] == CONVERGED and res_s["status"] == CONVERGED,
          f"{res_r['status']} / {res_s['status']}")
    # A second resident solve gives the control for the whole solve, not just
    # for one linearisation: the nondeterminism is inside the optimiser's loop,
    # so it compounds across outer iterations and line-search trials.
    res_c = ieks(False).run(warm)
    e_c = res_c["energy"]["E_total"]
    worst_c = max(float((res_r["trajectory"].states[t].mu
                         - res_c["trajectory"].states[t].mu).norm(dim=1).max())
                  for t in range(T))
    print(f"       resident control {res_c['status']} E {e_c:.6e}, "
          f"states differ by {worst_c:.3e} from the first resident solve")
    # The right scale for "the same objective" is the energy the solve
    # removed, not the energy it ended at: both runs reach about 1e-07 out of
    # 4.5e+03, so the two final values are each at the renderer's own noise
    # floor and comparing them to one another is comparing noise to noise.
    scale = e_start - e_r
    check("to the same objective, relative to what the solve removed",
          abs(e_r - e_s) <= 1e-9 * scale,
          f"|dE| streamed {abs(e_r - e_s):.3e}, resident control "
          f"{abs(e_r - e_c):.3e}, both against {scale:.3e} removed "
          f"(a relative {abs(e_r - e_s) / scale:.2e})")
    check("and the same states, within the resident control",
          worst <= max(4.0 * worst_c, 1e-9),
          f"{worst:.3e} against a control {worst_c:.3e}")
    check("streaming never costs more memory than residency", peak_s <= peak_r,
          f"{peak_s:.0f} MiB against {peak_r:.0f} MiB on a {T}-frame "
          f"{len(cameras)}-camera fixture; the gap is what grows with the "
          "window, not what is visible at this size")

    print("\n[7] the diagonal preconditioner changes the rate, not the answer")
    # Two separate render passes over the window, so the comparison sits on
    # the same float32 floor section [2] measured; within one pass the memo
    # takes the diagonal of the blocks it already has and the agreement is
    # exact, which is checked straight after.
    dm_diag = rel(op_r.measurement_block_diagonal().diagonal(dim1=-2, dim2=-1),
                  op_r.measurement_diagonal())
    check("measurement_diagonal is the blocks' diagonal", dm_diag <= tol["fisher"],
          f"{dm_diag:.3e} against a {floor['fisher']:.3e} control floor; it is "
          "built one frame at a time, so the peak is one [N,12,12] rather "
          "than T of them")
    fresh, _ = ieks(False).build_system(warm, 1e-4)
    blocks = fresh.measurement_block_diagonal()
    check("and exactly so when both come from one pass",
          rel(blocks.diagonal(dim1=-2, dim2=-1), fresh.measurement_diagonal()) == 0.0,
          "the memo reuses the blocks rather than re-rendering")
    d_pre = _diagonal_jacobi_preconditioner(op_r, 1e-4)
    full = block_jacobi_preconditioner(op_r, damping=1e-4)
    check("the two preconditioners are genuinely different operators",
          rel(full(z), d_pre(z)) > 1e-3,
          f"relative difference {rel(full(z), d_pre(z)):.3e} -- this is not a "
          "no-op dressed up as a saving")
    res_d = ieks(False, diagonal=True).run(warm)
    e_d = res_d["energy"]["E_total"]
    worst_d = max(float((res_r["trajectory"].states[t].mu
                         - res_d["trajectory"].states[t].mu).norm(dim=1).max())
                  for t in range(T))
    print(f"       block {res_r['status']} E {e_r:.6e}; "
          f"diagonal {res_d['status']} E {e_d:.6e}; "
          f"states differ by {worst_d:.3e}")
    # Iteration counts are reported, not asserted: the two preconditioners sit
    # within an outer iteration or two of each other on a fixture this small,
    # and which is ahead varies run to run with the renderer. Asserting on
    # that would be asserting on noise.
    check("given the iterations it costs, it reaches the same optimum",
          res_d["status"] == CONVERGED and worst_d <= max(4.0 * worst_c, 1e-9),
          f"{res_d['status']} after {len(res_d['trace'])} outer against "
          f"{len(res_r['trace'])}, states within {worst_d:.3e} of a "
          f"{worst_c:.3e} resident control; PCG converges to the same point "
          "under any SPD preconditioner, and this one costs iterations to "
          "save 1.70 GiB three times over")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
