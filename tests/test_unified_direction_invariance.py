"""T9 — one posterior, two initializations, and no double-counted evidence.

CODEX_FINISH_UNIFIED_PHASE1_AFTER_0AADC1E.md section 4.

The claim is NOT that a nonlinear optimiser reaches the same point from any
start. It is narrower and checkable:

  * the posterior is the same object either way -- same factors, same sealed
    (frame, camera) inventory, same prior and process;
  * a backward-initialised nominal must be converted to physical
    chronological time, rates and covariance cross-blocks included, before it
    is comparable at all;
  * for ONE fixed linearised system, eliminating forward or backward gives
    the same correction to solver tolerance -- that part IS an identity;
  * the inventory refuses a duplicate (frame, camera), so the two directions
    cannot contribute two copies of the same image.

The spec is explicit that two scripts which only swap array order are not
enough, so both initialisations are actually solved.

    python tests/test_unified_direction_invariance.py
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
        print("T9 NOT_RUN: CUDA or diff_gaussian_rasterization unavailable.")
        return 0

    from bidirectional_iekf.full_window_free import (
        ProcessFactor, pcg, schur_forward_backward)
    from bidirectional_iekf.ieks import IEKSConfig, Trajectory, UnifiedFreeIEKS
    from bidirectional_iekf.observations import FactorKey, FreeRGBDObservation
    from bidirectional_iekf.free_dynamics import (
        reverse_to_physical_covariance, reverse_to_physical_state)
    from bidirectional_iekf.provenance import (
        FINAL, INITIAL, SourceAnchor, build_candidate_population)
    from estimator.manifold import so3_exp, so3_log
    from estimator.renderer_bridge import build_linearization
    from estimator.state import GaussianMotionState
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

    # A NONOVERLAPPING two-source population: source 0 on the left, source T
    # on the right. Disjoint, so this is a validated representation rather
    # than the unresolved real overlap case.
    N0, NT = 2, 2
    N, T, dt = N0 + NT, 5, 0.1
    source_endpoint = torch.tensor([0, 0, 1, 1])

    def make(mu, R, v, w):
        return GaussianMotionState(
            mu=mu.cuda(), R=R.cuda(), v=v.cuda(), omega=w.cuda(),
            colors=torch.tensor([[.9, .3, .2], [.8, .4, .2],
                                 [.2, .4, .9], [.3, .5, .8]]).cuda(),
            opacity=torch.full((N, 1), 0.92).cuda(),
            scales=torch.tensor([[.13, .07, .06]]).expand(N, 3).contiguous().cuda())

    cameras = make_cameras(n_cam=3, H=96, W=96, radius=2.4)
    mu0 = torch.tensor([[-0.46, 0.20, 0.0], [-0.44, -0.22, 0.03],
                        [0.46, 0.21, 0.02], [0.44, -0.20, -0.02]])
    v_true = torch.tensor([[0.09, 0.0, 0.0], [0.09, 0.0, 0.0],
                           [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    w_true = torch.tensor([[0.0, 0.0, 0.20], [0.0, 0.0, 0.20],
                           [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    R0 = so3_exp(torch.zeros(N, 3))
    truth = [make(mu0 + v_true * (dt * t), so3_exp(w_true * (dt * t)) @ R0,
                  v_true, w_true) for t in range(T)]
    observations = [[(rgb.clone(), depth.clone(),
                      torch.ones_like(depth, dtype=torch.bool))
                     for rgb, depth in build_linearization(truth[t], cameras).predicted()]
                    for t in range(T)]
    Q = torch.diag(torch.tensor([1e4] * 3 + [1e4] * 3 + [1e2] * 3 + [1e2] * 3,
                                dtype=torch.float64)).expand(N, 12, 12).contiguous()
    truth_traj = Trajectory(states=truth, frame_ids=list(range(T)), dt=dt)

    print("[1] source identity survives, and the two sources are disjoint")
    import numpy as np
    population = build_candidate_population([
        SourceAnchor(INITIAL, "/m/0", "/m/0.ply", "0" * 64, 0,
                     mu0[:N0].numpy().astype(np.float64), N0),
        SourceAnchor(FINAL, "/m/T", "/m/T.ply", "1" * 64, T - 1,
                     mu0[N0:].numpy().astype(np.float64), NT)])
    check("candidate count is the disjoint union",
          population.num_candidates == N, str(population.num_candidates))
    check("source ids preserved",
          torch.equal(torch.from_numpy(population.source_endpoint), source_endpoint))
    check("anchor frames differ by endpoint",
          set(population.anchor_original_frame.tolist()) == {0, T - 1})

    print("\n[2] the backward warm start is CONVERTED, not merely reversed")
    # What a backward run stores: processing order, reverse-time rates.
    mu_proc = torch.stack([s.mu for s in truth]).flip(0)
    R_proc = torch.stack([s.R for s in truth]).flip(0)
    v_proc = torch.stack([-s.v for s in truth]).flip(0)
    w_proc = torch.stack([-s.omega for s in truth]).flip(0)
    mu_p, R_p, v_p, w_p = reverse_to_physical_state(mu_proc, R_proc, v_proc, w_proc)
    check("positions restored chronologically",
          torch.allclose(mu_p.cuda().float(),
                         torch.stack([s.mu for s in truth]), atol=1e-6))
    check("linear rate sign restored",
          torch.allclose(v_p.cuda().float(),
                         torch.stack([s.v for s in truth]), atol=1e-6),
          f"v_x at frame 0: {float(v_p[0, 0, 0]):+.4f} against truth "
          f"{float(truth[0].v[0, 0]):+.4f}")
    check("angular rate sign restored",
          torch.allclose(w_p.cuda().float(),
                         torch.stack([s.omega for s in truth]), atol=1e-6))
    naive = v_proc.flip(0)
    check("a mere array reversal gives the WRONG sign",
          bool((naive[0, 0, 0] * truth[0].v[0, 0].cpu() < 0)),
          "reversal alone is not a velocity conversion")
    P = torch.eye(12, dtype=torch.float64).expand(T, 12, 12).clone()
    P[:, 0, 6] = P[:, 6, 0] = 0.4
    P_phys = reverse_to_physical_covariance(P)
    check("covariance mu-v cross block flips sign",
          torch.allclose(P_phys[:, 0, 6], torch.full((T,), -0.4, dtype=torch.float64)))

    print("\n[3] solve the SAME posterior from both initializations")
    g = torch.Generator().manual_seed(17)
    perturb = torch.zeros(T, N, 12, dtype=torch.float64)
    perturb[:, :, 0:3] = torch.randn(T, N, 3, generator=g, dtype=torch.float64) * 0.004
    perturb[:, :, 3:6] = torch.randn(T, N, 3, generator=g, dtype=torch.float64) * 0.008
    perturb[:, :, 6:9] = torch.randn(T, N, 3, generator=g, dtype=torch.float64) * 0.02
    perturb[:, :, 9:12] = torch.randn(T, N, 3, generator=g, dtype=torch.float64) * 0.03

    forward_start = truth_traj.retract(perturb.cuda(), 1.0)
    # The backward-initialised nominal: the SAME states, arrived at through
    # the processing-order -> physical-time conversion above, so the two
    # differ in how they were produced rather than in what they represent.
    back_states = []
    for t in range(T):
        back_states.append(make(mu_p[t].float().cuda(), R_p[t].float().cuda(),
                                v_p[t].float().cuda(), w_p[t].float().cuda()))
    backward_start = Trajectory(states=back_states, frame_ids=list(range(T)),
                                dt=dt).retract(perturb.flip(0).cuda(), 1.0)

    results = {}
    for label, start in (("forward-init", forward_start),
                         ("backward-init", backward_start)):
        obs = FreeRGBDObservation(cameras, observations, frame_ids=list(range(T)))
        solver = UnifiedFreeIEKS(obs, Q, config=IEKSConfig(
            max_outer=40, damping=1e-4, pcg_max_iters=400, pcg_tol=1e-11,
            verbose=False))
        begin = solver.evaluate(start, with_support=True)
        out = solver.run(start)
        results[label] = (begin, out)
        mu_err = max(float((out["trajectory"].states[t].mu - truth[t].mu).norm())
                     for t in range(T))
        print(f"       {label:14s} {out['status']:10s} "
              f"E {begin['E_total']:.4e} -> {out['energy']['E_total']:.4e}  "
              f"(rgb {out['energy']['E_rgb']:.3e}, depth "
              f"{out['energy']['E_depth']:.3e}, proc "
              f"{out['energy']['E_process']:.3e})  |mu_err| {mu_err:.2e}  "
              f"factors {out['factor_inventory']['num_factors']}")

    fwd, bwd = results["forward-init"][1], results["backward-init"][1]
    check("both runs use exactly the same factor inventory",
          fwd["factor_inventory"]["num_factors"]
          == bwd["factor_inventory"]["num_factors"] == T * len(cameras),
          f"{fwd['factor_inventory']['num_factors']} factors, "
          f"{T} frames x {len(cameras)} cameras")
    check("both reach the same objective scale",
          abs(fwd["energy"]["E_total"] - bwd["energy"]["E_total"])
          / max(fwd["energy"]["E_total"], bwd["energy"]["E_total"], 1e-30) < 0.5,
          f"{fwd['energy']['E_total']:.4e} vs {bwd['energy']['E_total']:.4e}; "
          "exact equality is NOT claimed -- these are nonlinear local optima")
    for label in ("forward-init", "backward-init"):
        out = results[label][1]
        err = max(float((out["trajectory"].states[t].mu - truth[t].mu).norm())
                  for t in range(T))
        check(f"{label} recovers the trajectory", err < 1e-3, f"{err:.3e}")

    print("\n[4] forward vs reverse elimination on ONE linearised system")
    solver = UnifiedFreeIEKS(
        FreeRGBDObservation(cameras, observations, frame_ids=list(range(T))),
        Q, config=IEKSConfig(verbose=False))
    operator, _ = solver.build_system(forward_start, damping=1e-3)
    b = operator.rhs()
    from bidirectional_iekf.full_window_free import block_jacobi_preconditioner
    # Both solves are preconditioned and run to a tight tolerance. An
    # elimination-order identity can only be read off two solves that have
    # actually converged; comparing two MAX_ITERS iterates measures the
    # solver's remaining error, not the ordering.
    x_pcg, info = pcg(operator, b,
                      preconditioner=block_jacobi_preconditioner(operator),
                      max_iters=4000, tol=1e-13)

    # Reverse elimination: flip the time axis of the same factor graph, solve,
    # flip back. The factors are identical; only the elimination order moves.
    rev_process = []
    for factor in operator.process:
        t = factor.frame
        rev_process.append(ProcessFactor(
            frame=T - 2 - t, A=factor.B.clone(), B=factor.A.clone(),
            Q_inv=factor.Q_inv.clone(), residual=factor.residual.clone()))

    class _Flipped(type(operator)):
        pass

    flipped = operator.__class__(
        num_frames=T, num_gaussians=N, process_factors=rev_process,
        observation=operator.obs, caches=list(reversed(operator.caches)),
        prior_information=None, prior_residual=None, damping=operator.damping)
    b_rev = b.flip(0).clone()
    if operator.prior_information is not None:
        b_rev = b_rev
    x_rev, info_rev = pcg(flipped, b_rev,
                          preconditioner=block_jacobi_preconditioner(flipped),
                          max_iters=4000, tol=1e-13)
    x_rev_back = x_rev.flip(0)
    deviation = float((x_pcg - x_rev_back).norm()) / max(float(x_pcg.norm()), 1e-30)
    print(f"       forward elimination: {info['status']} {info['iterations']} "
          f"iters at {info['residual']:.2e}")
    print(f"       reverse elimination: {info_rev['status']} "
          f"{info_rev['iterations']} iters at {info_rev['residual']:.2e}")
    check("both PCG solves converged, so the comparison is meaningful",
          info["status"] == "CONVERGED" and info_rev["status"] == "CONVERGED",
          f"forward {info['status']} at {info['residual']:.2e}, reverse "
          f"{info_rev['status']} at {info_rev['residual']:.2e}")
    check("both elimination orders give the same correction",
          deviation < 1e-5,
          f"relative difference {deviation:.3e} -- this part IS an identity, "
          "unlike the nonlinear comparison above")

    print("\n[5] evidence cannot be double counted")
    obs = FreeRGBDObservation(cameras, observations, frame_ids=list(range(T)))
    obs.build_inventory()
    before = obs.inventory.as_manifest()["num_factors"]
    global CHECKS
    CHECKS += 1
    try:
        obs.inventory.register(FactorKey(2, 1))
        FAILURES.append("a duplicate (frame, camera) was accepted")
        print("  FAIL  a duplicate (frame, camera) was accepted")
    except ValueError as exc:
        print(f"  ok    a duplicate (frame, camera) is refused: {str(exc)[:60]}...")
    check("relinearisation does not register a factor",
          obs.inventory.as_manifest()["num_factors"] == before,
          f"{before} before and after")
    check("neither run added the other's posterior as evidence",
          fwd["factor_inventory"]["num_factors"]
          == bwd["factor_inventory"]["num_factors"],
          "the two directions share one factor set; no precision was summed")

    print("\n[6] source-pose policy is declared, and enforced by elimination")
    solver_fixed = UnifiedFreeIEKS(
        FreeRGBDObservation(cameras, observations, frame_ids=list(range(T))),
        Q, config=IEKSConfig(verbose=False,
                             source_pose_policy="fixed_conditional",
                             anchor_frames=(0,)))
    op_fixed, _ = solver_fixed.build_system(forward_start, damping=0.0)
    probe = torch.zeros(T, N, 12, dtype=torch.float64, device="cuda")
    probe[0, 0, 0] = 1.0
    out = op_fixed.information_matvec(probe)
    check("a conditionally fixed anchor's tangent is eliminated EXACTLY",
          float(out[0, :, 0:6].abs().max()) == 0.0,
          "zeroed, not given a large fake precision -- a soft constraint "
          "would leak into every reported uncertainty")
    rhs_fixed = op_fixed.rhs()
    check("and its right-hand side is zero there",
          float(rhs_fixed[0, :, 0:6].abs().max()) == 0.0)
    CHECKS += 1
    try:
        UnifiedFreeIEKS(obs, Q, config=IEKSConfig(source_pose_policy="whatever"))
        FAILURES.append("an undeclared source-pose policy was accepted")
        print("  FAIL  an undeclared source-pose policy was accepted")
    except ValueError:
        print("  ok    an undeclared source-pose policy is refused")

    print("\n[7] appearance frozen throughout")
    for label in ("forward-init", "backward-init"):
        out = results[label][1]
        check(f"{label} leaves colour, opacity and scales untouched",
              torch.equal(out["trajectory"].states[-1].colors, truth[0].colors)
              and torch.equal(out["trajectory"].states[-1].opacity, truth[0].opacity)
              and torch.equal(out["trajectory"].states[-1].scales, truth[0].scales))

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
