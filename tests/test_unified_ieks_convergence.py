"""T8d — a genuinely CONVERGED nonlinear RGB-D IEKS, and why the first try wasn't.

Section B of CODEX_REVIEW_AFTER_BF3565D_FIX_T8_THEN_T9_T10.md.

The earlier T8 fixture stopped at MAX_ITERS having removed ~1% of the
objective, and the review was right to refuse that as a pass. The cause was
found by ablation rather than guessed, and it is a property of photometric
alignment, not a solver defect:

    at 2.4 m with scale 0.14 and tanfov 0.483, a Gaussian's projected radius
    is about 5.8 px at 96x96. The fixture moved it 0.2 world units, about
    8.3 px. Predicted and observed blobs barely overlap, so the image
    gradient at the nominal does not point at the optimum. That is the
    capture radius of photometric alignment.

Ablations that RULED OUT the alternatives, each run to completion:

    process stiffness Q swept over 1000x   -> identical result, so the
                                              process term was not the cause
    rate blocks (v, omega) frozen          -> identical result
    rotation + rates frozen                -> identical result
    preconditioner given the measurement
      Fisher it was missing                -> helped the solve, not the stall
    LM damping scaled by diag(A)           -> steps shrank correctly, stall
                                              remained

With the perturbation inside the capture radius the same code converges to
the MAP exactly, and recovers the velocity -- which has NO image Jacobian at
all -- purely through the temporal process coupling.

    python tests/test_unified_ieks_convergence.py
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


def structured_Q(n, q_mu, q_phi, q_v, q_w):
    diag = torch.tensor([q_mu] * 3 + [q_phi] * 3 + [q_v] * 3 + [q_w] * 3,
                        dtype=torch.float64)
    return torch.diag(diag).expand(n, 12, 12).contiguous()


def main() -> int:
    if not cuda_ready():
        print("T8d NOT_RUN: CUDA or diff_gaussian_rasterization unavailable.")
        return 0

    from bidirectional_iekf.ieks import (
        CONVERGED, IEKSConfig, STALLED, Trajectory, UnifiedFreeIEKS)
    from bidirectional_iekf.observations import FreeRGBDObservation
    from estimator.manifold import so3_exp
    from estimator.renderer_bridge import build_linearization
    from estimator.state import GaussianMotionState
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

    def make(mu, v):
        n = mu.shape[0]
        return GaussianMotionState(
            mu=mu.cuda(), R=so3_exp(torch.zeros(n, 3)).cuda(), v=v.cuda(),
            omega=torch.zeros(n, 3).cuda(),
            colors=torch.tensor([[.9, .3, .2], [.2, .8, .3],
                                 [.3, .3, .9], [.9, .8, .2]])[:n].cuda(),
            opacity=torch.full((n, 1), 0.92).cuda(),
            scales=torch.tensor([[.14, .07, .06]]).expand(n, 3).contiguous().cuda())

    cameras = make_cameras(n_cam=3, H=96, W=96, radius=2.4)
    T, dt, N = 5, 0.1, 4
    mu0 = torch.tensor([[-0.35, 0.25, 0.0], [0.35, 0.25, 0.05],
                        [-0.30, -0.28, 0.02], [0.32, -0.26, -0.03]])
    # Inside the capture radius: ~0.04 total travel against a ~5.8 px
    # projected radius. The large-motion case is kept below as the
    # documented negative control.
    v_true = torch.tensor([[0.10, 0.0, 0.0], [0.0, 0.09, 0.0],
                           [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    truth = [make(mu0 + v_true * (dt * t), v_true) for t in range(T)]
    observations = [[(rgb.clone(), depth.clone(),
                      torch.ones_like(depth, dtype=torch.bool))
                     for rgb, depth in build_linearization(truth[t], cameras).predicted()]
                    for t in range(T)]

    print("[1] the optimum is exact and reachable")
    obs = FreeRGBDObservation(cameras, observations, frame_ids=list(range(T)))
    Q = structured_Q(N, 1e4, 1e4, 1e2, 1e2)
    ieks = UnifiedFreeIEKS(obs, Q, config=IEKSConfig(
        max_outer=40, damping=1e-4, pcg_max_iters=400, pcg_tol=1e-11,
        verbose=False))
    at_truth = ieks.evaluate(Trajectory(states=truth, frame_ids=list(range(T)), dt=dt))
    check("E(truth) is zero to machine precision", at_truth["E_total"] < 1e-9,
          f"E_total = {at_truth['E_total']:.3e}")

    print("\n[2] the IEKS converges to it from a wrong, zero-velocity start")
    warm = Trajectory(states=[make(mu0, torch.zeros(N, 3)) for _ in range(T)],
                      frame_ids=list(range(T)), dt=dt)
    start = ieks.evaluate(warm)["E_total"]
    result = ieks.run(warm)
    accepted = [r for r in result["trace"] if r["accepted"]]
    final = result["energy"]["E_total"]
    pose_err = max(float((result["trajectory"].states[t].mu - truth[t].mu).norm())
                   for t in range(T))
    vel_err = float((result["trajectory"].states[-1].v - v_true.cuda()).norm())

    print(f"       E {start:.4e} -> {final:.4e} "
          f"({100 * (1 - final / start):.3f}% reduced), "
          f"{len(accepted)} accepted of {len(result['trace'])}")
    print(f"       final |g| = {result['trace'][-1]['grad_norm']:.3e}, "
          f"|d| = {result['trace'][-1]['step_norm']:.3e}")
    check("status is CONVERGED, not MAX_ITERS or STALLED",
          result["status"] == CONVERGED,
          f"{result['status']}; convergence is decided on STATE stationarity "
          "because |g| floors at the float32 rendering noise (~4e-2 here) "
          "long after the state has stopped moving")
    check("the objective is driven to the optimum",
          final < 1e-6 * start, f"{final:.3e} from {start:.3e}")
    check("every accepted step decreased the true rerendered posterior",
          all(accepted[i]["E_total"] <= accepted[i - 1]["E_total"] + 1e-9
              for i in range(1, len(accepted))))
    check("positions recovered", pose_err < 1e-3, f"max |mu_err| = {pose_err:.3e}")
    check("velocities recovered through the process coupling alone",
          vel_err < 1e-2,
          f"|v_err| = {vel_err:.3e}; the image Jacobian has no velocity "
          "columns, so this came entirely from the temporal factors")

    print("\n[3] the documented negative control: outside the capture radius")
    v_big = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.45, 0.0],
                          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    truth_big = [make(mu0 + v_big * (dt * t), v_big) for t in range(T)]
    obs_big = FreeRGBDObservation(
        cameras,
        [[(rgb.clone(), depth.clone(), torch.ones_like(depth, dtype=torch.bool))
          for rgb, depth in build_linearization(truth_big[t], cameras).predicted()]
         for t in range(T)],
        frame_ids=list(range(T)))
    ieks_big = UnifiedFreeIEKS(obs_big, Q, config=IEKSConfig(
        max_outer=15, damping=1e-4, pcg_max_iters=400, pcg_tol=1e-11,
        verbose=False))
    warm_big = Trajectory(states=[make(mu0, torch.zeros(N, 3)) for _ in range(T)],
                          frame_ids=list(range(T)), dt=dt)
    start_big = ieks_big.evaluate(warm_big)["E_total"]
    big = ieks_big.run(warm_big)
    reduction = 1 - big["energy"]["E_total"] / start_big
    print(f"       E {start_big:.4e} -> {big['energy']['E_total']:.4e} "
          f"({100 * reduction:.2f}% reduced), status {big['status']}")
    check("a displacement beyond the footprint is NOT solved",
          reduction < 0.5,
          f"only {100 * reduction:.2f}% of the objective removed -- the "
          "projected radius is ~5.8 px and the displacement ~8.3 px, so the "
          "image gradient at the nominal does not point at the optimum. This "
          "is the capture radius of photometric alignment, recorded as a "
          "limitation rather than hidden by a smaller fixture.")
    check("and it is not misreported as converged",
          big["status"] != CONVERGED,
          f"status {big['status']} -- the quadratic model still predicts "
          "reduction, so the stationarity certificate is not met and the run "
          "must not be labelled converged whatever its step norm did")

    print("\n[4] a flat objective with an uncertified decrement is not CONVERGED")
    stalled = [r for r in big["trace"] if not r["accepted"]]
    check("the stalling run recorded rejected steps", len(stalled) >= 1,
          f"{len(stalled)} rejections")
    check("STALLED is a distinct status from CONVERGED", STALLED != CONVERGED)

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
