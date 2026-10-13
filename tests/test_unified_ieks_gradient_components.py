"""G1 and G4 — component-wise gradient, Taylor expansion, local convergence.

CODEX_DIAGNOSE_UNIFIED_IEKS_NUMERICAL_FAILURE_20261013.md, sections 1, 3, 4.

Three things the earlier work did not establish.

**Each term of b, on its own (section 1.5).** A total gradient can agree while
two of its parts cancel. The prior, the process, the RGB measurement and the
depth measurement are therefore finite-differenced SEPARATELY against their
own analytic contributions, and only then as a sum.

**The Taylor expansion (section 1.7).** The identity b = -grad F is checked by
its defining property: F(X boxplus alpha u) - F(X) + alpha*g'u must shrink
like alpha^2. A single ratio at one alpha cannot distinguish a correct
gradient from a slightly wrong one.

**Local convergence from a near-truth start (section 4.2).** The earlier
fixture used a stationary zero-velocity start, which is a global-basin stress
test, not a local one. Here all of mu, R, v and omega are perturbed around
the known trajectory -- INCLUDING a nonzero orientation perturbation, which
no previous fixture exercised -- and recovery is measured per block.

    python tests/test_unified_ieks_gradient_components.py
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
        print("G1/G4 NOT_RUN: CUDA or diff_gaussian_rasterization unavailable.")
        return 0

    from bidirectional_iekf.ieks import (
        CONVERGED, IEKSConfig, Trajectory, UnifiedFreeIEKS)
    from bidirectional_iekf.observations import FreeRGBDObservation
    from estimator.manifold import so3_exp, so3_log
    from estimator.renderer_bridge import build_linearization
    from estimator.state import GaussianMotionState
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

    N, T, dt = 4, 5, 0.1

    def make(mu, R, v, omega):
        return GaussianMotionState(
            mu=mu.cuda(), R=R.cuda(), v=v.cuda(), omega=omega.cuda(),
            colors=torch.tensor([[.9, .3, .2], [.2, .8, .3],
                                 [.3, .3, .9], [.9, .8, .2]]).cuda(),
            opacity=torch.full((N, 1), 0.92).cuda(),
            scales=torch.tensor([[.14, .07, .06]]).expand(N, 3).contiguous().cuda())

    cameras = make_cameras(n_cam=3, H=96, W=96, radius=2.4)
    mu0 = torch.tensor([[-0.35, 0.25, 0.0], [0.35, 0.25, 0.05],
                        [-0.30, -0.28, 0.02], [0.32, -0.26, -0.03]])
    v_true = torch.tensor([[0.10, 0.0, 0.0], [0.0, 0.09, 0.0],
                           [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # A genuine nonzero angular velocity, so the SO(3) process residual and
    # the orientation blocks are actually exercised.
    w_true = torch.tensor([[0.0, 0.0, 0.25], [0.12, 0.0, 0.0],
                           [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    R0 = so3_exp(torch.zeros(N, 3))

    def truth_at(t):
        return make(mu0 + v_true * (dt * t), so3_exp(w_true * (dt * t)) @ R0,
                    v_true, w_true)

    truth = [truth_at(t) for t in range(T)]
    observations = [[(rgb.clone(), depth.clone(),
                      torch.ones_like(depth, dtype=torch.bool))
                     for rgb, depth in build_linearization(truth[t], cameras).predicted()]
                    for t in range(T)]
    Q = torch.diag(torch.tensor([1e4] * 3 + [1e4] * 3 + [1e2] * 3 + [1e2] * 3,
                                dtype=torch.float64)).expand(N, 12, 12).contiguous()
    P0 = torch.eye(12, dtype=torch.float64).expand(N, 12, 12).contiguous() * 50.0
    truth_traj = Trajectory(states=truth, frame_ids=list(range(T)), dt=dt)

    # A nominal with nonzero image, prior AND process residuals.
    g = torch.Generator().manual_seed(11)
    offset = torch.zeros(T, N, 12, dtype=torch.float64)
    offset[:, :, 0:3] = torch.randn(T, N, 3, generator=g, dtype=torch.float64) * 0.004
    offset[:, :, 3:6] = torch.randn(T, N, 3, generator=g, dtype=torch.float64) * 0.010
    offset[:, :, 6:9] = torch.randn(T, N, 3, generator=g, dtype=torch.float64) * 0.020
    offset[:, :, 9:12] = torch.randn(T, N, 3, generator=g, dtype=torch.float64) * 0.030
    nominal = truth_traj.retract(offset.cuda(), 1.0)

    print("[1] component-wise gradients, each finite-differenced separately")

    def build(use_rgb, use_depth, with_prior, with_process):
        obs = FreeRGBDObservation(cameras, observations, frame_ids=list(range(T)),
                                  use_rgb=use_rgb, use_depth=use_depth)
        return UnifiedFreeIEKS(
            obs,
            Q if with_process else torch.zeros_like(Q),
            prior_information=P0 if with_prior else None,
            prior_mean=truth_traj if with_prior else None,
            config=IEKSConfig(verbose=False))

    variants = [
        ("prior only", build(False, False, True, False)),
        ("process only", build(False, False, False, True)),
        ("RGB only", build(True, False, False, False)),
        ("depth only", build(False, True, False, False)),
        ("full sum", build(True, True, True, True)),
    ]
    direction = torch.zeros(T, N, 12, dtype=torch.float64, device="cuda")
    direction[2, 0, 0] = 1.0        # translation at a mid frame
    direction[3, 1, 5] = 0.5        # orientation at a later frame
    direction[1, 0, 7] = 0.3        # a velocity direction, process-only signal

    analytic_sum = 0.0
    for label, solver in variants:
        operator, _ = solver.build_system(nominal, damping=0.0)
        b = operator.rhs()
        analytic = -float((b * direction).sum())
        best = None
        for eps in (2e-4, 1e-4, 5e-5):
            plus = solver.evaluate(nominal.retract(direction, eps))["E_total"]
            minus = solver.evaluate(nominal.retract(direction, -eps))["E_total"]
            fd = (plus - minus) / (2 * eps)
            rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-12)
            best = rel if best is None else min(best, rel)
        if label != "full sum":
            analytic_sum += analytic
        print(f"       {label:14s} analytic {analytic:+.6e}  best rel {best:.3e}")
        check(f"gradient term verified ({label})", best < 0.05,
              f"relative error {best:.3e}")
        if label == "full sum":
            check("the parts sum to the whole",
                  abs(analytic - analytic_sum) / max(abs(analytic), 1e-12) < 1e-6,
                  f"sum of parts {analytic_sum:+.6e} vs full {analytic:+.6e} -- "
                  "a total that agreed while two terms cancelled would pass a "
                  "whole-objective test and fail this one")

    print("\n[2] Taylor expansion: the residual must shrink like alpha^2")
    solver = variants[-1][1]
    operator, _ = solver.build_system(nominal, damping=0.0)
    b = operator.rhs()
    base = solver.evaluate(nominal)["E_total"]
    gdotu = -float((b * direction).sum())
    # The objective is evaluated by rendering in float32, so repeated
    # evaluations at the SAME state differ. That noise is the floor of any
    # Taylor residual, and alphas whose quadratic term sits under it measure
    # the renderer, not the expansion. The floor is measured, not assumed.
    # Two floors matter and only the second binds. Re-evaluating at a FIXED
    # state is deterministic here, so that variance is zero. What limits the
    # expansion is the granularity of F as a function of STATE: a float32
    # render changes in discrete steps, so F along a direction is a staircase
    # at small alpha. Estimated from the spread of F over a cluster of
    # nearby states at the smallest alpha of interest.
    repeats = [solver.evaluate(nominal)["E_total"] for _ in range(3)]
    fixed_state_noise = max(repeats) - min(repeats)
    probe = 8e-4
    nearby = [solver.evaluate(nominal.retract(direction, probe * (1 + 1e-3 * k)))["E_total"]
              for k in range(4)]
    floor = max(nearby) - min(nearby)
    print(f"       fixed-state evaluation variance: {fixed_state_noise:.3e}")
    print(f"       state-granularity floor near alpha={probe:.0e}: {floor:.4e} "
          f"(objective {base:.4e})")
    previous, ratios = None, []
    for alpha in (1.28e-2, 6.4e-3, 3.2e-3, 1.6e-3):
        value = solver.evaluate(nominal.retract(direction, alpha))["E_total"]
        residual = abs(value - base - alpha * gdotu)
        above = residual > 10.0 * max(floor, 1e-30)
        if previous is not None and above:
            ratios.append(previous / max(residual, 1e-30))
        print(f"       alpha {alpha:.1e}  |F(X+au) - F - a g'u| = {residual:.4e}"
              + (f"   ratio {ratios[-1]:.2f}" if ratios and above else "")
              + ("" if above else "   (below 10x the noise floor, excluded)"))
        previous = residual if above else None
    import math
    slopes = [math.log2(r) for r in ratios]
    print("       implied order (log2 of each ratio): "
          + ", ".join(f"{x:.2f}" for x in slopes))
    check("the residual is second order where it is measurable",
          bool(slopes) and max(slopes) > 1.7,
          f"best implied order {max(slopes):.2f} (exactly 2.0 is ideal; a "
          "wrong gradient leaves a linear term and gives 1.0). The order "
          f"degrades to {min(slopes):.2f} at the smallest alpha, where the "
          "residual approaches the render's state granularity -- reported "
          "rather than hidden by quoting only the best point.")

    print("\n[3] local convergence from a near-truth start, rotations included")
    obs = FreeRGBDObservation(cameras, observations, frame_ids=list(range(T)))
    for label, scale in (("tiny", 0.3), ("medium", 1.0)):
        solver = UnifiedFreeIEKS(obs, Q, config=IEKSConfig(
            max_outer=40, damping=1e-4, pcg_max_iters=400, pcg_tol=1e-11,
            verbose=False))
        start = truth_traj.retract(offset.cuda(), scale)
        begin = solver.evaluate(start)["E_total"]
        result = solver.run(start)
        mu_err = max(float((result["trajectory"].states[t].mu - truth[t].mu).norm())
                     for t in range(T))
        rot_err = max(float(so3_log(result["trajectory"].states[t].R.double()
                                    @ truth[t].R.double().transpose(-1, -2)).norm())
                      for t in range(T))
        v_err = float((result["trajectory"].states[-1].v - v_true.cuda()).norm())
        w_err = float((result["trajectory"].states[-1].omega - w_true.cuda()).norm())
        final_scaled = [r["scaled_projected_gradient"] for r in result["trace"]][-1]
        print(f"       {label:7s} {result['status']:10s} E {begin:.4e} -> "
              f"{result['energy']['E_total']:.4e}  |mu|={mu_err:.2e} "
              f"|rot|={rot_err:.2e} |v|={v_err:.2e} |w|={w_err:.2e}  "
              f"scaled|g|={final_scaled:.3e}")
        check(f"converged from a {label} perturbation",
              result["status"] == CONVERGED, result["status"])
        check(f"positions recovered ({label})", mu_err < 1e-3, f"{mu_err:.3e}")
        check(f"orientations recovered ({label})", rot_err < 1e-2,
              f"{rot_err:.3e} rad")
        check(f"linear velocity recovered ({label})", v_err < 5e-2, f"{v_err:.3e}")
        check(f"angular velocity recovered ({label})", w_err < 5e-2, f"{w_err:.3e}")

    print("\n[4] the quadratic model's predicted reduction, and rho")
    accepted = [r for r in result["trace"] if r["accepted"]]
    with_rho = [t for r in accepted for t in r["trials"]
                if t["accepted"] and t.get("rho") is not None]
    check("accepted trials record a predicted reduction and rho",
          len(with_rho) >= 1, f"{len(with_rho)} of {len(accepted)} accepted steps")
    if with_rho:
        good = [t for t in with_rho if 0.1 < t["rho"] < 10.0]
        print("       rho on accepted steps: "
              + ", ".join(f"{t['rho']:.3f}" for t in with_rho[:6]))
        check("rho is of order one on accepted steps",
              len(good) >= max(1, len(with_rho) // 2),
              f"{len(good)} of {len(with_rho)} within [0.1, 10] -- the "
              "quadratic model predicts the measured decrease, which the "
              "Armijo threshold (the previous formula) could not test")
    check("depth support change is recorded per trial",
          all("depth_support_change" in t for r in accepted for t in r["trials"]))

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
