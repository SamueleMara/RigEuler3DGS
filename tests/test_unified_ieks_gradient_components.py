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
    # The probe MUST touch frame 0, or the prior term contributes exactly
    # zero and its "verification" is a 0/0 comparison that proves nothing.
    # The earlier version probed frames 1, 2 and 3 only.
    direction = torch.zeros(T, N, 12, dtype=torch.float64, device="cuda")
    direction[0, 0, 0] = 0.7        # frame-0 translation  -> prior
    direction[0, 1, 4] = 0.4        # frame-0 ORIENTATION  -> SO(3) prior
    direction[0, 2, 8] = 0.2        # frame-0 velocity     -> prior
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
        if label == "prior only":
            check("the prior gradient probe is NONZERO",
                  abs(analytic) > 1e-6,
                  f"analytic {analytic:+.6e}; a zero/zero comparison is a "
                  "trivial control, never the evidence for a pass")
        if label == "full sum":
            check("the parts sum to the whole",
                  abs(analytic - analytic_sum) / max(abs(analytic), 1e-12) < 1e-6,
                  f"sum of parts {analytic_sum:+.6e} vs full {analytic:+.6e} -- "
                  "a total that agreed while two terms cancelled would pass a "
                  "whole-objective test and fail this one")

    print("\n[2] Taylor expansion against the ANALYTIC quadratic model")
    print("        R(a) = F(X+au) - F(X) - a g'u  must equal 0.5 a^2 u'A0 u,")
    print("        with A0 the undamped operator. Comparing R against an")
    print("        independently computed quantity tests the gradient AND the")
    print("        curvature; a ratio test alone tests neither against a known")
    print("        value. Signed residuals and R/a, R/a^2 are reported so a")
    print("        residual linear term would be visible as a constant R/a.")
    operator, _ = solver.build_system(nominal, damping=0.0)
    b = operator.rhs()
    base = solver.evaluate(nominal)["E_total"]
    gdotu = -float((b * direction).sum())
    uAu = float((direction * operator.information_matvec(direction)).sum())
    check("the curvature along the probe is positive", uAu > 0, f"u'A0u = {uAu:.4e}")

    rows = []
    for alpha in (2.56e-2, 1.28e-2, 6.4e-3, 3.2e-3, 1.6e-3, 8e-4, 4e-4):
        value = solver.evaluate(nominal.retract(direction, alpha))["E_total"]
        residual = value - base - alpha * gdotu       # SIGNED
        quadratic = 0.5 * alpha * alpha * uAu
        rows.append((alpha, residual, quadratic))
        print(f"        a {alpha:8.2e}  R {residual:+11.4e}  "
              f"0.5a^2u'A0u {quadratic:11.4e}  R/a {residual / alpha:10.3e}  "
              f"R/a^2 {residual / alpha ** 2:10.3e}")

    # The floor is estimated from where R DEPARTS from the analytic model, and
    # from the non-monotonicity of that departure -- not from the spread of F
    # over nearby states, which contains a genuine derivative and was the
    # wrong estimator used previously.
    excess = [abs(r - q) for _, r, q in rows]
    tail = excess[-3:]
    floor = max(tail)
    monotone_excess = all(tail[i] <= tail[i - 1] for i in range(1, len(tail)))
    print(f"        |R - quadratic| across the sweep: "
          + ", ".join(f"{e:.2e}" for e in excess))
    print(f"        estimated evaluation granularity: {floor:.3e} "
          f"(from the small-alpha excess, which is "
          f"{'monotone' if monotone_excess else 'NON-monotone -> noise, not a linear term'})")
    check("the small-alpha excess is non-monotone, i.e. noise not a linear term",
          not monotone_excess,
          "a genuine linear error in g would give a residual growing steadily "
          "as R/a -> const; the measured excess instead jumps up and down")

    above = [(a, r, q) for a, r, q in rows if abs(r) > 10.0 * floor]
    check("there is an above-noise range to test", len(above) >= 3,
          f"{len(above)} of {len(rows)} alphas have |R| > 10x the granularity")
    worst = max(abs(r - q) / max(abs(q), 1e-30) for _, r, q in above)
    print(f"        above-noise alphas: "
          + ", ".join(f"{a:.2e}" for a, _, _ in above))
    check("R matches the analytic quadratic model where measurable",
          worst < 0.20,
          f"worst relative deviation {worst:.3f} over {len(above)} alphas; "
          "this validates g'u AND u'A0u together")
    below = [a for a, r, _ in rows if abs(r) <= 10.0 * floor]
    if below:
        print(f"        NUMERICALLY_INCONCLUSIVE below a = {max(below):.1e}: the "
              "residual there is at the render's evaluation granularity, so no "
              "order can be established, and none is claimed.")

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
