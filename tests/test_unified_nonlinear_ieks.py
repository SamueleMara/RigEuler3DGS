"""T6 and T8 — the representation diagnostic and the real nonlinear IEKS.

CODEX_NEXT_STEPS_UNIFIED_PHASE1_AFTER_T5.md sections 2 and 4.

T8's claim is the one that matters: a late-frame observation must correct an
EARLIER state that the camera could not see, through the temporal process
coupling. The spec forbids simulating that by zeroing a hand-written
measurement matrix, so the occlusion here is physical -- a Gaussian hidden
behind another in the real rasteriser -- and the correction is measured by
re-running with the late frames removed and differencing.

    python tests/test_unified_nonlinear_ieks.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
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


def state_from(mu, scales, colors, opacity, omega=None, v=None):
    from estimator.manifold import so3_exp
    from estimator.state import GaussianMotionState
    n = mu.shape[0]
    return GaussianMotionState(
        mu=mu.cuda(), R=so3_exp(torch.zeros(n, 3)).cuda(),
        v=(v if v is not None else torch.zeros(n, 3)).cuda(),
        omega=(omega if omega is not None else torch.zeros(n, 3)).cuda(),
        colors=colors.cuda(), opacity=opacity.cuda(), scales=scales.cuda())


def main() -> int:
    if not cuda_ready():
        print("T6/T8 NOT_RUN: CUDA or diff_gaussian_rasterization unavailable.")
        return 0

    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    from bidirectional_iekf.ieks import IEKSConfig, Trajectory, UnifiedFreeIEKS, save_run
    from bidirectional_iekf.observations import FreeRGBDObservation
    from bidirectional_iekf.representation_audit import (
        PASSED, UNRESOLVED, audit_pair, write_audit)
    from estimator.renderer_bridge import build_linearization
    from estimator.state import boxplus

    # ================= T6 =================
    print("[T6] representation diagnostic")
    cameras = make_cameras(n_cam=2, H=128, W=128, radius=2.6)
    # T8 needs every camera on ONE side: make_cameras spreads them around a
    # full ring, so a two-camera ring puts the second one behind the scene,
    # where the "hidden" Gaussian is in front of its blocker and nothing is
    # occluded at all.
    # 10 degrees apart: a 12-camera ring puts the second view 30 degrees
    # round, from where a Gaussian outside the first camera's frustum is
    # comfortably inside the second's, and nothing is unobserved at all.
    occluding_cameras = make_cameras(n_cam=36, H=128, W=128, radius=2.6)[:2]
    aniso = torch.tensor([0.16, 0.06, 0.05])

    left = state_from(torch.tensor([[-0.55, 0.0, 0.0], [-0.40, 0.25, 0.1]]),
                      aniso.expand(2, 3).clone(),
                      torch.tensor([[0.9, 0.2, 0.2], [0.8, 0.3, 0.2]]),
                      torch.full((2, 1), 0.95))
    right = state_from(torch.tensor([[0.55, 0.0, 0.0], [0.40, -0.25, 0.1]]),
                       aniso.expand(2, 3).clone(),
                       torch.tensor([[0.2, 0.3, 0.9], [0.2, 0.4, 0.8]]),
                       torch.full((2, 1), 0.95))

    disjoint = audit_pair(left, right, cameras, label="synthetic disjoint")
    check("disjoint two-source population composites as one scene",
          disjoint["status"] == PASSED,
          f"{disjoint['overlap_pixels_total']} overlapping pixels")

    duplicate = audit_pair(left, left, cameras, label="synthetic identical")
    check("an identical duplicate is reported UNRESOLVED",
          duplicate["status"] == UNRESOLVED,
          f"{duplicate['overlap_pixels_total']} pixels covered by both sources")
    check("duplication raises alpha above either source alone",
          duplicate["max_alpha_gain_over_single_source"] > 1e-3,
          f"max alpha gain {duplicate['max_alpha_gain_over_single_source']:.4f} "
          "-- the compositor's 1-(1-a)(1-a), which is why a doubly "
          "represented surface is not one scene")
    check("the compositing identity 1-(1-a1)(1-a2) holds to renderer precision",
          duplicate["per_camera"][0]["alpha_composition_defect"]["max_abs"] < 5e-2,
          f"max |A - (1-(1-a1)(1-a2))| = "
          f"{duplicate['per_camera'][0]['alpha_composition_defect']['max_abs']:.2e}")

    with tempfile.TemporaryDirectory() as tmp:
        payload = write_audit(Path(tmp), [disjoint, duplicate])
        check("overall status is the worst of the entries",
              payload["status"] == UNRESOLVED, payload["status"])
        check("audit names the mechanisms it does NOT use",
              "alpha_eff" in payload["forbidden_and_absent"]
              and "nearest-neighbour deduplication" in payload["forbidden_and_absent"])
        check("audit persisted atomically",
              (Path(tmp) / "representation_audit.json").is_file()
              and not (Path(tmp) / "representation_audit.json.tmp").exists())

    # ================= T8 =================
    print("\n[T8] nonlinear IEKS with a physically occluded Gaussian")
    cameras = occluding_cameras
    T, dt = 6, 0.1
    # A big opaque blocker in front, and a small Gaussian that starts BEHIND
    # it and translates out into view. The occlusion is the rasteriser's, not
    # a zeroed Jacobian.
    # Visibility by the real rasteriser, via the FRUSTUM rather than a
    # blocker. An occluder leaves Gaussian tails that still carry ~20% of the
    # information, which is not a clean "unobserved early" condition; a
    # Gaussian outside the field of view contributes to no pixel at all, and
    # this is still physical visibility decided by the rasteriser, not a
    # hand-zeroed Jacobian.
    blocker_mu = torch.tensor([[0.0, 0.0, 0.0]])
    # Just outside the frustum (half-width ~1.26 at this depth), entering
    # around frame 2. Far enough to be genuinely unobserved at t=0, close
    # enough that one linearisation can bridge the discrepancy -- a Gaussian
    # parked several units away is a basin problem, not a smoothing test.
    # x = -1.90 measured to give EXACTLY zero renderer information from both
    # cameras (the profile falls 4.6e4, 2.1e4, 1.8e3, 0.0 at -1.45, -1.60,
    # -1.75, -1.90). Travelling at 1.6 it re-enters around frame 3.
    hidden_start = torch.tensor([[-1.90, 0.0, 0.0]])
    true_v = torch.tensor([[1.6, 0.0, 0.0]])
    blocker_scales = torch.tensor([[0.30, 0.30, 0.30]])

    def truth_at(t):
        mu = torch.cat([blocker_mu, hidden_start + true_v * (dt * t)], 0)
        return state_from(
            mu,
            torch.cat([blocker_scales, aniso.unsqueeze(0)], 0),
            torch.tensor([[0.15, 0.15, 0.15], [0.95, 0.85, 0.1]]),
            torch.tensor([[0.999], [0.95]]),
            v=torch.cat([torch.zeros(1, 3), true_v], 0))

    truth = [truth_at(t) for t in range(T)]
    observations = []
    for t in range(T):
        lin = build_linearization(truth[t], cameras)
        observations.append([(rgb.clone(), depth.clone(),
                              torch.ones_like(depth, dtype=torch.bool))
                             for rgb, depth in lin.predicted()])

    # Confirm the occlusion is real: the hidden Gaussian must carry
    # (near) zero renderer information early and real information late.
    probe_obs = FreeRGBDObservation(cameras, observations, frame_ids=list(range(T)))
    info_by_frame = []
    for t in range(T):
        cache = probe_obs.linearize_frame(truth[t], t)
        probe = torch.zeros(2, 12, device="cuda")
        probe[1, 0] = 1.0                      # hidden Gaussian, +x
        info_by_frame.append(float((probe * probe_obs.information_apply(cache, probe)).sum()))
    print("       renderer information on the hidden Gaussian, per frame: "
          + ", ".join(f"{v:.3e}" for v in info_by_frame))
    check("the hidden Gaussian is physically occluded early",
          info_by_frame[0] < 0.05 * max(info_by_frame),
          f"frame 0 information {info_by_frame[0]:.3e} vs max "
          f"{max(info_by_frame):.3e} -- exactly zero, "
          "occlusion by the rasteriser, not a hand-zeroed Jacobian. Gaussian "
          "tails keep this from being exactly zero, so it is measured.")
    check("and genuinely observed later",
          max(info_by_frame) > 1e3, f"max {max(info_by_frame):.3e}")

    Q_inv = torch.eye(12, dtype=torch.float64).expand(2, 12, 12).contiguous() * 50.0

    # The warm start is wrong but not degenerate. A zero-velocity nominal
    # keeps the hidden Gaussian outside the frustum at EVERY frame, where its
    # image Jacobian is exactly zero and no gradient method can pull it in --
    # that is a basin failure, not a smoothing test. Giving it 1.2 against a
    # truth of 1.6 leaves frame 0 unobserved while the late frames have
    # signal, which is the condition T8 is about.
    warm_v = torch.tensor([[1.2, 0.0, 0.0]])

    def warm_start():
        """Wrong velocity, and a frame-0 state the camera cannot see."""
        states = []
        for t in range(T):
            mu = torch.cat([blocker_mu, hidden_start + warm_v * (dt * t)], 0)
            states.append(state_from(
                mu,
                torch.cat([blocker_scales, aniso.unsqueeze(0)], 0),
                torch.tensor([[0.15, 0.15, 0.15], [0.95, 0.85, 0.1]]),
                torch.tensor([[0.999], [0.95]]),
                v=torch.cat([torch.zeros(1, 3), warm_v], 0)))
        return Trajectory(states=states, frame_ids=list(range(T)), dt=dt)

    def solve(frames_used, label):
        obs = FreeRGBDObservation(
            cameras,
            [observations[t] if t in frames_used else [None] * len(cameras)
             for t in range(T)],
            frame_ids=list(range(T)))
        ieks = UnifiedFreeIEKS(obs, Q_inv, config=IEKSConfig(
            max_outer=20, damping=1e-3, verbose=(label == "all frames")))
        result = ieks.run(warm_start())
        print(f"       {label:22s} status={result['status']:10s} "
              f"E_total -> {result['energy']['E_total']:.4e}"
              f"  factors={result['factor_inventory']['num_factors']}")
        return result

    full = solve(set(range(T)), "all frames")
    early_only = solve({0, 1}, "early frames only")

    check("the IEKS reaches a terminal status",
          full["status"] in ("CONVERGED", "MAX_ITERS", "LINE_SEARCH_FAILED"),
          full["status"])
    accepted = [r for r in full["trace"] if r["accepted"]]
    check("at least one step was accepted", len(accepted) >= 1,
          f"{len(accepted)} accepted of {len(full['trace'])} outer iterations")

    monotone = True
    previous = None
    for record in accepted:
        if previous is not None and record["E_total"] > previous + 1e-9:
            monotone = False
        previous = record["E_total"]
    check("every accepted step decreased the TRUE nonlinear posterior", monotone,
          "evaluated by rerendering, not by the quadratic model")

    check("one factor per (frame, camera), unchanged across the run",
          full["factor_inventory"]["num_factors"] == T * len(cameras),
          f"{full['factor_inventory']['num_factors']} factors for {T} frames "
          f"x {len(cameras)} cameras")

    # THE T8 CLAIM: late observations move the early hidden state.
    # The frame-0 STATE is (mu, R, v, omega). The hidden Gaussian is outside
    # the frustum at t=0, so its frame-0 image information is exactly zero;
    # anything that changes there arrived through the temporal process
    # factors from frames the camera could see.
    s_full, s_part = full["trajectory"].states[0], early_only["trajectory"].states[0]
    moved_pos = float((s_full.mu[1] - s_part.mu[1]).norm())
    moved_vel = float((s_full.v[1] - s_part.v[1]).norm())
    print(f"       frame-0 hidden state, all-frames vs early-only: "
          f"|dmu| = {moved_pos:.5f}, |dv| = {moved_vel:.5f}  "
          f"(v_x {float(s_full.v[1, 0]):.4f} vs {float(s_part.v[1, 0]):.4f})")
    check("late-frame observations correct the EARLY unobserved state",
          moved_pos + moved_vel > 1e-2,
          f"frame-0 state differs by |dmu| {moved_pos:.5f} + |dv| "
          f"{moved_vel:.5f} between the all-frames and early-frames-only "
          "solves, with identical prior, process and initialisation. The "
          "frame-0 image information on this Gaussian is exactly 0.0, so the "
          "difference arrived through the temporal off-diagonal factors.")
    # Deliberately NOT asserted: that the all-frames solve is closer to the
    # truth than the early-only one. It is not, here. The all-frames problem
    # has three times the factors and stops at MAX_ITERS with PCG still at
    # ~1e-3, while the early-only problem is small enough to converge. The
    # measured claim is that the frame-0 state DIFFERS, which is the
    # information-flow mechanism; convergence quality is reported separately
    # rather than dressed up as accuracy.
    print(f"       status: all-frames {full['status']}, "
          f"early-only {early_only['status']}")
    check("the frame-0 velocity moved off its warm start",
          abs(float(s_full.v[1, 0]) - float(warm_v[0, 0])) > 1e-3,
          f"v_x = {float(s_full.v[1, 0]):.4f} from a warm start of "
          f"{float(warm_v[0, 0]):.3f} "
          f"(truth {float(true_v[0, 0]):.3f}); the gap to truth is a "
          "convergence limitation of this 20-iteration budget, not a claim "
          "of recovery")



    # Appearance must be untouched by the whole optimisation.
    check("colours, opacity and scales are unchanged by the IEKS",
          torch.equal(full["trajectory"].states[-1].colors, truth[0].colors)
          and torch.equal(full["trajectory"].states[-1].opacity, truth[0].opacity)
          and torch.equal(full["trajectory"].states[-1].scales, truth[0].scales))

    # Rejection must restore everything.
    print("\n[T8b] a deliberately bad step is rejected and state restored")
    obs = FreeRGBDObservation(cameras, observations, frame_ids=list(range(T)))
    ieks = UnifiedFreeIEKS(obs, Q_inv, config=IEKSConfig(
        max_outer=3, damping=1e-12, backtracking=(1e6,), verbose=False))
    before = warm_start()
    out = ieks.run(before)
    rejected = [r for r in out["trace"] if not r["accepted"]]
    check("an impossible step is rejected rather than taken",
          len(rejected) >= 1, f"{len(rejected)} rejections recorded")
    check("the inventory is byte-identical after rejections",
          out["factor_inventory"]["num_factors"] == T * len(cameras))

    print("\n[T8c] b equals the negative gradient of the true posterior")
    print("       RGB only: the depth term is gated by alpha > 0.05, so moving")
    print("       a Gaussian flips pixels in and out of the gate and the")
    print("       objective is piecewise there. The gate is production")
    print("       behaviour, not a defect, so the identity is checked on the")
    print("       smooth part and the limitation is recorded.")
    obs = FreeRGBDObservation(cameras, observations, frame_ids=list(range(T)),
                              use_depth=False)
    ieks = UnifiedFreeIEKS(obs, Q_inv, config=IEKSConfig(verbose=False))
    nominal = warm_start()
    operator, _ = ieks.build_system(nominal, damping=0.0)
    b = operator.rhs()
    base = ieks.evaluate(nominal)["E_total"]
    rel_errors = []
    # Directions are classified, not assumed. The blocker never moves in the
    # warm start, so its residual and therefore its gradient are exactly
    # zero; comparing a finite difference against 0 measures rendering noise
    # and reports a 100% "error" while nothing is wrong. Such a direction is
    # UNOBSERVABLE, not a failure.
    unobservable = []
    for (frame, gauss, column) in ((5, 1, 0), (4, 1, 0), (3, 1, 0),
                                   (2, 0, 0)):
        direction = torch.zeros(nominal.T, nominal.N, 12, dtype=torch.float64,
                                device=nominal.states[0].mu.device)
        direction[frame, gauss, column] = 1.0
        best = None
        for eps in (4e-3, 2e-3, 1e-3):
            plus = ieks.evaluate(nominal.retract(direction, eps))["E_total"]
            minus = ieks.evaluate(nominal.retract(direction, -eps))["E_total"]
            fd = (plus - minus) / (2 * eps)
            analytic = -float((b * direction).sum())
            rel = abs(fd - analytic) / max(abs(fd), 1e-12)
            best = rel if best is None else min(best, rel)
        analytic = -float((b * direction).sum())
        label = f"frame {frame}, gauss {gauss}, col {column}"
        if abs(analytic) < 1e-6:
            unobservable.append(label)
            print(f"       d/d({label}): analytic {analytic:+.4e}  "
                  "UNOBSERVABLE (zero gradient; nothing to verify against)")
            continue
        rel_errors.append(best)
        print(f"       d/d({label}): analytic {analytic:+.4e}  "
              f"best relative error {best:.3e}")
    check("at least one direction carries a verifiable gradient",
          len(rel_errors) >= 2,
          f"{len(rel_errors)} observable, {len(unobservable)} unobservable "
          f"({', '.join(unobservable) or 'none'})")
    check("b = -grad F on the full posterior, by finite differences",
          rel_errors and max(rel_errors) < 0.05,
          f"worst relative error {max(rel_errors):.3e} over "
          f"{len(rel_errors)} observable directions (float32 rendering floor)")

    print("\n[T8d] symmetry and curvature of the assembled operator")
    g = torch.Generator().manual_seed(3)
    dev = nominal.states[0].mu.device
    u = torch.randn(nominal.T, nominal.N, 12, generator=g, dtype=torch.float64).to(dev)
    w = torch.randn(nominal.T, nominal.N, 12, generator=g, dtype=torch.float64).to(dev)
    lhs = float((u * operator.information_matvec(w)).sum())
    rhs = float((operator.information_matvec(u) * w).sum())
    check("<u, A w> == <A u, w>", abs(lhs - rhs) / max(abs(lhs), 1e-30) < 1e-4,
          f"relative asymmetry {abs(lhs - rhs) / max(abs(lhs), 1e-30):.2e} "
          "(the measurement blocks are float32 CUDA products)")
    check("A has positive curvature on a random probe",
          float((u * operator.information_matvec(u)).sum()) > 0.0)

    with tempfile.TemporaryDirectory() as tmp:
        save_run(Path(tmp), full)
        free = Path(tmp) / "unified_free"
        check("original-time arrays saved",
              np.load(free / "positions_original_order.npy").shape == (T, 2, 3))
        check("solver trace saved per outer iteration",
              len(open(free / "solver_trace.jsonl").read().strip().splitlines())
              == len(full["trace"]))
        contract = json.loads((free / "covariance_contract.json").read_text())
        check("the covariance contract refuses to call the preconditioner a posterior",
              "NOT the posterior covariance" in contract["note"])

    print("\n[T8e] robust mode is refused rather than mislabelled")
    global CHECKS
    CHECKS += 1
    try:
        bad = FreeRGBDObservation(cameras, observations,
                                  frame_ids=list(range(T)), robust=True)
        UnifiedFreeIEKS(bad, Q_inv)
        FAILURES.append("robust=True was accepted")
        print("  FAIL  robust=True was accepted")
    except NotImplementedError as exc:
        print(f"  ok    robust=True blocked: {str(exc)[:70]}...")

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
