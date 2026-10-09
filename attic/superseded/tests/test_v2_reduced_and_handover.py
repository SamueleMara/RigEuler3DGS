"""V2 sections 14, 15, 17, 18: handover, reduced-filter validation,
certificate V2 and the multi-criterion coherence gate.

The §15 trajectories are run through the REAL renderer: the reduced normal
system comes from the CUDA JVP/VJP at each frame, so "the measurement
contributes" is a statement about actual image information.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.certificate_v2 import (  # noqa: E402
    CERTIFIED_LOCAL_MAP,
    CONVERGED_BUT_UNOBSERVABLE,
    LINEAR_SOLVE_INEXACT,
    REJECTED_MODEL,
    TRUST_REGION_STALLED,
    CertificateThresholds,
    assess,
    normalized_optimality,
)
from estimator.measurement import build_precision  # noqa: E402
from estimator.observability import analyze_blocks  # noqa: E402
from estimator.renderer_bridge import CameraSpec, build_linearization  # noqa: E402
from estimator.state import GaussianMotionState  # noqa: E402
from estimator.trust_region import (  # noqa: E402
    STOP_BOUNDARY,
    STOP_FORCING,
    STOP_MAX_ITERS,
    TrustRegionHistory,
    TrustRegionRecord,
)
from joint_reduction.coherence_gate import (  # noqa: E402
    CoherenceGate,
    GateConfig,
    WindowSample,
    angle_between,
    jaccard,
    relative_speed_change,
)
from joint_reduction.handover import canonical_handover, canonical_hash  # noqa: E402
from joint_reduction.joint_filter import JointState  # noqa: E402
from joint_reduction.reduced_validation import (  # noqa: E402
    MANDATORY_TRAJECTORIES,
    WRONG_INITIAL,
    TrajectoryResult,
    run_process_only,
    run_trajectory,
    summarize,
)
from joint_reduction.renderer_reduction import (  # noqa: E402
    RenderJointModel,
    joint_normal_system,
)

DEV = "cuda"
PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def camera(width=72, height=72, fov=0.9):
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


def part_scene(n=160, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros((n, 3))
    mu[:, 0] = (torch.rand(n, generator=g) - 0.5) * 0.35
    mu[:, 1] = (torch.rand(n, generator=g) - 0.5) * 0.35
    mu[:, 2] = 2.0 + torch.rand(n, generator=g) * 0.2
    z = torch.zeros((n, 3))
    return GaussianMotionState(
        mu=mu.to(DEV), R=torch.eye(3).repeat(n, 1, 1).to(DEV),
        v=z.clone().to(DEV), omega=z.to(DEV),
        colors=torch.rand((n, 3), generator=g).to(DEV),
        opacity=torch.full((n, 1), 0.85).to(DEV),
        scales=torch.full((n, 3), 0.055).to(DEV))


def at(state, mu, R):
    return GaussianMotionState(mu=mu, R=R, v=state.v, omega=state.omega,
                               colors=state.colors, opacity=state.opacity,
                               scales=state.scales)


def rgbd_at(state, model, q, cams):
    mu, R = model.transport(state.mu, state.R, float(q))
    lin = build_linearization(at(state, mu, R), cams)
    return [(v.rgb.clone(), v.depth.clone(), (v.alpha > 0.05).clone())
            for v in lin.views]


def precisions_residuals(lin, gt):
    prec, res, energy = [], [], 0.0
    for view, (rgb_gt, d_gt, d_valid) in zip(lin.views, gt):
        rr = rgb_gt - view.rgb
        pr = build_precision(rr, valid_mask=None, sigma_rgb=0.1, robust=True,
                             nu=5.0, layout=("rgb", "rgb", "rgb"), channel_dim=0)
        rd = d_gt - view.depth
        pd = build_precision(rd.unsqueeze(0), valid_mask=d_valid.unsqueeze(0),
                             sigma_rgb=(), depth_sigma_model="constant:0.02:0.0",
                             depth_range=d_gt.unsqueeze(0), robust=True, nu=5.0,
                             layout=("depth",), channel_dim=0).squeeze(0)
        prec.append((pr, pd))
        res.append((rr, rd))
        energy += float(0.5 * (pr * rr * rr).sum() + 0.5 * (pd * rd * rd).sum())
    return prec, res, energy


# ===========================================================================
# section 15
# ===========================================================================
def test_section15_reduced_filter_trajectories():
    """All six mandatory trajectories, each run process-only and fused."""
    cams = [camera()]
    state = part_scene()
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    model = RenderJointModel(kind="prismatic", axis=axis)
    dt, n_frames = 1.0 / 20.0, 14

    results = []
    for name, q_of_t in MANDATORY_TRAJECTORIES.items():
        gt_cache = {t: rgbd_at(state, model, q_of_t((t + 1) * dt), cams)
                    for t in range(n_frames)}

        def normal_system(t, q_lin, _gt=gt_cache):
            mu, R = model.transport(state.mu, state.R, float(q_lin))
            lin = build_linearization(at(state, mu, R), cams)
            prec, res, e = precisions_residuals(lin, _gt[t])
            # NB the signature is (residuals, precisions), not the other way
            # round: b_q survives a swap because the product commutes, but the
            # Fisher would get weighted by the signed residuals and can then
            # come out negative.
            info, b = joint_normal_system(lin, model, mu, res, prec)
            return info, b, e

        off = WRONG_INITIAL.get(name, {"qdot0_offset": 0.0, "q0_offset": 0.0})
        # The filter starts at the truth EXCEPT where the trajectory is
        # deliberately mis-initialized (section 15 cases 5 and 6).
        true_qdot0 = (q_of_t(dt) - q_of_t(0.0)) / dt
        s0 = JointState(
            torch.tensor(q_of_t(0.0) + off["q0_offset"], dtype=torch.float64),
            torch.tensor(true_qdot0 + off["qdot0_offset"], dtype=torch.float64),
            torch.tensor([[1e-2, 0.0], [0.0, 1e-1]], dtype=torch.float64))

        # The acceleration of every one of these trajectories is genuinely
        # unmodelled by the constant-velocity process, so Q must be non-zero.
        # Both modes get the same accel_psd, so the comparison stays fair: it
        # changes the covariance, not the mean prediction.
        psd = 10.0
        proc = run_process_only(name, q_of_t, s0, dt, n_frames, accel_psd=psd)
        fused = run_trajectory(name, q_of_t, s0, dt, normal_system, n_frames,
                               accel_psd=psd)
        r = TrajectoryResult(name=name, process_only=proc, fused=fused)
        results.append(r)
        print(r.format())

    s = summarize(results)
    for r in results:
        check(f"§15 '{r.name}': fused beats process-only",
              r.fused_beats_process,
              f"process {r.process_only.rmse:.6f} vs fused {r.fused.rmse:.6f}")
        check(f"§15 '{r.name}': the visual update is not negligible",
              r.visual_update_is_meaningful,
              f"max |q_visual_delta| {r.fused.max_visual_delta:.6e}")
    check("§15 every mandatory trajectory passes", not s["failing"],
          f"failing: {s['failing']}")

    # The nine mandatory per-frame fields must all be present.
    row = results[0].fused.logs[0].to_dict()
    for key in ("q_pred", "qdot_pred", "q_visual_delta", "qdot_visual_delta",
                "q_post", "qdot_post", "prior_energy", "measurement_energy",
                "joint_information"):
        check(f"§15 per-frame log contains '{key}'", key in row)


def test_section15_constant_speed_would_be_vacuous():
    """The control section 15 is implicitly asking for: with constant speed,
    process-only ALREADY nails it, so a pass there proves nothing."""
    dt, n = 1.0 / 20.0, 12
    q_of_t = lambda t: 0.4 * t          # noqa: E731
    s0 = JointState(torch.tensor(0.0, dtype=torch.float64),
                    torch.tensor(0.4, dtype=torch.float64),
                    torch.tensor([[1e-2, 0.0], [0.0, 1e-1]], dtype=torch.float64))
    proc = run_process_only("constant", q_of_t, s0, dt, n)
    print(f"       constant-speed process-only RMSE: {proc.rmse:.3e}")
    check("§15 constant speed is predicted exactly by the prior alone",
          proc.rmse < 1e-12,
          f"RMSE {proc.rmse:.3e} -- so it cannot demonstrate that the "
          f"measurement contributes, which is why §15 excludes it")
    check("§15 the mandatory list therefore excludes constant speed",
          "constant" not in MANDATORY_TRAJECTORIES
          and len(MANDATORY_TRAJECTORIES) == 6,
          f"{sorted(MANDATORY_TRAJECTORIES)}")


# ===========================================================================
# section 14
# ===========================================================================
def test_section14_handover_refit():
    """The handover must not inherit free-stage drift, and the canonical
    reference must be bitwise unchanged across it."""
    cams = [camera()]
    state = part_scene(seed=4)
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    model = RenderJointModel(kind="prismatic", axis=axis)
    dt = 1.0 / 20.0
    true_q = [0.02, 0.04, 0.06, 0.08, 0.10]
    frames = [rgbd_at(state, model, q, cams) for q in true_q]

    mu0, R0 = state.mu.clone(), state.R.clone()
    h0 = canonical_hash(mu0, R0)

    def energy_fn(frame, mu, R):
        lin = build_linearization(at(state, mu, R), cams)
        _, _, e = precisions_residuals(lin, frame)
        return e

    # The free stage "drifted": its q estimates are biased AND its geometry
    # carries per-Gaussian noise no rigid joint can reproduce.
    torch.manual_seed(5)
    drift = torch.randn_like(mu0) * 0.01
    free_q = [q + 0.025 for q in true_q]
    free_geom = (model.transport(mu0 + drift, R0, free_q[-1]))

    res = canonical_handover(mu0, R0, model, frames, energy_fn,
                             free_stage_q=free_q, dt=dt,
                             free_stage_geometry=free_geom)
    print(f"       energy before refit (rigid @ free-stage q): "
          f"{res.energy_before['total']:.4f}")
    print(f"       energy after  refit (rigid @ canonical)   : "
          f"{res.energy_after['total']:.4f}")
    print(f"       free-stage deformed geometry total        : "
          f"{res.energy_before['free_stage_deformed_total']:.4f}")
    print(f"       q refit: {[f'{q:.5f}' for q in res.q_refit]}")
    print(f"       q true : {[f'{q:.5f}' for q in true_q]}")

    check("§14 the refit reduces the handover energy", res.improved,
          f"{res.energy_before['total']:.4f} -> {res.energy_after['total']:.4f}")
    check("§14 the canonical reference is bitwise unchanged",
          res.canonical_unchanged and res.canonical_hash_before == h0,
          res.canonical_hash_before[:16])
    err = max(abs(a - b) for a, b in zip(res.q_refit, true_q))
    check("§14 the refitted bootstrap trajectory recovers the true q",
          err < 0.01, f"max |q_refit - q_true| = {err:.5f}")
    check("§14 the free stage's biased q is corrected",
          max(abs(a - b) for a, b in zip(free_q, true_q)) > err,
          f"free-stage bias 0.025 -> refit error {err:.5f}")
    check("§14 (q_L, qdot_L) are initialized from the window fit",
          abs(res.q_final - true_q[-1]) < 0.01
          and abs(res.qdot_final - 0.4) < 0.15,
          f"q_L {res.q_final:.5f}, qdot_L {res.qdot_final:.4f} (true 0.4)")
    check("§14 the posterior covariance is positive definite",
          bool((torch.linalg.eigvalsh(res.covariance) > 0).all()))

    with tempfile.TemporaryDirectory() as d:
        paths = res.write(d)
        for f in ("handover_before_energy.json", "handover_after_refit_energy.json",
                  "bootstrap_q_refit.csv", "canonical_reference_hash.json"):
            check(f"§14 artifact '{f}' is written",
                  f in paths and Path(paths[f]).exists())


# ===========================================================================
# section 17
# ===========================================================================
def rec(**kw):
    base = dict(iteration=0, objective_before=10.0, objective_after=1.0,
                predicted_reduction=9.0, actual_reduction=9.0, rho=1.0,
                trust_radius_before=1.0, trust_radius_after=1.0,
                step_norm_scaled=1e-4, step_norm_translation=1e-4,
                step_norm_rotation=1e-5, pcg_stop_reason=STOP_FORCING,
                pcg_iterations=5, pcg_rel_residual=1e-6, forcing_eta=1e-3,
                boundary_hit=False, lm_lambda=0.0, accepted=True,
                gradient_norm_scaled=1e-6, optimality_eta_g=1e-6)
    base.update(kw)
    return TrustRegionRecord(**base)


def history(*records):
    h = TrustRegionHistory()
    for r in records:
        h.add(r)
    return h


def test_section17_statuses_are_distinct():
    fisher = torch.eye(6).reshape(1, 6, 6).repeat(40, 1, 1) * 100.0
    strong = analyze_blocks(fisher, tau=1e-2)
    weak = analyze_blocks(fisher * 1e-8, tau=1e-2)

    c = assess(history(rec()), strong, innovation_stats={"nis": 1.0})
    check("§17 a clean run is CERTIFIED_LOCAL_MAP",
          c.status == CERTIFIED_LOCAL_MAP, c.status + f" failing {c.failed}")

    c2 = assess(history(rec(accepted=False, rho=-1.0)), strong)
    check("§17 nothing accepted is REJECTED_MODEL",
          c2.status == REJECTED_MODEL, c2.status)

    c3 = assess(history(rec(trust_radius_after=1e-9)), strong)
    check("§17 a collapsed radius is TRUST_REGION_STALLED",
          c3.status == TRUST_REGION_STALLED, c3.status)

    c4 = assess(history(rec(pcg_stop_reason=STOP_MAX_ITERS)), strong)
    check("§17 an unlabelled linear solve is LINEAR_SOLVE_INEXACT",
          c4.status == LINEAR_SOLVE_INEXACT, c4.status)

    c5 = assess(history(rec()), weak, innovation_stats={"nis": 1.0})
    check("§17 a converged but unmeasured state is CONVERGED_BUT_UNOBSERVABLE",
          c5.status == CONVERGED_BUT_UNOBSERVABLE,
          f"{c5.status}, fraction weak {weak.fraction_weak:.1%}")
    check("§17 and that is NOT reported as certified", not c5.is_certified)

    check("§17 the statuses are genuinely distinct",
          len({c.status, c2.status, c3.status, c4.status, c5.status}) == 5)


def test_section17_monotone_alone_is_not_enough():
    """The headline rule: a monotone objective must not by itself certify."""
    fisher = torch.eye(6).reshape(1, 6, 6).repeat(20, 1, 1) * 100.0
    strong = analyze_blocks(fisher, tau=1e-2)
    # monotone and accepted, but the gradient is still large
    c = assess(history(rec(gradient_norm_scaled=1e3)), strong,
               innovation_stats={"nis": 1.0})
    check("§17 monotone + accepted with a large gradient is NOT certified",
          not c.is_certified, c.status)
    check("§17 and the failing condition is named",
          "4_first_order_optimality_small" in c.failed, str(c.failed))
    check("§17 the thresholds are persisted with the verdict",
          "eta_g_max" in c.to_dict()["thresholds"])
    check("§17 all ten conditions are recorded",
          len(c.conditions) == 10, f"{len(c.conditions)} conditions")

    # a weak direction claimed as recovered must break certification
    c2 = assess(history(rec()), strong, innovation_stats={"nis": 1.0},
                weak_directions_claimed=3)
    check("§17 claiming an unsupported weak direction breaks certification",
          not c2.is_certified and "9_no_weak_direction_claimed" in c2.failed)

    # non-finite innovation must break certification
    c3 = assess(history(rec()), strong, innovation_stats={"nis": float("nan")})
    check("§17 a non-finite innovation statistic breaks certification",
          not c3.is_certified and "7_innovation_finite" in c3.failed)


def test_section17_normalized_optimality():
    check("§17 eta_g shrinks as the state grows",
          normalized_optimality(1.0, 0.0) > normalized_optimality(1.0, 99.0))
    check("§17 eta_g = ||g|| / (1 + ||delta||)",
          abs(normalized_optimality(2.0, 3.0) - 0.5) < 1e-12)
    check("§17 a non-finite gradient gives infinite eta_g",
          normalized_optimality(float("nan"), 1.0) == float("inf"))


# ===========================================================================
# section 18
# ===========================================================================
def sample(frame, omega, v, residual=0.05, sigma=0.1, support=None,
           status=CERTIFIED_LOCAL_MAP, model="prismatic", rank=6):
    return WindowSample(
        frame=frame, omega=torch.tensor(omega, dtype=torch.float64),
        v=torch.tensor(v, dtype=torch.float64), residual=residual,
        residual_sigma=sigma, twist_covariance_trace=1e-4,
        observable_rank=rank,
        support=(torch.ones(10) if support is None else support),
        certificate_status=status, model_name=model, model_score=0.1)


def test_section18_gate_requires_all_criteria():
    g = CoherenceGate(GateConfig(window=3))
    for t in range(3):
        g.push(sample(t, [0.0, 0.0, 1.0], [0.3, 0.0, 0.0]))
    out = g.evaluate()
    check("§18 a stable window reduces", out["reduce"], str(out["reason"]))
    check("§18 all required quantities are reported",
          all(k in out["criteria"] for k in
              ("omega_turn_deg", "speed_change", "normalized_residual",
               "support_jaccard", "min_observable_rank",
               "twist_covariance_trace", "model_names")))

    # a turning axis blocks reduction
    g2 = CoherenceGate(GateConfig(window=3))
    g2.push(sample(0, [0.0, 0.0, 1.0], [0.3, 0.0, 0.0]))
    g2.push(sample(1, [0.0, 0.5, 1.0], [0.3, 0.0, 0.0]))
    g2.push(sample(2, [0.0, 1.0, 1.0], [0.3, 0.0, 0.0]))
    o2 = g2.evaluate()
    check("§18 a still-turning twist axis blocks reduction", not o2["reduce"])
    check("§18 and says so", any("turning" in r for r in o2["reason"]),
          str(o2["reason"]))

    # a changing speed blocks reduction
    g3 = CoherenceGate(GateConfig(window=3))
    for t, s in enumerate([0.1, 0.3, 0.9]):
        g3.push(sample(t, [0.0, 0.0, 1.0], [s, 0.0, 0.0]))
    o3 = g3.evaluate()
    check("§18 a still-changing speed blocks reduction", not o3["reduce"])
    check("§18 and says so", any("speed" in r for r in o3["reason"]),
          str(o3["reason"]))

    # unstable support blocks reduction
    g4 = CoherenceGate(GateConfig(window=3))
    sup = [torch.ones(10), torch.cat([torch.ones(5), torch.zeros(5)]),
           torch.cat([torch.zeros(5), torch.ones(5)])]
    for t in range(3):
        g4.push(sample(t, [0.0, 0.0, 1.0], [0.3, 0.0, 0.0], support=sup[t]))
    o4 = g4.evaluate()
    check("§18 an unstable support set blocks reduction", not o4["reduce"])
    check("§18 and says so", any("support" in r for r in o4["reason"]),
          str(o4["reason"]))

    # an uncertified state blocks reduction
    g5 = CoherenceGate(GateConfig(window=3))
    for t in range(3):
        g5.push(sample(t, [0.0, 0.0, 1.0], [0.3, 0.0, 0.0],
                       status=TRUST_REGION_STALLED))
    o5 = g5.evaluate()
    check("§18 an uncertified state update blocks reduction", not o5["reduce"])

    # a flip-flopping model choice blocks reduction
    g6 = CoherenceGate(GateConfig(window=3))
    for t, mname in enumerate(["prismatic", "revolute", "prismatic"]):
        g6.push(sample(t, [0.0, 0.0, 1.0], [0.3, 0.0, 0.0], model=mname))
    o6 = g6.evaluate()
    check("§18 an unstable model choice blocks reduction", not o6["reduce"])

    # low observable rank blocks reduction
    g7 = CoherenceGate(GateConfig(window=3))
    for t in range(3):
        g7.push(sample(t, [0.0, 0.0, 1.0], [0.3, 0.0, 0.0], rank=1))
    check("§18 an unobservable twist blocks reduction",
          not g7.evaluate()["reduce"])


def test_section18_residual_is_uncertainty_normalized():
    """The explicit point of §18: the residual is judged against its own
    uncertainty, NOT against a hand-raised absolute threshold."""
    # A residual of 0.5 passes when its uncertainty is large...
    g = CoherenceGate(GateConfig(window=2, max_normalized_residual=3.0))
    for t in range(2):
        g.push(sample(t, [0.0, 0.0, 1.0], [0.3, 0.0, 0.0],
                      residual=0.5, sigma=0.4))
    check("§18 a large residual passes when its own uncertainty is large",
          g.evaluate()["reduce"], "0.5 / 0.4 = 1.25 sigma")
    # ...and the SAME residual fails when the fit is precise.
    g2 = CoherenceGate(GateConfig(window=2, max_normalized_residual=3.0))
    for t in range(2):
        g2.push(sample(t, [0.0, 0.0, 1.0], [0.3, 0.0, 0.0],
                       residual=0.5, sigma=0.01))
    o = g2.evaluate()
    check("§18 the same residual fails when the fit is precise",
          not o["reduce"], "0.5 / 0.01 = 50 sigma")
    check("§18 so the gate is not a raised absolute threshold",
          any("uncertainty" in r for r in o["reason"]), str(o["reason"]))

    check("§18 an incomplete window never reduces",
          not CoherenceGate(GateConfig(window=4)).evaluate()["reduce"])


def test_section18_primitives():
    a = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    b = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    check("§18 angle_between is in degrees",
          abs(angle_between(a, b) - 90.0) < 1e-9, f"{angle_between(a, b)}")
    check("§18 a zero twist has no defined direction",
          math.isnan(angle_between(a, torch.zeros(3, dtype=torch.float64))))
    v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    v2 = torch.tensor([1.5, 0.0, 0.0], dtype=torch.float64)
    check("§18 relative speed change", abs(relative_speed_change(v2, v1) - 0.5) < 1e-9)
    check("§18 Jaccard of identical supports is 1",
          abs(jaccard(torch.ones(5), torch.ones(5)) - 1.0) < 1e-12)
    check("§18 Jaccard of disjoint supports is 0",
          jaccard(torch.tensor([1, 1, 0, 0]), torch.tensor([0, 0, 1, 1])) == 0.0)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
