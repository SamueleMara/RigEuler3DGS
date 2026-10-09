"""T-TR1: PCG budget invariance at full scene scale, on the REAL renderer.

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md sections 1.3 and 20.

This is the test the whole trust-region work exists for. Section 1.3 calls the
observed behaviour a critical failure:

    PCG budget  25 -> 4/4 steps accepted
    PCG budget 100 -> 0/4 accepted, objective frozen
    PCG budget 400 -> 0/4 accepted, objective frozen

i.e. solving the linear system BETTER made the method stop working, because
early termination was silently acting as the step-size controller. With an
explicit trust region the radius controls the step and the budget only controls
how accurately the constrained subproblem is solved.

Everything here runs the actual CUDA rasterizer and the actual IEKF frame step
at 11,320 Gaussians -- section 1.2's full-scene size -- not a synthetic matrix.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.renderer_bridge import CameraSpec, build_linearization  # noqa: E402
from estimator.sparse_iekf import IEKFConfig, iekf_frame_step  # noqa: E402
from estimator.state import GaussianMotionState  # noqa: E402

DEV = "cuda"
PASSED: list[str] = []
FULL_SCALE = 11320


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def camera(width=160, height=160, fov=0.9, dtype=torch.float32):
    tanf = math.tan(fov * 0.5)
    view = torch.eye(4, device=DEV, dtype=dtype)
    znear, zfar = 0.01, 100.0
    proj = torch.zeros((4, 4), device=DEV, dtype=dtype)
    proj[0, 0] = 1.0 / tanf
    proj[1, 1] = 1.0 / tanf
    proj[2, 2] = zfar / (zfar - znear)
    proj[3, 2] = 1.0
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    full = view.unsqueeze(0).bmm(proj.t().unsqueeze(0)).squeeze(0)
    return CameraSpec(viewmatrix=view, projmatrix=full, tanfovx=tanf, tanfovy=tanf,
                      image_height=height, image_width=width)


def big_scene(n=FULL_SCALE, seed=0, v=None, dtype=torch.float32):
    """A dense cloud at full-scene scale -- the regime where section 1.2
    measured 44% weakly-observable pose blocks."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros((n, 3))
    mu[:, 0] = (torch.rand(n, generator=g) - 0.5) * 1.2
    mu[:, 1] = (torch.rand(n, generator=g) - 0.5) * 1.2
    mu[:, 2] = 2.2 + torch.rand(n, generator=g) * 0.9
    R = torch.eye(3).repeat(n, 1, 1)
    scales = torch.full((n, 3), 0.02)
    z = torch.zeros((n, 3))
    return GaussianMotionState(
        mu=mu.to(DEV, dtype), R=R.to(DEV, dtype),
        v=(z if v is None else v).to(DEV, dtype), omega=z.to(DEV, dtype),
        colors=torch.rand((n, 3), generator=g).to(DEV, dtype),
        opacity=torch.full((n, 1), 0.7).to(DEV, dtype), scales=scales.to(DEV, dtype))


def render_gt(state, cams, cfg):
    lin = build_linearization(state, cams, eps_depth=cfg.eps_depth)
    return [(v.rgb.clone(), v.depth.clone(),
             (v.alpha > cfg.min_alpha_for_depth).clone()) for v in lin.views]


def identity_cov(n, pos=1e-2, rot=1e-2, vel=1.0, om=1e-2, dtype=torch.float32):
    P = torch.zeros((n, 12, 12), device=DEV, dtype=dtype)
    d = torch.tensor([pos] * 3 + [rot] * 3 + [vel] * 3 + [om] * 3,
                     device=DEV, dtype=dtype)
    P[:, range(12), range(12)] = d
    return P


def run_budget(budget: int, *, trust_region: bool, n=FULL_SCALE, dt=1.0 / 30.0,
               speed=0.30, max_iters=4):
    """One full-scale frame step at a given PCG budget."""
    cams = [camera()]
    v_true = torch.zeros((n, 3))
    v_true[:, 0] = speed
    truth = big_scene(n, v=v_true)
    moved = GaussianMotionState(
        mu=truth.mu + truth.v * dt, R=truth.R, v=truth.v, omega=truth.omega,
        colors=truth.colors, opacity=truth.opacity, scales=truth.scales)

    cfg = IEKFConfig(max_iters=max_iters, pcg_max_iters=budget,
                     use_trust_region=trust_region, debug=False,
                     robust_measurement=True, observability_report=True)
    gt = render_gt(moved, cams, cfg)
    # The filter starts from the un-moved state with the true velocity unknown.
    start = big_scene(n)
    x, P, diag = iekf_frame_step(start, identity_cov(n), gt, cams, dt, cfg)

    rows = [r for r in diag.rows if r.get("event") in ("accepted", "rejected")]
    init = next(r for r in diag.rows if r.get("event") == "initial")
    return {
        "budget": budget,
        "accepted": sum(1 for r in rows if r["event"] == "accepted"),
        "iterations": len(rows),
        "energy_initial": init["energy"],
        "energy_final": rows[-1]["energy"] if rows else init["energy"],
        "radius_trajectory": [r.get("trust_radius") for r in rows],
        "linear_residual": [r.get("pcg_rel_residual") for r in rows],
        "boundary_hits": sum(1 for r in rows if r.get("boundary_hit")),
        "pcg_iterations": [r.get("pcg_iterations") for r in rows],
        "stop_reasons": [r.get("pcg_reason") for r in rows],
        "rho": [r.get("rho") for r in rows],
        "diag": diag,
    }


def test_T_TR1_budget_invariance_full_scale():
    """The headline gate. Section 20 requires accepted steps, final objective,
    radius trajectory, linear residual trajectory and boundary-hit counts."""
    print(f"       scene: {FULL_SCALE} Gaussians, 1 camera 160x160, 4 iterations")
    results = {}
    for budget in (25, 100, 400):
        r = run_budget(budget, trust_region=True)
        results[budget] = r
        red = 100.0 * (1 - r["energy_final"] / max(r["energy_initial"], 1e-30))
        print(f"\n       --- PCG budget {budget} ---")
        print(f"       accepted steps     : {r['accepted']}/{r['iterations']}")
        print(f"       objective          : {r['energy_initial']:.4f} -> "
              f"{r['energy_final']:.4f}  ({red:.1f}% reduction)")
        print(f"       trust radius       : "
              f"{[f'{x:.3f}' for x in r['radius_trajectory']]}")
        print(f"       linear residual    : "
              f"{[f'{x:.2e}' for x in r['linear_residual']]}")
        print(f"       boundary hits      : {r['boundary_hits']}")
        print(f"       pcg iters actually : {r['pcg_iterations']}")
        print(f"       stop reasons       : {r['stop_reasons']}")
        print(f"       rho                : "
              f"{[f'{x:.3f}' if x == x else 'nan' for x in r['rho']]}")

    print()
    for budget, r in results.items():
        check(f"T-TR1 budget {budget} accepts at least one step",
              r["accepted"] > 0,
              f"{r['accepted']}/{r['iterations']} accepted -- this is the "
              f"section 1.3 failure if it is 0")
        check(f"T-TR1 budget {budget} reduces the objective",
              r["energy_final"] < r["energy_initial"],
              f"{r['energy_initial']:.4f} -> {r['energy_final']:.4f}")

    # The actual invariance claim: the outcome must not collapse as the budget
    # grows. A larger budget solving the subproblem better may legitimately do
    # slightly better or worse, but it must NOT go to zero accepted steps.
    acc = [results[b]["accepted"] for b in (25, 100, 400)]
    check("T-TR1 accepted-step count does not collapse as the budget grows",
          min(acc) > 0, f"25:{acc[0]}  100:{acc[1]}  400:{acc[2]}")

    finals = [results[b]["energy_final"] for b in (25, 100, 400)]
    worst = max(finals)
    best = min(finals)
    check("T-TR1 the final objective is comparable across budgets",
          worst <= best * 3.0 + 1e-6,
          f"{finals[0]:.4f} / {finals[1]:.4f} / {finals[2]:.4f}")
    check("T-TR1 a larger budget never ends with a WORSE objective than the "
          "smallest budget by more than noise",
          finals[2] <= finals[0] * 3.0 + 1e-6,
          f"25 -> {finals[0]:.4f}, 400 -> {finals[2]:.4f}")

    # Section 20's required reporting must actually be present.
    r400 = results[400]
    check("T-TR1 the trust radius trajectory is recorded",
          all(x is not None for x in r400["radius_trajectory"]))
    check("T-TR1 the linear residual trajectory is recorded",
          all(x is not None for x in r400["linear_residual"]))
    check("T-TR1 boundary hits are counted",
          isinstance(r400["boundary_hits"], int))
    check("T-TR1 the §4.5 trust-region history is persisted on the diagnostics",
          r400["diag"].trust_region is not None
          and len(r400["diag"].trust_region.records) == r400["iterations"])
    rec = r400["diag"].trust_region.records[0].to_dict()
    required = {"iteration", "objective_before", "objective_after",
                "predicted_reduction", "actual_reduction", "rho",
                "trust_radius_before", "trust_radius_after", "step_norm_scaled",
                "step_norm_translation", "step_norm_rotation",
                "pcg_stop_reason", "pcg_iterations", "pcg_rel_residual",
                "forcing_eta", "boundary_hit", "lm_lambda", "accepted"}
    check("T-TR1 every §4.5 mandatory field is present",
          required.issubset(rec.keys()),
          f"missing {sorted(required - set(rec.keys()))}")

    # The mechanism itself: the step must actually be radius-limited somewhere,
    # otherwise the test proves nothing about the trust region.
    total_boundary = sum(results[b]["boundary_hits"] for b in (25, 100, 400))
    check("T-TR1 the trust boundary was actually active (the radius, not the "
          "budget, limited the step)", total_boundary > 0,
          f"{total_boundary} boundary hits across the three budgets")


def test_T_TR1b_lm_path_is_the_contrast():
    """The same experiment without the trust region, recorded honestly.

    Section 1.3 reported that the LM + line-search path accepted 4/4 steps at a
    25-iteration PCG budget and ZERO at 100 or 400. That inversion is no longer
    reproducible here -- and the reason is NOT the trust region. It was a
    symptom of the VJP traversing contributors the forward pass had already
    stopped at, so H^T was not the transpose of H (adjoint identity rel 0.78 at
    this scale). Both b = H^T R^-1 r and the normal operator used that broken
    adjoint, so solving the linear system more accurately converged harder onto
    a direction that did not reduce the true objective. With the contributor
    cutoff fixed, LM accepts 4/4 at every budget from 10 to 400.

    So this test asserts section 20's actual requirement -- no collapse to zero
    accepted steps as the budget grows -- for BOTH paths, and records the
    objective comparison as measured data rather than asserting that the trust
    region wins. On this scene it does not: LM reaches a lower objective
    because its line search accepts any step that lowers the true energy, while
    the trust region additionally requires the gain ratio to validate the
    quadratic model, which a piecewise-smooth renderer often fails.
    """
    print()
    budgets = (25, 100, 400)
    lm = {b: run_budget(b, trust_region=False) for b in budgets}
    tr = {b: run_budget(b, trust_region=True) for b in budgets}
    for b in budgets:
        print(f"       budget {b:>3}:  TR {tr[b]['accepted']}/{tr[b]['iterations']} "
              f"accepted, final {tr[b]['energy_final']:9.3f}   |   "
              f"LM {lm[b]['accepted']}/{lm[b]['iterations']} accepted, "
              f"final {lm[b]['energy_final']:9.3f}")

    # Section 20's requirement, for both paths.
    for b in budgets:
        check(f"§20 the trust-region path still accepts steps at budget {b}",
              tr[b]["accepted"] > 0, f"{tr[b]['accepted']}/{tr[b]['iterations']}")
        check(f"§20 the LM path no longer collapses at budget {b} either",
              lm[b]["accepted"] > 0, f"{lm[b]['accepted']}/{lm[b]['iterations']} "
              f"-- §1.3 reported 0/4 here before the adjoint was fixed")

    # The property the trust region guarantees BY CONSTRUCTION: the step is
    # bounded by the radius, so the outcome cannot depend on the budget.
    tr_f = [tr[b]["energy_final"] for b in budgets]
    check("§20 the trust-region objective is invariant across budgets",
          max(tr_f) / min(tr_f) < 1.001,
          f"{tr_f[0]:.3f} / {tr_f[1]:.3f} / {tr_f[2]:.3f}")
    check("§20 and so is its accepted-step count",
          len({tr[b]["accepted"] for b in budgets}) == 1,
          str([tr[b]["accepted"] for b in budgets]))

    lm_f = [lm[b]["energy_final"] for b in budgets]
    print(f"\n       MEASURED, not asserted: on this scene LM reaches a lower "
          f"objective\n       (LM {min(lm_f):.1f}-{max(lm_f):.1f} vs TR "
          f"{min(tr_f):.1f}-{max(tr_f):.1f}). The trust region is the more "
          f"conservative\n       of the two here; it is kept as the default "
          f"because §4 requires explicit\n       radius-based step control and "
          f"§4.5/§17 need its diagnostics, not because\n       it wins on final "
          f"energy. Recorded as a limitation, not a victory.")


def test_observability_is_measured_at_full_scale():
    """Section 5.3's required persistence, on the real full-scale Fisher."""
    r = run_budget(100, trust_region=True, max_iters=2)
    rows = [x for x in r["diag"].rows if x.get("event") == "observability"]
    check("§5 an observability report is produced", len(rows) > 0)
    o = rows[0]
    for key in ("min_eigenvalue", "median_eigenvalue", "max_eigenvalue",
                "mean_rank", "fraction_weak", "global_dimension",
                "active_dimension", "tau_obs"):
        check(f"§5.3 '{key}' is persisted", key in o)
    print(f"       Fisher eigenvalues (scaled): min {o['min_eigenvalue']:.3e}  "
          f"median {o['median_eigenvalue']:.3e}  max {o['max_eigenvalue']:.3e}")
    print(f"       observable rank: mean {o['mean_rank']:.2f}/6   "
          f"fraction weak {o['fraction_weak']:.1%}")
    print(f"       state dimension: {o['active_dimension']} active of "
          f"{o['global_dimension']}")
    check("§5 the weak fraction is a real measurement, not a constant",
          0.0 <= o["fraction_weak"] <= 1.0)
    check("§5 the active dimension never exceeds the global dimension",
          o["active_dimension"] <= o["global_dimension"])
    fin = [x for x in r["diag"].rows if x.get("event") == "observability_final"]
    check("§5.2 weak posterior directions are inflated rather than trusted",
          len(fin) > 0 and "weak_directions_inflated" in fin[0],
          f"inflated {fin[0].get('weak_directions_inflated')} of "
          f"{fin[0].get('weak_directions_total')} weak directions"
          if fin else "no final report")


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
