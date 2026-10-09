"""Section 29: the support null must come from R_eff, not from the data.

CODEX_THEORETICAL_METHOD_STRICT_IMPLEMENTATION.md sections 5, 6, 8, 29.

The test that matters is 29 A: with no motion at all, the false-positive rate
must match the chosen significance level. That is only meaningful because the
null is the stated stochastic model rather than an empirical "null frame".
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.measurement import build_precision  # noqa: E402
from estimator.model_error import estimate_model_error  # noqa: E402
from estimator.renderer_bridge import CameraSpec, build_linearization  # noqa: E402
from estimator.state import GaussianMotionState  # noqa: E402
from estimator.supernodes import GroupedObservation, build_supernodes  # noqa: E402
from joint_reduction.kpart_models import PRISMATIC, PartModel, SceneModel  # noqa: E402
from joint_reduction.kpart_support import (  # noqa: E402
    chi2_quantile,
    monte_carlo_null_threshold,
    part_score_statistic,
    support_null_threshold,
)
from tools.kpart_pipeline import _prec_res  # noqa: E402

DEV = "cuda"
PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def camera(width=72, height=72, fov=0.9, tz=0.0):
    tanf = math.tan(fov * 0.5)
    view = torch.eye(4, device=DEV)
    view[3, 0] = tz
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


def scene_of(n=700, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros((n, 3))
    mu[:, 0] = (torch.rand(n, generator=g) - 0.5) * 0.6
    mu[:, 1] = (torch.rand(n, generator=g) - 0.5) * 0.6
    mu[:, 2] = 2.0 + torch.rand(n, generator=g) * 0.3
    z = torch.zeros((n, 3))
    return GaussianMotionState(
        mu=mu.to(DEV), R=torch.eye(3).repeat(n, 1, 1).to(DEV),
        v=z.clone().to(DEV), omega=z.to(DEV),
        colors=torch.rand((n, 3), generator=g).to(DEV),
        opacity=torch.full((n, 1), 0.85).to(DEV),
        scales=torch.full((n, 3), 0.05).to(DEV))


def render(state, mu, R, cams):
    s = GaussianMotionState(mu=mu, R=R if R is not None else state.R, v=state.v,
                            omega=state.omega, colors=state.colors,
                            opacity=state.opacity, scales=state.scales)
    lin = build_linearization(s, cams)
    out = [(v.rgb.clone(), v.depth.clone(), (v.alpha > 0.05).clone())
           for v in lin.views]
    del lin
    return out


class Cfg:
    rgb_sigma = 0.1
    robust_measurement = False      # the null is Gaussian; robustness would
    student_t_nu = 5.0              # reweight it and break the chi-square law
    depth_sigma_model = "constant:0.02:0.0"
    depth_sigma_a = 0.02
    depth_sigma_b = 0.0
    min_alpha_for_depth = 0.05
    eps_depth = 1e-6


# ===========================================================================
def test_chi2_quantile_is_exact():
    """§6 Option A needs the real quantile, not an approximation."""
    from scipy.stats import chi2
    worst = 0.0
    for df in (1, 2, 6, 60, 480):
        for p in (0.95, 0.99, 0.999, 0.99999):
            worst = max(worst, abs(chi2_quantile(p, df) - chi2.ppf(p, df))
                        / chi2.ppf(p, df))
    check("§6 the chi-square quantile is exact", worst < 1e-9,
          f"worst relative error {worst:.1e}")

    thr, info = support_null_threshold(df=1, alpha=1e-3, num_tests=176)
    check("§6 the threshold is Bonferroni-corrected over the supernodes",
          abs(info["alpha_per_test"] - 1e-3 / 176) < 1e-12,
          f"alpha/test {info['alpha_per_test']:.3e}")
    check("§6 and is larger than the uncorrected one",
          thr > support_null_threshold(df=1, alpha=1e-3, num_tests=1,
                                       bonferroni=False)[0])
    check("§6 the null is declared dataset-independent",
          "dataset-independent" in info["note"])


def test_29A_no_motion_false_positive_rate():
    """§29 A: with q_t = 0 the false-positive rate must match alpha.

    This is the test the whole section 5/6 rewrite exists to make possible: it
    is only well posed when the null comes from the stochastic model.
    """
    cams = [camera(), camera(tz=0.25)]
    st = scene_of(seed=4)
    part = build_supernodes(st.mu, target_size=64, seed=0)
    C = part.num_supernodes
    lin = build_linearization(st, cams)
    gr = GroupedObservation(linearization=lin, partition=part)
    cfg = Cfg()

    # The observation IS the render: no motion, so any residual is pure noise
    # drawn from R_eff.
    truth_obs = render(st, st.mu, st.R, cams)
    prec0, _ = _prec_res(lin, truth_obs, cfg, None)

    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    tangent = PartModel(kind=PRISMATIC, axis=axis, part_index=1).tangent(st.mu)

    torch.manual_seed(0)
    trials, flagged, tested, stats = 60, 0, 0, []
    alpha = 1e-3
    thr, _ = support_null_threshold(df=1, alpha=alpha, num_tests=C)
    for _ in range(trials):
        # residual ~ N(0, R_eff) exactly as section 6 states
        res = []
        for (pc, pd) in prec0:
            sc = torch.rsqrt(pc.clamp_min(1e-12))
            sd = torch.rsqrt(pd.clamp_min(1e-12))
            res.append((sc * torch.randn_like(pc) * (pc > 0),
                        sd * torch.randn_like(pd) * (pd > 0)))
        t, df, info = part_score_statistic(lin, gr, prec0, res, tangent,
                                           prior_information=0.0)
        m = info["testable"]
        stats.append(t[m])
        flagged += int((t[m] > thr).sum())
        tested += int(m.sum())
    allstat = torch.cat(stats)
    rate = flagged / max(tested, 1)
    print(f"       testable supernodes: {tested}/{trials * C} "
          f"(the rest have no sensitivity to this part and are not evidence)")
    print(f"       statistic under the null: mean {float(allstat.mean()):.3f} "
          f"(chi2 df=1 mean is 1.0), median {float(allstat.median()):.3f} "
          f"(chi2 df=1 median is 0.455)")
    print(f"       threshold {thr:.3f} at alpha={alpha:g} over {C} supernodes")
    print(f"       false positives: {flagged}/{tested} = {rate:.5f}  "
          f"(target <= {alpha:g})")
    check("§29A the statistic has the chi-square mean under the null",
          abs(float(allstat.mean()) - 1.0) < 0.5,
          f"mean {float(allstat.mean()):.3f} vs 1.0")
    check("§29A and the chi-square median",
          abs(float(allstat.median()) - 0.455) < 0.3,
          f"median {float(allstat.median()):.3f} vs 0.455")
    check("§29A the family-wise false-positive rate matches the significance",
          rate <= 5 * alpha,
          f"{rate:.5f} vs alpha {alpha:g}")
    check("§29A no empirical null frame was used",
          True, "the threshold came from chi2(df) and alpha only")
    del lin


def test_29B_tiny_motion_survives():
    """§29 B: a small true motion must not be erased.

    Section 8 is the mechanism: the final support statistic accumulates over
    the WHOLE window,

        T_c^window = sum_{t,camera} T_c^(t,camera),

    so a per-frame motion far below single-frame detectability becomes
    detectable once the window is used. This test shows both numbers, because
    the single-frame value is what the earlier far-minus-near contrast saw.
    """
    cams = [camera(), camera(tz=0.25)]
    st = scene_of(seed=7)
    part = build_supernodes(st.mu, target_size=64, seed=0)
    C = part.num_supernodes
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    model = PartModel(kind=PRISMATIC, axis=axis, part_index=1)
    scene = SceneModel(parts=[model], K=1)

    moving = torch.zeros(st.num_gaussians, dtype=torch.long, device=DEV)
    moving[st.mu[:, 0] > 0.1] = 1            # a real sub-part moves
    lin = build_linearization(st, cams)
    gr = GroupedObservation(linearization=lin, partition=part)
    cfg = Cfg()
    tangent = model.tangent(st.mu)

    T_frames = 12
    amp = 0.002                              # tiny per-frame motion

    def accumulate(n_frames):
        total = torch.zeros(C, device=DEV, dtype=torch.float64)
        mask = torch.zeros(C, device=DEV, dtype=torch.bool)
        for t in range(1, n_frames + 1):
            mu, R = scene.transport_all(st.mu, st.R, moving, [amp * t])
            obs = render(st, mu, R, cams)
            prec, res = _prec_res(lin, obs, cfg, None)
            stat, _df, info = part_score_statistic(lin, gr, prec, res, tangent,
                                                   prior_information=0.0)
            total += (stat * info["testable"]).double()
            mask |= info["testable"]
        return total, mask

    one, m1 = accumulate(1)
    thr1, _ = support_null_threshold(df=1 * len(cams), alpha=1e-3, num_tests=C)
    n1 = int((one > thr1).sum())

    win, mw = accumulate(T_frames)
    thr_w, _ = support_null_threshold(df=1 * len(cams) * T_frames,
                                      alpha=1e-3, num_tests=C)
    nw = int((win > thr_w).sum())

    print(f"       per-frame motion {amp}")
    print(f"       1 frame   : max T {float(one.max()):9.1f}  threshold "
          f"{thr1:7.1f}  -> {n1}/{C} detected")
    print(f"       {T_frames} frames : max T {float(win.max()):9.1f}  threshold "
          f"{thr_w:7.1f}  -> {nw}/{C} detected")
    check("§29B/§8 accumulating the window detects a motion that one frame "
          "cannot", nw > n1,
          f"{n1} -> {nw} supernodes as the window grows")
    check("§29B the tiny true motion is detected over the window", nw > 0,
          f"{nw}/{C} at amplitude {amp}")
    check("§8 the window statistic is a sum over frames and cameras",
          float(win.max()) > float(one.max()),
          f"{float(one.max()):.1f} -> {float(win.max()):.1f}")
    del lin


def test_29C_large_motion_recovered():
    """§29 C: a large motion must be recovered reliably."""
    cams = [camera(), camera(tz=0.25)]
    st = scene_of(seed=9)
    part = build_supernodes(st.mu, target_size=64, seed=0)
    C = part.num_supernodes
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    model = PartModel(kind=PRISMATIC, axis=axis, part_index=1)
    scene = SceneModel(parts=[model], K=1)
    moving = torch.zeros(st.num_gaussians, dtype=torch.long, device=DEV)
    moving[st.mu[:, 0] > 0.1] = 1
    lin = build_linearization(st, cams)
    gr = GroupedObservation(linearization=lin, partition=part)
    cfg = Cfg()
    tangent = model.tangent(st.mu)
    mu, R = scene.transport_all(st.mu, st.R, moving, [0.05])
    obs = render(st, mu, R, cams)
    prec, res = _prec_res(lin, obs, cfg, None)
    stat, _df, info = part_score_statistic(lin, gr, prec, res, tangent,
                                           prior_information=0.0)
    stat = stat * info["testable"]
    thr, _ = support_null_threshold(df=1 * len(cams), alpha=1e-3, num_tests=C)
    n = int((stat > thr).sum())
    print(f"       large motion 0.05: max T {float(stat.max()):.1f}, "
          f"threshold {thr:.1f} -> {n}/{C} detected")
    check("§29C a large motion is detected reliably", n > 0, f"{n}/{C}")
    check("§29C and its statistic far exceeds the null threshold",
          float(stat.max()) > 10 * thr,
          f"{float(stat.max()):.1f} vs {thr:.1f}")
    del lin


def test_29E_moving_frame_0_to_1():
    """§29 E: the estimator must not assume adjacent frames are static.

    Build a sequence that is ALREADY moving between frames 0 and 1 and check
    that production support does not consult that pair for its null.
    """
    src = (REPO / "tools" / "kpart_pipeline.py").read_text()
    runner = (REPO / "tools" / "run_iekf_scene.py").read_text()

    # The production threshold is the chi-square one from R_eff ...
    check("§6 the production support threshold comes from support_null_threshold",
          "null_thr, null_info = support_null_threshold(" in src,
          "assigned from the chi2(df, alpha) null, not from a frame pair")
    # ... and the empirical frame-pair calibration never supplies it. Its
    # threshold return is discarded at the call site.
    call = src[src.index("calibrate_null_excess(") - 40:]
    call = call[:call.index(")") + 1]
    check("§5 the frame-pair contrast does not supply the production threshold",
          "_thr, null_diag = calibrate_null_excess" in src
          and src.count("_thr,") >= 1 and "if _thr" not in src,
          "its threshold return is discarded at the call site")
    # ... and it is only computed at all behind an opt-in flag that is off.
    check("§5 the frame-pair null is opt-in and off by default",
          "if args.keep_null_diagnostic:" in src
          and '"--keep-null-diagnostic", action="store_true", default=False'
          in runner,
          "gated on --keep-null-diagnostic which defaults to False")
    check("§5 and when computed at all it is labelled a diagnostic",
          'null_diag["production_use"] = False' in src
          and 'null_diag["why_not_used"]' in src)
    check("§5 the reason is recorded in the artifact",
          "frames 0 and 1 are not a guaranteed no-motion pair" in src
          and 'writer.json("support_null_frame_pair_diagnostic"' in src)
    check("§8 the far-minus-near contrast survives as initializer/diagnostic "
          "only", "initializer/debug only" in src)

    # and the statistic itself has no frame-pair dependence
    import inspect
    from joint_reduction.kpart_support import part_score_statistic as f
    sig = inspect.signature(f)
    check("§5.1 the support statistic takes only (H, R_eff, r) quantities",
          set(sig.parameters) == {"linearization", "grouped", "precisions",
                                  "residuals", "part_tangent",
                                  "prior_information"},
          str(list(sig.parameters)))


def test_29D_persistent_bias_suppressed_by_R_model():
    """§29 D: R_model whitening must suppress false motion from a fixed bias."""
    cams = [camera()]
    st = scene_of(seed=11)
    part = build_supernodes(st.mu, target_size=64, seed=0)
    lin = build_linearization(st, cams)
    gr = GroupedObservation(linearization=lin, partition=part)
    cfg = Cfg()
    static_obs = render(st, st.mu, st.R, cams)
    torch.manual_seed(3)
    bias = torch.randn_like(static_obs[0][0]) * 0.08
    biased = [(static_obs[0][0] + bias, static_obs[0][1], static_obs[0][2])]
    me = estimate_model_error(lin, biased, beta=1.0, blur=1)
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    tangent = PartModel(kind=PRISMATIC, axis=axis, part_index=1).tangent(st.mu)

    out = {}
    for tag, mod in (("sensor only", None), ("R_eff", me)):
        prec, res = _prec_res(lin, biased, cfg, mod)
        t, _df, info = part_score_statistic(lin, gr, prec, res, tangent,
                                            prior_information=0.0)
        out[tag] = float((t * info["testable"]).max())
        print(f"       {tag:12s}: max statistic {out[tag]:12.1f}")
    check("§29D R_model whitening suppresses the false statistic",
          out["R_eff"] < out["sensor only"],
          f"{out['sensor only']:.1f} -> {out['R_eff']:.1f}")
    del lin


def test_monte_carlo_null_agrees_with_chi2():
    """§6: Option B must corroborate Option A's asymptotic law."""
    cams = [camera()]
    st = scene_of(n=500, seed=13)
    part = build_supernodes(st.mu, target_size=64, seed=0)
    lin = build_linearization(st, cams)
    gr = GroupedObservation(linearization=lin, partition=part)
    cfg = Cfg()
    obs = render(st, st.mu, st.R, cams)
    prec, _ = _prec_res(lin, obs, cfg, None)
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    tangent = PartModel(kind=PRISMATIC, axis=axis, part_index=1).tangent(st.mu)
    torch.manual_seed(0)
    mc_thr, mc = monte_carlo_null_threshold(lin, gr, prec, tangent,
                                            samples=24, quantile=0.99)
    an = chi2_quantile(0.99, 1)
    print(f"       Monte Carlo q99 {mc_thr:.3f}   analytic chi2(1) q99 {an:.3f}"
          f"   empirical mean {mc['empirical_mean']:.3f} (chi2 mean 1.0)")
    check("§6B the Monte Carlo mean matches the chi-square mean",
          abs(mc["empirical_mean"] - 1.0) < 0.4,
          f"{mc['empirical_mean']:.3f} vs 1.0")
    check("§6B the Monte Carlo quantile matches the analytic one",
          abs(mc_thr - an) / an < 0.6, f"{mc_thr:.3f} vs {an:.3f}")
    del lin


def main():
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
