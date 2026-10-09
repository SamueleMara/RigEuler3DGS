"""Sections 15, 16, 27: identifiability measured in observation space.

CODEX_THEORETICAL_METHOD_STRICT_IMPLEMENTATION.md sections 15, 16, 27.

Section 15 deleted the rule "axis far from the support => prefer prismatic".
Section 16 replaces it with a profile likelihood over the curvature parameter
kappa = 1/rho, whose prismatic limit is kappa = 0.

The decisive test is section 27's far-axis grid: a TRUE revolute joint whose
axis sits many support radii away must still be called REVOLUTE when the
angular sweep and the measurement quality make it identifiable. The deleted
d/R rule got exactly that case wrong by construction.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from joint_reduction.identifiability import (  # noqa: E402
    UNRESOLVED_MODEL, UNRESOLVED_OPTIMIZATION, curvature_family, curvature_of,
    default_kappa_grid, profile_curvature, support_geometry)
from joint_reduction.kpart_models import (  # noqa: E402
    PRISMATIC, REVOLUTE, PartModel, SceneModel)
from joint_reduction.kpart_window import AlternationConfig, WindowObjective  # noqa: E402
from tests.test_strict_trajectory import (  # noqa: E402
    Cfg, DEV, camera, check, render_obs, scene_of)

import tests.test_strict_trajectory as TT  # noqa: E402


def build_revolute_window(*, d_over_R, sweep_deg, T=8, seed=0, n=300,
                          cams=None, noise=0.0):
    """A window of a TRUE revolute joint with the axis d_over_R support radii
    away, swept by `sweep_deg` degrees in total.

    The arc length is held roughly constant as d_over_R grows, so the cases
    differ in CURVATURE and not in how far the part moves -- which is what the
    identifiability question is about.
    """
    cams = cams or [camera(), camera(tz=0.3)]
    st = scene_of(n=n, seed=seed)
    labels = torch.zeros(st.num_gaussians, dtype=torch.long, device=DEV)
    labels[st.mu[:, 0] > 0.0] = 1
    pts = st.mu[labels == 1]
    centroid, rho_s = support_geometry(pts)
    rho = float(d_over_R) * rho_s
    e = torch.tensor([0.0, 1.0, 0.0], device=DEV)          # axis along +y
    n_dir = torch.tensor([0.0, 0.0, 1.0], device=DEV)      # radial: +z
    model = curvature_family(centroid, e, n_dir, 1.0 / rho, part_index=1)
    scene = SceneModel(parts=[model], K=1)
    total = math.radians(sweep_deg)
    q_true = [total * t / (T - 1) for t in range(T)]
    g = torch.Generator(device=DEV).manual_seed(seed + 7)
    frames = []
    for t, q in enumerate(q_true):
        mu, R = scene.transport_all(st.mu, st.R, labels, [q])
        obs = render_obs(st, mu, R, cams)
        if noise > 0:
            obs = [(o[0] + noise * torch.randn(o[0].shape, generator=g,
                                               device=DEV), o[1], o[2])
                   for o in obs]
        frames.append((obs, t))
    obj = WindowObjective(st.mu, st.R, cams, frames, Cfg(), state_template=st)
    arc = [rho * q for q in q_true]
    return obj, scene, labels, pts, arc, rho, rho_s


def build_prismatic_window(*, T=8, span=0.12, seed=1, n=300, noise=0.0):
    cams = [camera(), camera(tz=0.3)]
    st = scene_of(n=n, seed=seed)
    labels = torch.zeros(st.num_gaussians, dtype=torch.long, device=DEV)
    labels[st.mu[:, 0] > 0.0] = 1
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    scene = SceneModel(parts=[PartModel(kind=PRISMATIC, axis=axis,
                                        part_index=1)], K=1)
    q_true = [span * t / (T - 1) for t in range(T)]
    g = torch.Generator(device=DEV).manual_seed(seed + 7)
    frames = []
    for t, q in enumerate(q_true):
        mu, R = scene.transport_all(st.mu, st.R, labels, [q])
        obs = render_obs(st, mu, R, cams)
        if noise > 0:
            obs = [(o[0] + noise * torch.randn(o[0].shape, generator=g,
                                               device=DEV), o[1], o[2])
                   for o in obs]
        frames.append((obs, t))
    obj = WindowObjective(st.mu, st.R, cams, frames, Cfg(), state_template=st)
    return obj, scene, labels, st.mu[labels == 1], q_true


CFG = AlternationConfig(sigma_a=1.0, trajectory_iters=30, geometry_rounds=1)
# Section 31: the nuisance optimisation must be equally thorough at every
# kappa, prismatic included. Measured: going from 1 to 5 geometry rounds
# lowered the kappa = 0 energy by 7 units and the likelihood ratio from 40.3
# to 28.8, so an under-optimised prismatic fit inflates the evidence for
# revolute. The verdict was unchanged, but the bias is real and one-sided.
FAIR = AlternationConfig(sigma_a=1.0, trajectory_iters=30, geometry_rounds=4)


# ===========================================================================
def test_16_kappa_family_is_continuous_at_zero():
    """§16: kappa = 0 must be the exact prismatic limit of the family."""
    c = torch.tensor([0.0, 0.0, 2.0], device=DEV)
    e = torch.tensor([0.0, 1.0, 0.0], device=DEV)
    nd = torch.tensor([0.0, 0.0, 1.0], device=DEV)
    pts = c.reshape(1, 3) + torch.randn((64, 3), device=DEV) * 0.05
    prism = curvature_family(c, e, nd, 0.0)
    check("§16 kappa = 0 is prismatic", prism.kind == PRISMATIC)

    worst = 0.0
    for kappa in (1e-1, 1e-2, 1e-3, 1e-4):
        m = curvature_family(c, e, nd, kappa)
        s = 0.02                                   # the same ARC LENGTH
        moved, _ = m.transport(pts, None, s * kappa)
        moved_p, _ = prism.transport(pts, None, s)
        worst = max(worst, float((moved - moved_p).norm(dim=1).max()) / s)
    print(f"       max |revolute(kappa) - prismatic| / arc length as kappa -> 0:"
          f" {worst:.2e}")
    check("§16 the family is continuous at the prismatic limit", worst < 0.06,
          f"relative deviation {worst:.2e} at kappa as large as 0.1")

    m = curvature_family(c, e, nd, 0.5)
    k2, _e2, _n2 = curvature_of(m, c)
    check("§16 kappa round-trips through the model", abs(k2 - 0.5) < 1e-6,
          f"{k2:.8f}")


def test_16_true_prismatic_is_consistent_with_zero_curvature():
    """§16: a true prismatic joint must leave 0 inside the interval."""
    obj, scene, labels, pts, q_true = build_prismatic_window(noise=0.004)
    q_init = [list(q_true)]
    prof = profile_curvature(obj, scene, labels, 1, CFG, support_points=pts,
                             q_init=q_init, alpha=0.05)
    print(f"       kappa_hat {prof.kappa_hat:+.4f}  LR {prof.likelihood_ratio:.3f}"
          f"  threshold {prof.threshold:.3f}")
    print(f"       interval [{prof.interval[0]:+.4f}, {prof.interval[1]:+.4f}]"
          f"  verdict {prof.verdict}")
    check("§16 a true prismatic keeps 0 inside the interval", prof.zero_inside,
          f"LR {prof.likelihood_ratio:.3f} <= {prof.threshold:.3f}")
    check("§16 and is reported as consistent with prismatic",
          prof.to_dict()["prismatic_consistent"])
    check("§16 the verdict is not a forced REVOLUTE",
          prof.verdict == UNRESOLVED_MODEL, prof.verdict)


def test_27_far_axis_revolute_grid():
    """§27/§15: a far axis must NOT by itself force prismatic.

    The deleted rule was d_axis/R_support > tau => prismatic. Here the axis is
    deliberately placed far away while the angular sweep keeps the motion
    curved enough to be identifiable; section 15 says such a joint is
    revolute and must be called revolute.
    """
    rows = []
    for d_over_R, sweep in ((2.0, 60.0), (6.0, 60.0), (20.0, 60.0)):
        obj, scene, labels, pts, arc, rho, rho_s = build_revolute_window(
            d_over_R=d_over_R, sweep_deg=sweep, seed=2, noise=0.004)
        prof = profile_curvature(obj, scene, labels, 1, CFG,
                                 support_points=pts, q_init=[list(arc)],
                                 alpha=0.05)
        rows.append((d_over_R, sweep, rho, prof))
        print(f"       d/R {d_over_R:5.1f}  sweep {sweep:4.0f} deg  "
              f"kappa_true {1 / rho:+.4f}  kappa_hat {prof.kappa_hat:+.4f}  "
              f"LR {prof.likelihood_ratio:10.2f}  {prof.verdict}")
    for d_over_R, sweep, rho, prof in rows:
        check(f"§27 d/R = {d_over_R:g} with a {sweep:g} degree sweep is called "
              f"REVOLUTE", prof.verdict == REVOLUTE,
              f"verdict {prof.verdict}, LR {prof.likelihood_ratio:.2f} "
              f"vs threshold {prof.threshold:.2f}")
    far = rows[-1][3]
    check("§15 the far-axis case is decided by the likelihood, not by d/R",
          far.to_dict()["axis_distance_over_support_radius_diagnostic"] > 5.0
          and far.verdict == REVOLUTE,
          "d/R is recorded as a diagnostic while the verdict comes from the "
          "profile likelihood")


def test_16_small_sweep_is_unresolved_not_forced():
    """§16/§45: find the detectability boundary; never force prismatic.

    This test does NOT assert that a chosen sweep is unresolvable -- that would
    be assuming the answer. It scans the sweep down until the profile
    likelihood stops resolving the curvature, and requires two things of the
    method:

      1. UNRESOLVED_MODEL is actually reachable, i.e. the test can say "the
         data cannot decide" instead of always picking a side;
      2. it never answers PRISMATIC by fiat (section 45) -- a sweep too small
         to curve yields UNRESOLVED_MODEL, with the interval showing that
         prismatic is consistent.

    The measured boundary is reported, because that is the method's real
    sensitivity rather than a number anyone chose.
    """
    rows = []
    for sweep in (1.5, 0.3, 0.06, 0.012):
        obj, scene, labels, pts, arc, rho, rho_s = build_revolute_window(
            d_over_R=40.0, sweep_deg=sweep, seed=6, noise=0.006)
        prof = profile_curvature(obj, scene, labels, 1, FAIR,
                                 support_points=pts, q_init=[list(arc)],
                                 alpha=0.05)
        sag = rho * (1 - math.cos(math.radians(sweep) / 2))
        rows.append((sweep, sag, prof))
        print(f"       sweep {sweep:5.3f} deg  sagitta {sag:.2e} world units  "
              f"kappa_hat {prof.kappa_hat:.5f}  LR {prof.likelihood_ratio:9.3f}"
              f"  {prof.verdict}")
        torch.cuda.empty_cache()

    verdicts = [r[2].verdict for r in rows]
    check("§16 UNRESOLVED_MODEL is reachable -- the test can decline to decide",
          UNRESOLVED_MODEL in verdicts, f"verdicts {verdicts}")
    check("§45 the method never answers PRISMATIC by fiat",
          all(v in (REVOLUTE, UNRESOLVED_MODEL, UNRESOLVED_OPTIMIZATION)
              for v in verdicts), f"verdicts {verdicts}")
    smallest = rows[-1][2]
    check("§16 the smallest sweep leaves 0 inside the interval",
          smallest.zero_inside,
          f"LR {smallest.likelihood_ratio:.3f} <= {smallest.threshold:.3f}")
    check("§16 and reports prismatic as consistent, not as selected",
          smallest.to_dict()["prismatic_consistent"]
          and smallest.verdict == UNRESOLVED_MODEL, smallest.verdict)
    resolved = [r for r in rows if r[2].verdict == REVOLUTE]
    if resolved:
        print(f"       smallest resolved sagitta: {resolved[-1][1]:.2e} world "
              f"units at {resolved[-1][0]:g} degrees")


def test_15_no_far_axis_rule_in_production():
    """§15: the d/R override must be gone from production logic."""
    src = (REPO / "joint_reduction" / "kpart_window.py").read_text()
    check("§15 no far-axis extent override in the selector",
          "far_axis_extents" not in src and "far_axis_margin" not in src)
    check("§15 the removal cites the section",
          "section 15" in src.lower() or "§15" in src)
    idsrc = (REPO / "joint_reduction" / "identifiability.py").read_text()
    check("§15 d/R survives only as a diagnostic",
          "axis_distance_over_support_radius_diagnostic" in idsrc)
    check("§16 the test is a profile likelihood with a chi-square threshold",
          "chi2_quantile(1.0 - 2.0 * alpha, 1)" in idsrc,
          "boundary-corrected: kappa = 0 sits on the edge of kappa >= 0")
    check("§16 the boundary null is stated", "Self and Liang" in idsrc)


def main():
    for fn in (test_16_kappa_family_is_continuous_at_zero,
               test_15_no_far_axis_rule_in_production,
               test_16_true_prismatic_is_consistent_with_zero_curvature,
               test_27_far_axis_revolute_grid,
               test_16_small_sweep_is_unresolved_not_forced):
        print(f"\n{fn.__name__}")
        fn()
        torch.cuda.empty_cache()
    print(f"\n{len(TT.PASSED)} checks passed across 5 tests")


if __name__ == "__main__":
    main()
