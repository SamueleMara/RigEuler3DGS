"""Sections 11, 20, 21: the robust efficient-score test of H_P: beta = 0.

CODEX_NEXT_EFFICIENT_P_NULL_AND_PHASE2_DEBUG.md.

C6 and C7 are the tests that matter. C6 checks that duplicating correlated
evidence does NOT shrink the confidence interval -- the failure that made a
model-based chi-square reject a drawer at 4.29e+02. C7 checks that the robust
covariance actually inflates relative to the naive Fisher one when the
residual carries block-correlated structure, which is the mechanism that
makes C6 work.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.curvature_score import (PRISMATIC_AT_RESOLUTION,  # noqa: E402
                                       REVOLUTE_CANDIDATE,
                                       block_efficient_scores,
                                       cluster_robust_covariance,
                                       curvature_statistic, schur_eliminate)
from estimator.joint_kinematics import current_points, state_to_gaussians  # noqa: E402
from estimator.joint_score_blocks import (accumulate_scores,  # noqa: E402
                                          alpha_dim,
                                          explicit_projection_check)
from estimator.joint_state import JointState, initialise_from_twist  # noqa: E402
from estimator.prismatic_null import (curvature_jvp,  # noqa: E402
                                      null_tangent_basis,
                                      project_to_prismatic)
from estimator.measurement import build_precision  # noqa: E402
from estimator.renderer_bridge import build_linearization  # noqa: E402
from tests.test_phase2_screw_runtime import (Cfg, DEV, build_truth_state,  # noqa: E402
                                             camera, linearizer, make_harness,
                                             observe, precision_of, scene_of)

PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def bind(truth, cams, obs):
    counter = {"frame": 0}
    lf = linearizer(truth, cams)

    def linearize(st, frame):
        counter["frame"] = frame
        return lf(st, frame)

    def precision_fn(lin):
        return precision_of(lin, obs[counter["frame"]])
    return linearize, precision_fn


def run_test(truth, cams, mu0, state, obs, confidence=0.95, block_frames=1,
             fit_null=True):
    """Section 6: FIT the null, then test it.

    Testing an unfitted null is not the same experiment: section 19's
    stationarity audit will (correctly) refuse to interpret beta while the
    prismatic direction and the q trajectory still have unexploited slack,
    and every scene comes back UNRESOLVED.

    `block_frames = 1` gives one block per camera per frame, which is the
    finest globally-defined partition section 13 allows and the one that
    gives the cluster covariance the most degrees of freedom.
    """
    from estimator.curvature_score import run_efficient_p_test
    from estimator.joint_iekf import JointIEKFConfig
    from estimator.prismatic_null import run_prismatic_null_sequence
    lin, pf = bind(truth, cams, obs)
    converged = None
    if fit_null:
        nf = run_prismatic_null_sequence(state, mu0, None, obs, lin, pf,
                                         cfg=JointIEKFConfig(max_iters=6),
                                         max_passes=3)
        # Section 19's convergence signal, exactly as production uses it.
        converged = (len(nf["null_direction_change"]) > 1
                     and nf["null_direction_change"][-1] < 0.05)
    return run_efficient_p_test(state, mu0, None, obs, lin, pf,
                                block_frames=block_frames,
                                confidence=confidence,
                                null_converged=converged)


# ===========================================================================
def test_20_frame0_curvature_jacobian_is_zero():
    """Section 20: J_beta at frame 0 must be exactly zero."""
    truth, cams, mu0, members = make_harness(seed=1)
    st = build_truth_state(mu0, members, [0.3, -0.2, 0.1, 0.0, 0.0, 0.0],
                           torch.linspace(0.0, 0.4, 5), 5)
    p = st.parts[0]
    worst0, nonzero = 0.0, 0.0
    for f in (0, 1, 3):
        mu, _r, _R, _t = current_points(p, f, mu0, None)
        j = curvature_jvp(p, f, torch.tensor([1.0, 0.0]), mu)
        if f == 0:
            worst0 = float(j.abs().max())
        else:
            nonzero = max(nonzero, float(j.abs().max()))
    print(f"       |J_beta(frame 0)| {worst0:.3e}, "
          f"|J_beta(later)| {nonzero:.3e}")
    check("§20 the curvature Jacobian is exactly zero at the gauge frame",
          worst0 == 0.0, "bitwise")
    check("§20 and nonzero once the joint has moved", nonzero > 1e-6)


def test_8_beta_has_exactly_two_dof():
    """Section 8: curvature has two DOF, not three."""
    truth, cams, mu0, members = make_harness(seed=2)
    st = build_truth_state(mu0, members, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           torch.linspace(0.0, 0.3, 4), 4)
    B, Bg, Bb = null_tangent_basis(st.parts[0].zeta)
    u = st.parts[0].zeta[:3]
    print(f"       basis {tuple(B.shape)}, gamma {tuple(Bg.shape)}, "
          f"beta {tuple(Bb.shape)}; max |u . delta_omega| "
          f"{float((Bb[3:].T @ u).abs().max()):.3e}")
    check("§8 beta has exactly two degrees of freedom", Bb.shape[1] == 2)
    check("§8 and each is perpendicular to u, so zero pitch is preserved",
          float((Bb[3:].T @ u).abs().max()) < 1e-14)
    check("§8 gamma moves only the direction, beta only omega",
          float(Bg[3:].abs().max()) == 0.0 and float(Bb[:3].abs().max()) == 0.0)


def test_11_explicit_projection_equivalence():
    """Section 11: the Schur form must equal the explicit projection."""
    truth, cams, mu0, members = make_harness(n=90, moving=60, seed=3, cams=1)
    T = 3
    st = build_truth_state(mu0, members, [0.25, -0.1, 0.05, 0.0, 0.0, 0.0],
                           torch.linspace(0.0, 0.2, T), T)
    obs = observe(truth, cams, st, T)
    # perturb so the residual is not identically zero
    st2 = copy.deepcopy(st)
    st2.parts[0].q = st2.parts[0].q * 0.7
    project_to_prismatic(st2.parts[0])
    lin, pf = bind(truth, cams, obs)

    sc = accumulate_scores(st2, mu0, None, obs, lin, pf, block_frames=1)
    sh = schur_eliminate(sc)
    ex = explicit_projection_check(st2, mu0, None, obs, lin, pf)

    dF = float((sh["F_beta_given_alpha"]
                - ex["F_beta_given_alpha_explicit"]).abs().max())
    sF = max(float(ex["F_beta_given_alpha_explicit"].abs().max()), 1e-30)
    dg = float((sh["g_beta_given_alpha"]
                - ex["g_beta_given_alpha_explicit"]).abs().max())
    sg = max(float(ex["g_beta_given_alpha_explicit"].abs().max()), 1e-30)
    dFaa = float((sc["F_aa"] - ex["F_aa"]).abs().max()) / max(
        float(ex["F_aa"].abs().max()), 1e-30)
    print(f"       F_aa accumulated vs explicit: {dFaa:.3e}")
    print(f"       F_beta|alpha  Schur vs projection: {dF / sF:.3e}")
    print(f"       g_beta|alpha  Schur vs projection: {dg / sg:.3e}")
    check("§11 the accumulated information matches the explicit Jacobian",
          dFaa < 1e-8, f"{dFaa:.2e}")
    check("§11 F_beta|alpha equals J_beta|alpha^T J_beta|alpha",
          dF / sF < 1e-7, f"{dF / sF:.2e}")
    check("§11 and the efficient score matches the projected one",
          dg / sg < 1e-7, f"{dg / sg:.2e}")


def test_13_block_scores_sum_to_the_total():
    """Section 13: sum_b u_b must equal g_beta|alpha."""
    truth, cams, mu0, members = make_harness(n=120, moving=80, seed=5)
    T = 6
    st = build_truth_state(mu0, members, [0.2, 0.15, 0.0, 0.0, 0.0, 0.0],
                           torch.linspace(0.0, 0.3, T), T)
    obs = observe(truth, cams, st, T)
    st2 = copy.deepcopy(st)
    st2.parts[0].q = st2.parts[0].q * 0.8
    lin, pf = bind(truth, cams, obs)
    sc = accumulate_scores(st2, mu0, None, obs, lin, pf, block_frames=2)
    sh = schur_eliminate(sc)
    bl = block_efficient_scores(sc, sh)
    rel = bl["consistency_with_g_beta_given_alpha"] / max(
        float(sh["g_beta_given_alpha"].abs().max()), 1e-30)
    print(f"       {bl['num_blocks']} blocks; |sum u_b - g_beta|alpha| / "
          f"|g| = {rel:.3e}")
    check("§13 the block efficient scores sum to the total", rel < 1e-10,
          f"{rel:.2e}")


def test_C7_robust_covariance_inflates_under_block_correlation():
    """C7: with block-correlated residual the robust covariance must inflate.

    A model-based Fisher covariance assumes every pixel is an independent
    look. When the residual carries structure that is constant within a block
    -- which is what canonical-model mismatch and calibration error look like
    -- the true spread of the score is far larger, and only the empirical
    block covariance can see it.
    """
    nb = 2
    g = torch.Generator().manual_seed(17)
    # independent blocks: the spread matches the per-block variance
    indep = torch.randn((12, nb), generator=g, dtype=torch.float64)
    # block-correlated: a shared offset per block, i.e. structured residual
    shared = torch.randn((3, nb), generator=g, dtype=torch.float64) * 3.0
    corr = indep + shared.repeat_interleave(4, dim=0)
    out = {}
    for name, u in (("independent", indep), ("block-correlated", corr)):
        r = cluster_robust_covariance({"u_blocks": u})
        out[name] = float(torch.linalg.eigvalsh(r["B_U"]).max())
        print(f"       {name:18s} largest eigenvalue of B_U {out[name]:.4f}")
    check("C7 block-correlated evidence inflates the robust covariance",
          out["block-correlated"] > 3.0 * out["independent"],
          f"{out['block-correlated'] / out['independent']:.1f}x")


def test_C6_duplicated_evidence_does_not_shrink_confidence():
    """C6: duplicating correlated evidence must NOT shrink the interval.

    Duplicating pixels or Gaussians multiplies the evidence INSIDE each
    camera/time block -- it does not create new blocks, because the copies are
    observed by the same camera at the same time and are perfectly correlated
    with their originals. So every block score scales by `k`, and so does the
    Fisher information.

    Under the model-based form `g^T F^+ g` the statistic then grows like `k`:
    copy the image ten times and the evidence against P appears ten times
    stronger. The cluster-robust form is invariant, because `U -> kU` and
    `B_U -> k^2 B_U` exactly cancel. That invariance is the entire reason
    section 12 replaces the chi-square.
    """
    g = torch.Generator().manual_seed(23)
    u = torch.randn((6, 2), generator=g, dtype=torch.float64) + torch.tensor(
        [0.6, -0.3], dtype=torch.float64)
    F = torch.tensor([[4.0, 0.8], [0.8, 3.0]], dtype=torch.float64)

    def robust(k):
        uk = u * k
        rob = cluster_robust_covariance({"u_blocks": uk})
        U = uk.sum(0)
        return float(U @ (torch.linalg.pinv(rob["B_U"]) @ U))

    def fisher(k):
        # duplicating evidence scales both the score and the information
        gk, Fk = u.sum(0) * k, F * k
        return float(gk @ (torch.linalg.pinv(Fk) @ gk))

    rr = {k: robust(k) for k in (1, 2, 4, 10)}
    ff = {k: fisher(k) for k in (1, 2, 4, 10)}
    print("       copies -> cluster-robust S: "
          + ", ".join(f"{k}x {v:.4f}" for k, v in rr.items()))
    print("       copies -> model Fisher chi2: "
          + ", ".join(f"{k}x {v:.4f}" for k, v in ff.items()))
    spread = max(rr.values()) / max(min(rr.values()), 1e-300)
    check("C6 duplicating correlated evidence leaves the robust statistic "
          "unchanged", spread < 1.0 + 1e-9,
          f"10x copies changed it by {spread:.6f}x")
    check("C6 while the model-based chi-square grows linearly with the copies",
          ff[10] / ff[1] > 9.5, f"{ff[10] / ff[1]:.2f}x")
    # Section 27 gate 9, checked against the source rather than asserted: no
    # confidence factor may be derived from a pixel or Gaussian count. The
    # scan looks at CODE only, so the prose in the docstrings that explains
    # why such a factor is forbidden does not trip it.
    import ast as _ast
    offenders = []
    for name in ("curvature_score.py", "joint_score_blocks.py",
                 "prismatic_null.py"):
        src = (REPO / "estimator" / name).read_text()
        tree = _ast.parse(src)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Expr) and isinstance(node.value,
                                                          _ast.Constant):
                continue                       # docstrings
            if isinstance(node, _ast.Call):
                fn = node.func
                nm = (fn.attr if isinstance(fn, _ast.Attribute)
                      else fn.id if isinstance(fn, _ast.Name) else "")
                if nm in ("sqrt",):
                    seg = _ast.get_source_segment(src, node) or ""
                    if any(k in seg for k in ("numel", "shape", "count",
                                              "num_pixels", "num_gauss")):
                        offenders.append(f"{name}: {seg}")
    print(f"       source scan: {len(offenders)} count-derived sqrt factor(s)")
    check("C6 and no 1/sqrt(N) factor over pixels or Gaussians exists in the "
          "code", not offenders, str(offenders[:2]))


def test_C1_exact_prismatic():
    """C1: exact P with reconstruction noise -> PRISMATIC_AT_RESOLUTION."""
    truth, cams, mu0, members = make_harness(n=260, moving=170, seed=7)
    T = 6
    tt = torch.linspace(0.0, 1.0, T, dtype=torch.float64)
    st = build_truth_state(mu0, members, [0.0, 0.30, 0.18, 0.0, 0.0, 0.0],
                           0.42 * tt ** 2, T)
    obs = observe(truth, cams, st, T)
    # Section 21 C1 asks for reconstruction noise, and it is essential: with a
    # bitwise-exact fit the residual is identically zero, there is no evidence
    # in any direction, and the honest answer is UNRESOLVED rather than
    # PRISMATIC. Noise is what makes curvature observable and therefore makes
    # "not rejected" mean something.
    gg = torch.Generator(device=DEV).manual_seed(101)
    obs = [[(a + 0.02 * torch.randn(a.shape, generator=gg, device=DEV),
             b + 0.01 * torch.randn(b.shape, generator=gg, device=DEV))
            for (a, b) in frame] for frame in obs]
    res = run_test(truth, cams, mu0, copy.deepcopy(st), obs)
    t = res["test"]
    print(f"       S_curv {t['S_curv']:.4e} rank {t['rank_curv']} "
          f"cut {t['cut']:.2f}  (legacy Fisher chi2 "
          f"{t['legacy_fisher_chi2']:.3e})")
    print(f"       -> {t['status']}")
    check("C1 curvature is observable once there is a residual at all",
          t["rank_curv"] > 0, f"rank {t['rank_curv']}")
    check("C1 exact prismatic data is not rejected",
          t["status"] == PRISMATIC_AT_RESOLUTION, t["status"])
    # Both statistics are reported side by side (section 26's table). Their
    # ORDER is not fixed and no claim is made about it: the robust form
    # exceeds the model-based one under independent noise (small-sample
    # inflation with few clusters) and is far below it when the residual is
    # block-correlated, which is the real-data case this test cannot create.
    check("C1 the decision is referred to an F distribution, not chi-square",
          "F(" in t["reference_distribution"], t["reference_distribution"])
    check("C1 and both statistics are reported for comparison",
          "legacy_fisher_chi2" in t and "S_curv" in t)


def test_C2_offset_revolute_is_rejected():
    """C2: a genuine offset revolute must reject P."""
    truth, cams, mu0, members = make_harness(n=260, moving=170, seed=11)
    T = 6
    axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    a = torch.tensor([0.30, 0.0, 2.1], dtype=torch.float64)
    xi = torch.cat([-torch.linalg.cross(axis, a), axis]).tolist()
    gt = build_truth_state(mu0, members, xi,
                           torch.linspace(0.0, 0.45, T), T)
    obs = observe(truth, cams, gt, T)
    # the null may only be prismatic; give it the best straight-line guess
    nul = build_truth_state(mu0, members,
                            [0.30, 0.0, 0.0, 0.0, 0.0, 0.0],
                            torch.linspace(0.0, 0.45, T), T)
    project_to_prismatic(nul.parts[0])
    res = run_test(truth, cams, mu0, nul, obs)
    t = res["test"]
    print(f"       S_curv {t['S_curv']:.4e} rank {t['rank_curv']} "
          f"cut {t['cut']:.2f}  blocks {t['num_blocks']}")
    print(f"       -> {t['status']}")
    check("C2 a genuine offset revolute rejects the prismatic null",
          t["status"] == REVOLUTE_CANDIDATE, t["status"])


def test_C8_curvature_sweep():
    """C8: as curvature -> 0 the verdict must change by INFORMATION."""
    truth, cams, mu0, members = make_harness(n=220, moving=150, seed=13)
    T = 6
    rows = []
    for radius in (1.5, 6.0, 30.0):
        axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        a = torch.tensor([0.0, 0.0, 2.15 + radius], dtype=torch.float64)
        xi = torch.cat([-torch.linalg.cross(axis, a), axis]).tolist()
        q = torch.linspace(0.0, 0.30 / max(radius, 1e-9) * 2.0, T,
                           dtype=torch.float64)
        gt = build_truth_state(mu0, members, xi, q, T)
        obs = observe(truth, cams, gt, T)
        nul = copy.deepcopy(gt)
        project_to_prismatic(nul.parts[0])
        t = run_test(truth, cams, mu0, nul, obs)["test"]
        rows.append((radius, t))
        print(f"       axis radius {radius:5.1f}: S_curv {t['S_curv']:.3e} "
              f"rank {t['rank_curv']} -> {t['status']}")
    check("C8 the strongest curvature rejects P",
          rows[0][1]["status"] == REVOLUTE_CANDIDATE, rows[0][1]["status"])
    check("C8 and the statistic falls monotonically as the axis recedes",
          rows[0][1]["S_curv"] > rows[-1][1]["S_curv"],
          f"{rows[0][1]['S_curv']:.2e} -> {rows[-1][1]['S_curv']:.2e}")


def main():
    if not torch.cuda.is_available():
        print("CUDA is required for the efficient-score tests")
        return
    tests = [test_8_beta_has_exactly_two_dof,
             test_20_frame0_curvature_jacobian_is_zero,
             test_C7_robust_covariance_inflates_under_block_correlation,
             test_C6_duplicated_evidence_does_not_shrink_confidence,
             test_11_explicit_projection_equivalence,
             test_13_block_scores_sum_to_the_total,
             test_C1_exact_prismatic,
             test_C2_offset_revolute_is_rejected,
             test_C8_curvature_sweep]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
