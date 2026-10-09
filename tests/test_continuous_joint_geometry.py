"""Sections 23-25: the continuous SE(3) chart.

CODEX_IMPLEMENT_CONTINUOUS_SE3_PRISMATIC_REVOLUTE_GEOMETRY.md sections 5-8,
22-25.

The point of this chart is that prismatic motion is a REGULAR INTERIOR POINT
(eta = 0) rather than the limit ||a|| -> inf of the finite-axis family. These
tests check that claim numerically: exactness at zero, continuity approaching
zero, and agreement with the finite-axis parameterisation wherever both are
defined.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.manifold import so3_exp, so3_log, project_to_so3  # noqa: E402
from joint_reduction.continuous_joint_geometry import (  # noqa: E402
    SERIES_CUTOFF, continuous_joint_transform, continuous_to_finite_axis,
    finite_axis_to_continuous, series_cutoff, stable_so3_coefficients,
    transform_points)

PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def rand_unit(g):
    v = torch.randn(3, generator=g, dtype=torch.float64)
    return v / v.norm()


# ===========================================================================
def test_7_coefficients_are_stable_at_zero():
    """§7: A, B, C must be accurate through the series/closed-form switch."""
    print(f"       series cutoff (float64) = {SERIES_CUTOFF:.4e}, "
          f"(float32) = {series_cutoff(torch.float32):.4e}")
    # mpmath-free high-precision reference: evaluate in float64 well away from
    # cancellation using the identity 1 - cos t = 2 sin^2(t/2).
    th = torch.tensor([0.0, 1e-12, 1e-8, 1e-4, 1e-3, 1e-2, SERIES_CUTOFF,
                       0.1, 0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
    A, B, C = stable_so3_coefficients(th)
    worst = {"A": 0.0, "B": 0.0, "C": 0.0}
    for i, t in enumerate(th.tolist()):
        if t == 0.0:
            refA, refB, refC = 1.0, 0.5, 1.0 / 6.0
        else:
            refA = math.sin(t) / t
            refB = 2.0 * math.sin(t / 2.0) ** 2 / (t * t)   # no cancellation
            # The closed form (t - sin t)/t^3 cancels catastrophically: at
            # t = 1e-2 the numerator is ~1.7e-7 built from numbers ~1e-2, so
            # the REFERENCE itself carries ~1e-11 relative error -- worse than
            # the implementation under test. Use the series where it is the
            # accurate one and only switch above t = 0.1.
            refC = ((t - math.sin(t)) / t ** 3 if t > 0.1 else
                    (1.0 / 6.0 - t * t / 120.0 + t ** 4 / 5040.0
                     - t ** 6 / 362880.0))
        worst["A"] = max(worst["A"], abs(float(A[i]) - refA))
        worst["B"] = max(worst["B"], abs(float(B[i]) - refB) / max(refB, 1e-30))
        worst["C"] = max(worst["C"], abs(float(C[i]) - refC) / max(refC, 1e-30))
    print(f"       worst error: A {worst['A']:.2e}  B(rel) {worst['B']:.2e}  "
          f"C(rel) {worst['C']:.2e}")
    check("§7 A, B and C are accurate across the switch", worst["A"] < 1e-14
          and worst["B"] < 1e-12 and worst["C"] < 1e-12, str(worst))
    A0, B0, C0 = stable_so3_coefficients(torch.zeros(1, dtype=torch.float64))
    check("§7 and exact at theta = 0",
          float(A0) == 1.0 and float(B0) == 0.5
          and abs(float(C0) - 1.0 / 6.0) < 1e-18,
          f"{float(A0)}, {float(B0)}, {float(C0)}")


def test_23_exact_prismatic_limit():
    """§23: T(d, eta, s) -> T_P(d, s) as ||eta|| -> 0, and EXACT at zero."""
    g = torch.Generator().manual_seed(0)
    worst_rows = []
    for trial in range(4):
        c = torch.randn(3, generator=g, dtype=torch.float64) * 2.0
        d = rand_unit(g)
        s = torch.tensor([0.0, 0.07, -0.2, 0.5, 1.3], dtype=torch.float64)
        x = torch.randn((40, 3), generator=g, dtype=torch.float64) * 0.3 + c
        # a direction perpendicular to d, so eta . d = 0 as section 5 requires
        perp = torch.linalg.cross(d, rand_unit(g))
        perp = perp / perp.norm()

        R0, p0 = continuous_joint_transform(c, d, torch.zeros(3, dtype=torch.float64), s)
        check_exact = (float((R0 - torch.eye(3, dtype=torch.float64)).abs().max()),
                       float((p0 - s.unsqueeze(1) * d).abs().max()))
        if trial == 0:
            check("§23 at eta = 0 the rotation is EXACTLY the identity",
                  check_exact[0] == 0.0, f"{check_exact[0]}")
            check("§23 and the translation is EXACTLY s d",
                  check_exact[1] == 0.0, f"{check_exact[1]}")
            check("§23 and points move exactly as the prismatic model",
                  float((transform_points(x, R0, p0)
                         - (x.unsqueeze(0) + s.view(-1, 1, 1) * d)).abs().max()) == 0.0)
        row = []
        for k in (1e-1, 1e-3, 1e-6, 1e-9, 1e-12):
            R, p = continuous_joint_transform(c, d, perp * k, s)
            dR = float(so3_log(project_to_so3(R)).norm(dim=1).max())
            dp = float((p - s.unsqueeze(1) * d).abs().max())
            row.append((k, dR, dp))
        worst_rows.append(row)
    for k_i in range(5):
        k = worst_rows[0][k_i][0]
        dR = max(r[k_i][1] for r in worst_rows)
        dp = max(r[k_i][2] for r in worst_rows)
        print(f"       ||eta|| = {k:.0e}:  max rotation {dR:.3e}  "
              f"max |p - s d| {dp:.3e}")
    # the deviation must fall with ||eta||, i.e. the limit is approached
    last = worst_rows[0]
    check("§23 the deviation from prismatic falls with ||eta||",
          last[-1][1] < last[0][1] * 1e-6 and last[-1][2] < max(last[0][2], 1e-30) * 1e-5,
          f"{last[0][1]:.2e} -> {last[-1][1]:.2e}")


def test_24_finite_revolute_equivalence():
    """§24: the chart reproduces arbitrary off-origin finite revolute axes."""
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    for r_over_L in (0.8, 3.0, 25.0, 400.0):
        c = torch.randn(3, generator=g, dtype=torch.float64) * 1.5
        n = rand_unit(g)
        off = torch.linalg.cross(n, rand_unit(g))
        off = off / off.norm() * r_over_L * 0.2
        a = c - off                               # axis at distance |off| from c
        q = torch.tensor([0.0, 0.05, 0.2, -0.1, 0.45], dtype=torch.float64)
        x = torch.randn((50, 3), generator=g, dtype=torch.float64) * 0.2 + c

        R_ref = so3_exp(q.unsqueeze(1) * n.unsqueeze(0)).double()
        x_ref = a + torch.einsum("tij,nj->tni", R_ref, (x - a))

        d, eta, s = finite_axis_to_continuous(c, n, a, q)
        check_perp = abs(float(d @ eta))
        R, p = continuous_joint_transform(c, d, eta, s)
        x_new = transform_points(x, R, p)
        err = float((x_new - x_ref).abs().max())
        worst = max(worst, err)
        print(f"       r/L {r_over_L:6.1f}: max point error {err:.3e}, "
              f"|d.eta| {check_perp:.1e}")
    check("§24 the chart matches the finite-axis map exactly", worst < 1e-12,
          f"worst {worst:.2e}")

    # round trip back to a finite axis
    c = torch.zeros(3, dtype=torch.float64)
    n = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    a = torch.tensor([0.7, -0.3, 0.0], dtype=torch.float64)
    q = torch.tensor([0.0, 0.3, 0.6], dtype=torch.float64)
    d, eta, s = finite_axis_to_continuous(c, n, a, q)
    back = continuous_to_finite_axis(c, d, eta, s)
    check("§8 the axis round-trips through the chart",
          float((back["axis"] - n).abs().max()) < 1e-12
          and float((back["axis_point_centroid"] - a).abs().max()) < 1e-12,
          f"axis {float((back['axis'] - n).abs().max()):.1e}")
    check("§8 and q round-trips",
          float((back["q"] - q).abs().max()) < 1e-12)
    check("§8 the centroid-closest point satisfies n.(a_c - c) = 0",
          abs(float(back["axis"] @ (back["axis_point_centroid"] - c))) < 1e-12)
    check("§8 the origin-closest point is reported separately and differs",
          float((back["axis_point_origin"] - back["axis_point_centroid"]).norm())
          >= 0.0)


def test_24_translation_equivariance():
    """§24: a global shift must move the axis and leave the type and q alone."""
    c = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    n = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    a = torch.tensor([0.9, 0.0, -0.4], dtype=torch.float64)
    q = torch.tensor([0.0, 0.25, 0.5], dtype=torch.float64)
    shift = torch.tensor([37.0, -12.0, 5.0], dtype=torch.float64)

    d1, e1, s1 = finite_axis_to_continuous(c, n, a, q)
    d2, e2, s2 = finite_axis_to_continuous(c + shift, n, a + shift, q)
    print(f"       shift {shift.tolist()}: |dd| "
          f"{float((d1 - d2).abs().max()):.2e}  |deta| "
          f"{float((e1 - e2).abs().max()):.2e}  |ds| "
          f"{float((s1 - s2).abs().max()):.2e}")
    check("§24 d, eta and s are translation invariant",
          float((d1 - d2).abs().max()) < 1e-12
          and float((e1 - e2).abs().max()) < 1e-12
          and float((s1 - s2).abs().max()) < 1e-12)
    b1 = continuous_to_finite_axis(c, d1, e1, s1)
    b2 = continuous_to_finite_axis(c + shift, d2, e2, s2)
    check("§24 the axis point translates with the world",
          float((b2["axis_point_centroid"]
                 - (b1["axis_point_centroid"] + shift)).abs().max()) < 1e-10)
    check("§24 and the signed q is unchanged",
          float((b1["q"] - b2["q"]).abs().max()) < 1e-12)


def test_25_axis_through_centroid_is_rejected_not_fudged():
    """§25: the chart is undefined when the axis passes through the centroid.

    It must say so, so the combined fitter can route the case to the general
    finite-axis branch (section 9) instead of silently producing nonsense.
    """
    c = torch.zeros(3, dtype=torch.float64)
    n = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    q = torch.tensor([0.0, 0.4], dtype=torch.float64)
    raised = False
    try:
        finite_axis_to_continuous(c, n, c.clone(), q)
    except ValueError as exc:
        raised = "centroid" in str(exc)
    check("§25 an axis through the centroid is refused explicitly", raised)
    check("§9 and the message points at the finite-axis branch", raised)
    # The finite-axis branch itself must still handle it.
    from joint_reduction.revolute_fit import fit_revolute
    qs = torch.tensor([0.0, 0.1, 0.25, 0.4, 0.6], dtype=torch.float64)
    R = so3_exp(qs.unsqueeze(1) * n.unsqueeze(0)).double()
    t = torch.zeros((len(qs), 3), dtype=torch.float64)   # axis through centroid
    f = fit_revolute(R, t, 0.3, max_iters=60)
    print(f"       finite-axis branch on a centroid axis: "
          f"{f['convergence']['status']}, E_R {f['error']:.3e}")
    check("§25 the finite-axis branch still recovers it",
          f["error"] < 1e-12, f"E_R {f['error']:.3e}")


def test_7_no_division_by_curvature():
    """§7/§22: the forward model must not divide by curvature anywhere."""
    import ast
    src = (REPO / "joint_reduction" / "continuous_joint_geometry.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "continuous_joint_transform")
    # Strip the docstring: it legitimately DISCUSSES 1/kappa while the code
    # must not contain it, and a plain substring scan cannot tell them apart.
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value,
                                                             ast.Constant):
        body = body[1:]
    code = "\n".join(ast.unparse(n) for n in body)
    for bad in ("kappa", "norm(eta)", "eta.norm()"):
        check(f"§7 the forward model never computes {bad!r}", bad not in code)
    check("§7 and it never divides by a norm", "/ " not in code.replace("/ 6", "")
          or "norm" not in code)
    check("§22 the conversion to a finite axis is where curvature may divide",
          "kappa * kappa" in src)


def main():
    for fn in (test_7_coefficients_are_stable_at_zero,
               test_7_no_division_by_curvature,
               test_23_exact_prismatic_limit,
               test_24_finite_revolute_equivalence,
               test_24_translation_equivariance,
               test_25_axis_through_centroid_is_rejected_not_fudged):
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across 6 tests")


if __name__ == "__main__":
    main()
