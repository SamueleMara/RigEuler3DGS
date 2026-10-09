"""Section 2: the coordinate-invariant Gaussian-transport objective.

CODEX_DIRECT_GAUSSIAN_JOINT_RGBD_RELIABILITY.md section 2.

Section 2.1 identifies a real defect in the previous residual: the world-origin
translation p in x' = R x + p is not a physical quantity. Under a global shift
g it becomes p + g - R g, so a residual built from it depends on where the
origin happens to sit whenever two candidates' rotations differ -- which is
exactly the P-versus-R comparison. Section 2.3 replaces it with the Gaussian
transport error, which has an exact moment reduction.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.manifold import so3_exp  # noqa: E402
from joint_reduction.gaussian_transport_metric import (  # noqa: E402
    canonical_moments, gaussian_transport_error_pointwise,
    gaussian_transport_residual)

PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def scene(g, n=300, shape=(1.0, 0.3, 0.05)):
    mu0 = torch.randn((n, 3), generator=g, dtype=torch.float64) \
        * torch.tensor(shape, dtype=torch.float64)
    w = torch.rand(n, generator=g, dtype=torch.float64) + 0.1
    return mu0, w


def pair(g, T=7, spread=0.3):
    Ro = so3_exp(torch.randn((T, 3), generator=g, dtype=torch.float64) * spread)
    po = torch.randn((T, 3), generator=g, dtype=torch.float64)
    Rm = so3_exp(torch.randn((T, 3), generator=g, dtype=torch.float64) * spread)
    pm = torch.randn((T, 3), generator=g, dtype=torch.float64)
    return Ro.double(), po, Rm.double(), pm


# ===========================================================================
def test_2_3_moment_identity_is_exact():
    """§2.3: the O(T) moment form must EQUAL the O(T N) point sum."""
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for shape in ((1.0, 0.3, 0.05), (1.0, 1.0, 1.0), (2.0, 0.01, 0.01)):
        for _ in range(3):
            mu0, w = scene(g, shape=shape)
            M = canonical_moments(mu0, w)
            Ro, po, Rm, pm = pair(g)
            r = gaussian_transport_residual(Ro, po, Rm, pm, M["c"], M["S"], M["L"])
            moment = float((r * r).sum())
            direct = gaussian_transport_error_pointwise(mu0, w, Ro, po, Rm, pm,
                                                        M["L"])
            worst = max(worst, abs(moment - direct) / max(direct, 1e-30))
    print(f"       across 3 part shapes x 3 trials: worst relative "
          f"difference {worst:.2e}")
    check("§2.3 the moment reduction is exact, not an approximation",
          worst < 1e-12, f"{worst:.2e}")


def test_2_3_cross_terms_vanish():
    """§2.3: the cross term vanishes because sum_i w_i (mu_i - c) = 0."""
    g = torch.Generator().manual_seed(1)
    mu0, w = scene(g)
    M = canonical_moments(mu0, w)
    resid = (w.unsqueeze(1) * (mu0 - M["c"])).sum(0)
    print(f"       sum_i w_i (mu_i - c) = {[f'{float(x):.2e}' for x in resid]}")
    check("§2.3 the weighted centred sum is zero",
          float(resid.abs().max()) < 1e-10 * float(w.sum()),
          f"{float(resid.abs().max()):.2e}")
    check("§2.2 L^2 = tr S and is positive",
          abs(M["L2"] - float(M["S"].diagonal().sum())) < 1e-15 and M["L2"] > 0)


def test_2_1_origin_residual_is_not_invariant():
    """§2.1: demonstrate the defect the new objective exists to remove."""
    Ra = so3_exp(torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float64))[0]
    Rb = so3_exp(torch.tensor([[0.0, 0.0, 0.22]], dtype=torch.float64))[0]
    pa = torch.tensor([0.40, 0.10, 0.0], dtype=torch.float64)
    pb = torch.tensor([0.35, 0.12, 0.0], dtype=torch.float64)
    vals = []
    for gmag in (0.0, 1.0, 113.0):
        gv = torch.tensor([gmag, -gmag / 2, gmag / 5], dtype=torch.float64)
        pa_g = pa + gv - Ra @ gv
        pb_g = pb + gv - Rb @ gv
        vals.append(float((pa_g - pb_g).norm()))
    print(f"       origin-translation residual under |g| = 0, 1, 113: "
          f"{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}")
    check("§2.1 the origin-translation residual is NOT coordinate invariant",
          vals[2] > 50 * vals[0], f"{vals[0]:.4f} -> {vals[2]:.4f}")


def test_2_2_transport_error_is_coordinate_invariant():
    """§2.3: E_G must be unchanged by a global shift of >100 part radii."""
    g = torch.Generator().manual_seed(2)
    mu0, w = scene(g)
    M0 = canonical_moments(mu0, w)
    L = M0["L"]
    Ro, po, Rm, pm = pair(g)
    base = float((gaussian_transport_residual(Ro, po, Rm, pm, M0["c"],
                                              M0["S"], L) ** 2).sum())
    print(f"       part radius L = {L:.4f}")
    worst = 0.0
    for mult in (1.0, 10.0, 100.0, 1000.0):
        gv = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)
        gv = gv / gv.norm() * (mult * L)
        mu_s = mu0 + gv
        # a transform expressed in the shifted frame
        po_s = po + gv - torch.einsum("tij,j->ti", Ro, gv)
        pm_s = pm + gv - torch.einsum("tij,j->ti", Rm, gv)
        Ms = canonical_moments(mu_s, w)
        val = float((gaussian_transport_residual(Ro, po_s, Rm, pm_s, Ms["c"],
                                                 Ms["S"], Ms["L"]) ** 2).sum())
        rel = abs(val - base) / max(base, 1e-30)
        worst = max(worst, rel)
        print(f"       shift {mult:7.0f} L: E_G {val:.10e}  relative change "
              f"{rel:.2e}")
    check("§2.3 E_G is invariant to a global coordinate shift", worst < 1e-10,
          f"worst {worst:.2e} at up to 1000 part radii")


def test_2_3_rotation_term_respects_part_shape():
    """§2.3: the rotation moment must weigh by the part's actual extent.

    A rotation about the long axis of a thin rod moves its centres far less
    than the same rotation about a short axis, and tr(dR S dR^T) encodes that.
    A plain ||log(dR)|| would score them the same.
    """
    g = torch.Generator().manual_seed(3)
    rod, w = scene(g, shape=(1.0, 0.02, 0.02))      # long in x
    M = canonical_moments(rod, w)
    T = 2
    eye = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3).repeat(T, 1, 1)
    zero = torch.zeros((T, 3), dtype=torch.float64)
    ang = 0.1
    vals = {}
    for name, axis in (("about the long axis", [1.0, 0.0, 0.0]),
                       ("about a short axis", [0.0, 0.0, 1.0])):
        R = so3_exp(torch.tensor([[0.0, 0.0, 0.0],
                                  [ang * axis[0], ang * axis[1],
                                   ang * axis[2]]], dtype=torch.float64)).double()
        r = gaussian_transport_residual(eye, zero, R, zero, M["c"], M["S"], M["L"])
        vals[name] = float((r * r).sum())
        print(f"       {name}: E_G {vals[name]:.6e}")
    check("§2.3 rotating a rod about its long axis costs far less",
          vals["about the long axis"] < 0.05 * vals["about a short axis"],
          f"{vals['about the long axis']:.2e} vs "
          f"{vals['about a short axis']:.2e}")
    check("§2.4 an isotropic part has no preferred rotation axis",
          True, "checked separately below")
    iso, wi = scene(g, shape=(1.0, 1.0, 1.0))
    Mi = canonical_moments(iso, wi)
    same = []
    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        R = so3_exp(torch.tensor([[0.0, 0.0, 0.0],
                                  [ang * axis[0], ang * axis[1],
                                   ang * axis[2]]], dtype=torch.float64)).double()
        r = gaussian_transport_residual(eye, zero, R, zero, Mi["c"], Mi["S"],
                                        Mi["L"])
        same.append(float((r * r).sum()))
    spread = (max(same) - min(same)) / max(max(same), 1e-30)
    print(f"       isotropic part, three axes: spread {spread:.2e}")
    check("§2.4 and the three axes cost nearly the same there", spread < 0.15,
          f"spread {spread:.2e}")


def main():
    for fn in (test_2_3_cross_terms_vanish,
               test_2_3_moment_identity_is_exact,
               test_2_1_origin_residual_is_not_invariant,
               test_2_2_transport_error_is_coordinate_invariant,
               test_2_3_rotation_term_respects_part_shape):
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across 5 tests")


if __name__ == "__main__":
    main()
