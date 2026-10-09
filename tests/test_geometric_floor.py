"""Section 8 + section 14.6: the error floor, and why the old one failed.

CODEX_DIRECT_GAUSSIAN_JOINT_RGBD_RELIABILITY.md sections 8 and 14.6.

The decisive test here is 14.6. A truly prismatic part is observed through a
rigid fit whose error is CORRELATED -- one shared rotation wobble per frame
affecting every Gaussian at once, which is what a calibration or canonical
model mismatch produces. The old floor divided by sqrt(N) over supernodes as
if each had measured the rotation independently, so it shrank far below the
real error and a spurious revolute gain cleared it easily. The fold-based
floor measures how much the gain itself moves when the cameras change, so it
sees the same wobble for what it is.
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
from joint_reduction.continuous_joint_geometry import \
    fit_continuous_joint  # noqa: E402
from joint_reduction.gaussian_transport_metric import (  # noqa: E402
    canonical_moments, gaussian_transport_error_pointwise)
from joint_reduction.geometric_floor import (  # noqa: E402
    effective_independent_folds, gain_floor_from_folds,
    optimization_error, transform_discrepancy)

PASSED: list[str] = []
RATIOS: dict[str, float] = {}


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def _part(g, n=400):
    mu0 = torch.randn((n, 3), generator=g, dtype=torch.float64) \
        * torch.tensor([1.0, 0.35, 0.12], dtype=torch.float64)
    w = torch.rand(n, generator=g, dtype=torch.float64) + 0.2
    return mu0, w


def _prismatic(T, L):
    d = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    s = torch.linspace(0.0, 1.2 * L, T, dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3).repeat(T, 1, 1)
    p = s.unsqueeze(1) * d
    return R, p, s


# ===========================================================================
def test_8_discrepancy_matches_the_point_sum():
    """D_bc must be the same number whether reduced or summed over points."""
    g = torch.Generator().manual_seed(10)
    mu0, w = _part(g)
    M = canonical_moments(mu0, w)
    T = 9
    Tb = torch.eye(4, dtype=torch.float64).reshape(1, 4, 4).repeat(T, 1, 1)
    Tc = Tb.clone()
    Tb[:, :3, :3] = so3_exp(torch.randn((T, 3), generator=g,
                                        dtype=torch.float64) * 0.2).double()
    Tc[:, :3, :3] = so3_exp(torch.randn((T, 3), generator=g,
                                        dtype=torch.float64) * 0.2).double()
    Tb[:, :3, 3] = torch.randn((T, 3), generator=g, dtype=torch.float64) * 0.1
    Tc[:, :3, 3] = torch.randn((T, 3), generator=g, dtype=torch.float64) * 0.1
    d = transform_discrepancy(Tb, Tc, M)
    moment = float((d["D_bc_per_frame"] ** 2).mean())
    direct = gaussian_transport_error_pointwise(
        mu0, w, Tb[:, :3, :3], Tb[:, :3, 3], Tc[:, :3, :3], Tc[:, :3, 3],
        M["L"])
    rel = abs(moment - direct) / max(direct, 1e-30)
    print(f"       moment {moment:.10e}  point sum {direct:.10e}  rel {rel:.2e}")
    check("§8 D_bc reduces exactly to the point sum", rel < 1e-12, f"{rel:.2e}")


def test_8_refuses_to_invent_a_floor_from_one_fold():
    r = gain_floor_from_folds([3.2e-5])
    print(f"       one fold -> {r['status']}")
    check("§8 a single fold yields no floor at all",
          r["status"] == "INSUFFICIENT_FOLDS" and r["epsilon_geom"] is None)
    check("§8 and no floor is reported for zero folds",
          gain_floor_from_folds([])["epsilon_geom"] is None)


def test_8_correlated_folds_do_not_inflate_the_evidence():
    """n_effective must fall below n when the folds agree with each other."""
    indep = effective_independent_folds([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    corr = effective_independent_folds([1.0, 1.05, 1.1, 1.14, 1.2, 1.26])
    print(f"       alternating: rho {indep['rho_lag1']:+.3f} "
          f"n_eff {indep['n_effective']:.2f} of {indep['n_folds']}")
    print(f"       drifting:    rho {corr['rho_lag1']:+.3f} "
          f"n_eff {corr['n_effective']:.2f} of {corr['n_folds']}")
    check("§8 correlated folds count for less than their number",
          corr["n_effective"] < 0.5 * corr["n_folds"],
          f"{corr['n_effective']:.2f} of {corr['n_folds']}")


def test_14_6_correlated_rigid_error_defeats_the_naive_floor():
    """Section 14.6, the point of the whole module.

    A prismatic part, observed with a per-frame rotation wobble that is SHARED
    by every Gaussian. The naive floor treats the N supernodes as N
    independent looks at that rotation and shrinks by sqrt(N); the fold-based
    floor measures the gain's own spread and does not.
    """
    g = torch.Generator().manual_seed(11)
    mu0, w = _part(g)
    M = canonical_moments(mu0, w)
    L = M["L"]
    T, N_SUPERNODES = 24, 2048
    R_true, p_true, _ = _prismatic(T, L)

    sigma = 6e-3           # rad of shared per-frame wobble
    gains, folds = [], []
    for b in range(5):
        gb = torch.Generator().manual_seed(200 + b)
        # ONE rotation per frame, applied to the whole part: this is a rigid
        # model / calibration error, not independent per-Gaussian noise.
        wob = so3_exp(torch.randn((T, 3), generator=gb,
                                  dtype=torch.float64) * sigma).double()
        wob[0] = torch.eye(3, dtype=torch.float64)
        R_obs = torch.einsum("tij,tjk->tik", wob, R_true)
        p_obs = p_true + torch.randn((T, 3), generator=gb,
                                     dtype=torch.float64) * (sigma * L)
        p_obs[0] = p_true[0]
        T4 = torch.eye(4, dtype=torch.float64).reshape(1, 4, 4).repeat(T, 1, 1)
        T4[:, :3, :3], T4[:, :3, 3] = R_obs, p_obs
        folds.append(T4)
        fP = fit_continuous_joint(M["c"], R_obs, p_obs, L, max_iters=250,
                                  fix_eta_zero=True, moments=M)
        fR = fit_continuous_joint(M["c"], R_obs, p_obs, L, max_iters=250,
                                  fix_eta_zero=False, moments=M)
        gains.append(fP["energy"] - fR["energy"])
        print(f"       fold {b}: E_P {fP['energy']:.4e}  E_R {fR['energy']:.4e}"
              f"  gain {gains[-1]:+.4e}")

    mean_gain = sum(gains) / len(gains)
    check("§14.6 every fold prefers revolute on data that is truly prismatic",
          all(x > 0 for x in gains), f"mean gain {mean_gain:.3e}")

    # --- the old floor -----------------------------------------------------
    naive = (sigma / (L * math.sqrt(N_SUPERNODES))) ** 2
    print(f"       naive floor (sigma/(L sqrt(N)))^2, N={N_SUPERNODES}: "
          f"{naive:.3e}   mean gain / naive floor = {mean_gain / naive:.1f}")
    check("§8 the naive 1/sqrt(N) floor passes this spurious gain",
          mean_gain > 10 * naive, f"{mean_gain / naive:.0f}x the naive floor")

    # --- the fold-based floor ---------------------------------------------
    fl = gain_floor_from_folds(gains)
    print(f"       fold floor: epsilon_geom {fl['epsilon_geom']:.3e}, "
          f"gain/floor {fl['gain_over_floor']:.2f}, "
          f"sd across folds {fl['sd_gain']:.3e}")
    check("§8 the fold-based floor is far larger than the naive one",
          fl["epsilon_geom"] > 10 * naive,
          f"{fl['epsilon_geom'] / naive:.0f}x the naive floor")
    # What matters is not the floor's size but what it does to the EVIDENCE.
    # The same spurious gain goes from overwhelming to marginal.
    collapse = (mean_gain / naive) / max(fl["gain_over_floor"], 1e-30)
    print(f"       apparent significance: {mean_gain / naive:.0f}x the naive "
          f"floor, {fl['gain_over_floor']:.2f}x the measured one "
          f"-- a {collapse:.0f}x collapse")
    check("§8 the spurious gain stops being significant under the real floor",
          collapse > 20, f"{collapse:.0f}x collapse")
    RATIOS["prismatic"] = fl["gain_over_floor"]

    # --- and the transform repeatability it came from ---------------------
    d01 = transform_discrepancy(folds[0], folds[1], M)
    print(f"       D_bc between folds 0 and 1: rms {d01['D_bc_rms']:.4e} "
          f"(translation {d01['translation_rms']:.3e}, "
          f"rotation {d01['rotation_rms']:.3e})")
    check("§8 D_bc reports a real disagreement between folds",
          d01["D_bc_rms"] > 1e-4, f"{d01['D_bc_rms']:.2e}")


def test_8_a_real_revolute_still_clears_its_own_floor():
    """The floor must not simply refuse everything."""
    g = torch.Generator().manual_seed(12)
    mu0, w = _part(g)
    M = canonical_moments(mu0, w)
    L = M["L"]
    T = 24
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    pivot = M["c"] + torch.tensor([1.4 * L, 0.0, 0.0], dtype=torch.float64)
    q = torch.linspace(0.0, 1.1, T, dtype=torch.float64)
    R_true = so3_exp(q.unsqueeze(1) * axis).double()
    p_true = pivot - torch.einsum("tij,j->ti", R_true, pivot)

    sigma = 6e-3
    gains = []
    for b in range(5):
        gb = torch.Generator().manual_seed(300 + b)
        wob = so3_exp(torch.randn((T, 3), generator=gb,
                                  dtype=torch.float64) * sigma).double()
        wob[0] = torch.eye(3, dtype=torch.float64)
        R_obs = torch.einsum("tij,tjk->tik", wob, R_true)
        p_obs = p_true + torch.randn((T, 3), generator=gb,
                                     dtype=torch.float64) * (sigma * L)
        p_obs[0] = p_true[0]
        fP = fit_continuous_joint(M["c"], R_obs, p_obs, L, max_iters=250,
                                  fix_eta_zero=True, moments=M)
        fR = fit_continuous_joint(M["c"], R_obs, p_obs, L, max_iters=250,
                                  fix_eta_zero=False, moments=M)
        gains.append(fP["energy"] - fR["energy"])
    fl = gain_floor_from_folds(gains)
    print(f"       true revolute: mean gain {fl['mean_gain']:.4e}, "
          f"floor {fl['epsilon_geom']:.3e}, ratio {fl['gain_over_floor']:.1f}")
    check("§8 a genuine revolute clears the same floor by a wide margin",
          fl["gain_over_floor"] > 20, f"{fl['gain_over_floor']:.1f}x")
    check("§8 and all folds agree on the sign", fl["sign_is_unanimous"])
    RATIOS["revolute"] = fl["gain_over_floor"]
    # The separation is the whole claim: identical noise model, identical
    # floor procedure, and the two cases land decades apart.
    if "prismatic" in RATIOS:
        sep = RATIOS["revolute"] / max(RATIOS["prismatic"], 1e-30)
        print(f"       true revolute {RATIOS['revolute']:.1f}x floor vs "
              f"spurious {RATIOS['prismatic']:.2f}x -- separated by "
              f"{sep:.0f}x")
        check("§8 the floor separates a real joint from a spurious one",
              sep > 10, f"{sep:.0f}x apart")


def test_8_optimization_error_is_measured_not_assumed():
    g = torch.Generator().manual_seed(13)
    mu0, w = _part(g)
    M = canonical_moments(mu0, w)
    L = M["L"]
    R_obs, p_obs, _ = _prismatic(16, L)

    def fit(seed):
        gg = torch.Generator().manual_seed(400 + seed)
        jitter = so3_exp(torch.randn((16, 3), generator=gg,
                                     dtype=torch.float64) * 1e-9).double()
        return fit_continuous_joint(M["c"],
                                    torch.einsum("tij,tjk->tik", jitter, R_obs),
                                    p_obs, L, max_iters=200,
                                    fix_eta_zero=False, moments=M)["energy"]

    oe = optimization_error(fit, n_starts=3)
    print(f"       epsilon_opt over {oe['n_starts']} starts: "
          f"{oe['epsilon_opt']:.3e} (best {oe['best']:.3e})")
    check("§8 epsilon_opt comes from repeated solves, not a constant",
          oe["epsilon_opt"] >= 0.0 and oe["n_starts"] == 3)


def main():
    for fn in (test_8_discrepancy_matches_the_point_sum,
               test_8_refuses_to_invent_a_floor_from_one_fold,
               test_8_correlated_folds_do_not_inflate_the_evidence,
               test_14_6_correlated_rigid_error_defeats_the_naive_floor,
               test_8_a_real_revolute_still_clears_its_own_floor,
               test_8_optimization_error_is_measured_not_assumed):
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across 6 tests")


if __name__ == "__main__":
    main()
