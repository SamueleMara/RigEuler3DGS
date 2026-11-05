#!/usr/bin/env python
"""Section 5 of RigEuler3DGS_Codex_Surgical_Fixes_4c88725.md: MEASURE the surface fit of
estimator/surface_bootstrap.fit_surface on synthetic geometry with known truth.

Scenes (rank-1 sensor noise along each camera ray, sd per camera):
  planes     a tilted plane seen by 3 cameras at different ray angles (0.5 / 2 / 6 mm)
  edge       a 5 cm step; seeds 6 mm from the edge on the lower side
  panel      a 4 mm thin panel; front face seen from +z, back face from -z
  occluder   a plane partly hidden by a smaller plane 3 cm in front of it
Reported: centre-to-true-surface error (median / p90 / max), normal angle error,
layer merging (panel), and the calibration of the fit's own normal variance
(Monte-Carlo sd of the fitted centre vs the predicted sqrt(var_normal)).

The tolerances below are TEST tolerances, stated with each check; they do not gate the
pipeline. P0 (renderer Fisher) against actual estimation error is NOT evaluated here.

    python tests/test_bootstrap_geometry_eval.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CHECKS, FAILURES, REPORT = 0, [], {}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def observe(points, cams, sds, rng):
    """Noisy samples of `points` (per camera: [M,3]) along each camera's rays."""
    Y, D, V = [], [], []
    for P, c, s in zip(points, cams, sds):
        ray = P - c
        rng_ = np.linalg.norm(ray, axis=1)
        u = ray / rng_[:, None]
        Y.append(P + u * rng.normal(0, s, len(P))[:, None])
        D.append(u)
        V.append(np.full(len(P), s * s))
    return np.concatenate(Y), np.concatenate(D), np.concatenate(V)


def fit(Y, D, V, seeds, k=24):
    from estimator.surface_bootstrap import fit_surface
    return fit_surface(Y, D, V, np.full((len(Y), 3), 0.5), seeds, k=k, device=DEV)


def angle_deg(a, b):
    c = np.abs((a * b).sum(1)) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def stats(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)), "max": float(x.max())}


def plane_pts(n, R, rng, extent=0.1, offset=0.0):
    xy = rng.uniform(-extent, extent, (n, 2))
    return np.c_[xy, np.full(n, offset)] @ R.T


def main() -> int:
    rng = np.random.default_rng(0)
    # ---- tilted plane, three ray angles ------------------------------------
    th = math.radians(25)
    R = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
    n_true = R @ np.array([0, 0, 1.0])
    cams = [np.array([0, 0, 1.0]), np.array([0.7, 0, 0.7]), np.array([-0.5, 0.6, 0.6])]
    pts = [plane_pts(2000, R, rng) for _ in cams]
    Y, D, V = observe(pts, cams, (5e-4, 2e-3, 6e-3), rng)
    seeds = plane_pts(300, R, rng, extent=0.08)
    f = fit(Y, D, V, seeds)
    err = np.abs((f["mu"] @ n_true))
    ang = angle_deg(f["normal"], n_true[None])
    REPORT["planes"] = {"centre_error_mm": stats(1000 * err), "normal_error_deg": stats(ang)}
    print(f"  planes: centre error mm {REPORT['planes']['centre_error_mm']}, normal deg {REPORT['planes']['normal_error_deg']}")
    check("planes: median centre error < 0.5 mm (test tolerance: ~sensor sd of the best camera)",
          REPORT["planes"]["centre_error_mm"]["median"] < 0.5)
    check("planes: median normal error < 2 deg (test tolerance)", REPORT["planes"]["normal_error_deg"]["median"] < 2.0)
    # ---- calibration of var_normal over Monte-Carlo repetitions --------------
    mus = []
    for rep in range(30):
        r2 = np.random.default_rng(100 + rep)
        Y2, D2, V2 = observe(pts, cams, (5e-4, 2e-3, 6e-3), r2)
        mus.append(fit(Y2, D2, V2, seeds[:60])["mu"] @ n_true)
    emp = np.std(np.stack(mus), axis=0)
    pred = np.sqrt(f["var_normal"][:60])
    ratio = emp / pred
    REPORT["var_normal_calibration"] = {"empirical_over_predicted_sd": stats(ratio),
                                        "note": "seed-fit normal variance 1/sum(1/var_j); the fit-level quantity, not P0"}
    print(f"  var_normal: empirical / predicted sd {REPORT['var_normal_calibration']['empirical_over_predicted_sd']}")
    check("var_normal is not overconfident by more than 2x (test tolerance, median ratio)", float(np.median(ratio)) < 2.0,
          f"median ratio {float(np.median(ratio)):.2f}")
    # ---- sharp step ----------------------------------------------------------
    pts_e = []
    for c in cams:
        xy = rng.uniform(-0.1, 0.1, (3000, 2))
        pts_e.append(np.c_[xy, np.where(xy[:, 0] > 0, 0.05, 0.0)])
    Ye, De, Ve = observe(pts_e, cams, (5e-4, 5e-4, 5e-4), rng)
    se = np.c_[np.full(60, -0.006), np.linspace(-0.08, 0.08, 60), np.zeros(60)]
    fe = fit(Ye, De, Ve, se)
    REPORT["edge"] = {"centre_error_mm": stats(1000 * np.abs(fe["mu"][:, 2])),
                      "normal_error_deg": stats(angle_deg(fe["normal"], np.array([[0, 0, 1.0]])))}
    print(f"  edge: {REPORT['edge']}")
    check("edge: the step is not averaged in (p90 centre error < 2 mm, test tolerance)",
          REPORT["edge"]["centre_error_mm"]["p90"] < 2.0)
    # ---- thin panel: front face seen from +z, back face from -z --------------
    front_c = [np.array([0, 0, 1.0]), np.array([0.5, 0, 0.8])]
    back_c = [np.array([0, 0, -1.0]), np.array([-0.5, 0, -0.8])]
    gap = 0.004
    ptsf = [plane_pts(2500, np.eye(3), rng, offset=0.0) for _ in front_c]
    ptsb = [plane_pts(2500, np.eye(3), rng, offset=-gap) for _ in back_c]
    Yf, Df, Vf = observe(ptsf, front_c, (5e-4, 5e-4), rng)
    Yb, Db, Vb = observe(ptsb, back_c, (5e-4, 5e-4), rng)
    Yp, Dp, Vp = np.r_[Yf, Yb], np.r_[Df, Db], np.r_[Vf, Vb]
    sf = plane_pts(100, np.eye(3), rng, extent=0.08, offset=0.0)
    sb = plane_pts(100, np.eye(3), rng, extent=0.08, offset=-gap)
    for k in (24, 48):
        fp = fit(Yp, Dp, Vp, np.r_[sf, sb], k=k)
        z = fp["mu"][:, 2]
        ef, eb = np.abs(z[:100]), np.abs(z[100:] + gap)
        merged = float(np.mean(np.r_[ef, eb] > gap / 4))
        REPORT[f"panel_k{k}"] = {"front_error_mm": stats(1000 * ef), "back_error_mm": stats(1000 * eb),
                                 "fraction_merged(>1mm off own face)": merged}
        print(f"  panel k={k}: front {REPORT[f'panel_k{k}']['front_error_mm']}, back "
              f"{REPORT[f'panel_k{k}']['back_error_mm']}, merged fraction {merged:.2f}")
    check("panel (k=24): fewer than 10 % of seeds pulled > 1 mm toward the other face (test tolerance)",
          REPORT["panel_k24"]["fraction_merged(>1mm off own face)"] < 0.1)
    # ---- occluder ------------------------------------------------------------
    co = [np.array([0, 0, 1.0]), np.array([0.6, 0.2, 0.8]), np.array([-0.6, -0.2, 0.8])]
    pts_o = []
    for c in co:
        bgp = plane_pts(4000, np.eye(3), rng, extent=0.1)
        occ = plane_pts(800, np.eye(3), rng, extent=0.03, offset=0.03)
        # a background sample is visible only if the ray to it misses the occluder square
        t = (0.03 - c[2]) / (bgp[:, 2] - c[2])
        hit = c[None] + t[:, None] * (bgp - c[None])
        vis = ~((np.abs(hit[:, 0]) < 0.03) & (np.abs(hit[:, 1]) < 0.03))
        pts_o.append(np.r_[bgp[vis], occ])
    Yo, Do, Vo = observe(pts_o, co, (1e-3, 1e-3, 1e-3), rng)
    so = np.c_[np.linspace(-0.045, 0.045, 40), np.full(40, 0.035), np.zeros(40)]       # background, near the occluder
    fo = fit(Yo, Do, Vo, so)
    REPORT["occluder"] = {"background_centre_error_mm": stats(1000 * np.abs(fo["mu"][:, 2]))}
    print(f"  occluder: {REPORT['occluder']}")
    check("occluder: background seeds near the occluder stay on the background (p90 < 2 mm, test tolerance)",
          REPORT["occluder"]["background_centre_error_mm"]["p90"] < 2.0)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.write_text(json.dumps(REPORT, indent=1))
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
