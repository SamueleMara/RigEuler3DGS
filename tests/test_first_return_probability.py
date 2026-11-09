#!/usr/bin/env python
"""Gate A: the first-return CPU oracle (estimator/first_return_reference.py), CPU only.

  [A1] the delivered reference tests (verbatim behaviour)
  [A2] normalization |sum q + int f_U + q_empty - 1| < 1e-10 on random rays: 0..12
       contributors, lambda 0 / >0, dense high-alpha rays, per-camera near/far
  [A3] the closed-form unknown interval integral vs high-precision quadrature (mpmath),
       including far tails
  [A4] autodiff vs central finite differences of -log L in every z_i and alpha_i,
       and of a multi-ray sum
  [A5] prior schema: valid round trip; missing camera / bad units / negative hazard /
       piecewise hazard / edited content / impossible support all refused

    python tests/test_first_return_probability.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CHECKS, FAILURES = 0, []


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def run(D, z, alpha, lam, s=0.002, zn=0.3, zf=1.3):
    from estimator.first_return_reference import first_return_ray
    z = torch.tensor(z, dtype=torch.float64)
    a = torch.tensor(alpha, dtype=torch.float64)
    return first_return_ray(torch.tensor(D, dtype=torch.float64), z, a, torch.full_like(z, s),
                            z_near=zn, z_far=zf, lambda_unknown=lam, sigma_unknown=s)


def mass(r):
    return float(r.gaussian_first_hit_mass.sum() + r.unknown_first_hit_mass + r.no_return_mass)


def test_reference():
    print("\n[A1] delivered reference tests")
    for z, a, lam in [([], [], 0.4), ([.7], [.6], 0.0), ([.7], [.6], 2.0), ([.7, .9], [.3, .9], 0.8),
                      ([.9, .7], [.9, .3], 4.0)]:
        r = run(.85, z, a, lam)
        ok = abs(mass(r) - 1) < 1e-11 and math.isfinite(float(r.log_valid_depth_likelihood)) and \
            abs(float(r.gamma_gaussian.sum() + r.gamma_unknown) - 1) < 1e-10
        check(f"[A1] normalization + finite + posteriors sum to 1  (z={z}, lambda={lam})", ok)
    r = run(.75, [.75, .95], [.9, .9], .25, s=.004)
    check("[A1] hidden cavity: door gamma > 0.99, cavity gamma < 1e-7",
          float(r.gamma_gaussian[0]) > .99 and float(r.gamma_gaussian[1]) < 1e-7)
    ref, wrong = run(.95, [], [], .8, s=.003), run(.95, [.75], [.9], .8, s=.003)
    ratio = float((wrong.log_valid_depth_likelihood - ref.log_valid_depth_likelihood).exp())
    check("[A1] a false alpha=0.9 front surface makes the deeper return ~0.1x as likely", abs(ratio - .1) < .002,
          f"ratio {ratio:.4f}")
    r = run(.7, [.7], [.8], 0, s=.005)
    check("[A1] lambda = 0: q = alpha, q_empty = 1 - alpha, gamma = 1",
          abs(float(r.gaussian_first_hit_mass[0]) - .8) < 1e-12 and abs(float(r.no_return_mass) - .2) < 1e-12
          and abs(float(r.gamma_gaussian[0]) - 1) < 1e-12)
    r = run(.95, [], [], 1.0, s=.003)
    check("[A1] no modelled geometry: unknown explains everything",
          float(r.gamma_unknown) == 1 and float(r.unknown_first_hit_mass) > 0 and float(r.no_return_mass) > 0)
    r = run(.75, [.75, .75, .95], [.2, .3, .7], .4)
    check("[A1] coincident depths: finite, normalized", math.isfinite(float(r.log_valid_depth_likelihood))
          and abs(mass(r) - 1) < 1e-11)
    a_, b_ = run(.751, [.95, .75], [.5, .3], 1.7), run(.751, [.75, .95], [.3, .5], 1.7)
    check("[A1] invariant to the input order of distinct contributors",
          abs(float(a_.log_valid_depth_likelihood - b_.log_valid_depth_likelihood)) < 1e-12
          and torch.allclose(a_.gamma_gaussian, b_.gamma_gaussian.flip(0)))


def test_normalization():
    print("\n[A2] normalization on random rays")
    from estimator.first_return_reference import first_return_ray
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    for zn, zf in ((0.3, 1.3), (0.2, 2.5), (0.05, 6.0)):
        for lam in (0.0, 0.3, 4.0):
            for k in range(0, 13):
                z = zn + (zf - zn) * (0.02 + 0.96 * torch.rand(k, generator=g, dtype=torch.float64))
                a = 0.01 + 0.98 * torch.rand(k, generator=g, dtype=torch.float64)
                r = first_return_ray(torch.tensor(0.5 * (zn + zf), dtype=torch.float64), z, a,
                                     torch.full_like(z, 0.01), z_near=zn, z_far=zf, lambda_unknown=lam,
                                     sigma_unknown=0.02)
                worst = max(worst, abs(mass(r) - 1))
    check("[A2] |sum q + int f_U + q_empty - 1| < 1e-10 (3 supports x 3 hazards x 0..12 contributors)",
          worst < 1e-10, f"worst {worst:.2e}")
    z = torch.linspace(0.5, 0.9, 40, dtype=torch.float64)
    r = first_return_ray(torch.tensor(0.95, dtype=torch.float64), z, torch.full_like(z, 0.99),
                         torch.full_like(z, 0.005), z_near=0.3, z_far=1.3, lambda_unknown=1.0, sigma_unknown=0.005)
    check("[A2] dense alpha=0.99 x 40 ray: finite, normalized, no-return mass ~ 0.01^40",
          math.isfinite(float(r.log_valid_depth_likelihood)) and abs(mass(r) - 1) < 1e-10
          and float(r.log_no_return_probability) < -150, f"log q_empty {float(r.log_no_return_probability):.1f}")
    r = run(0.8, [0.5, 0.6], [0.4, 0.4], 0.0)
    check("[A2] lambda = 0: no unknown mass", float(r.unknown_first_hit_mass) == 0.0)


def test_quadrature():
    print("\n[A3] closed-form unknown interval vs high-precision quadrature")
    import mpmath as mp
    from estimator.first_return_reference import _log_unknown_interval
    mp.mp.dps = 50
    worst = worst_cf = 0.0
    cases = [(0.8, 0.5, 1.0, 0.7, 0.004), (0.8, 0.85, 1.2, 2.0, 0.004), (0.4, 0.5, 1.2, 1.3, 0.003),
             (1.25, 0.3, 0.6, 0.5, 0.002), (0.6, 0.3, 1.3, 25.0, 0.01), (0.31, 0.3, 0.32, 0.1, 0.0005)]
    for D, a, b, lam, s in cases:
        logSa = -0.37
        # log-domain quadrature: the integrand is normalized by its value at its peak
        # (D - lambda s^2 clipped to [a, b]) and split at +- 2^k s/8 around it, so the
        # far-tail cases (peak tens of sigmas outside the interval) are resolved
        lf = lambda z: (logSa + mp.log(lam) - lam * (z - a) - (D - z) ** 2 / (2 * s * s)      # noqa: E731
                        - mp.log(s * mp.sqrt(2 * mp.pi)))
        pk = min(max(D - lam * s * s, a), b)
        pts = {mp.mpf(a), mp.mpf(b), mp.mpf(pk)}
        for k in range(0, 12):
            for sg in (-1, 1):
                x = pk + sg * (2 ** k) * s / 8
                if a < x < b:
                    pts.add(mp.mpf(x))
        c0 = lf(pk)
        ref = c0 + mp.log(mp.quad(lambda z: mp.e ** (lf(z) - c0), sorted(pts)))
        zl, zh = (a - D + lam * s * s) / s, (b - D + lam * s * s) / s
        dphi = (mp.ncdf(-zl) - mp.ncdf(-zh)) if zl >= 0 else (mp.ncdf(zh) - mp.ncdf(zl))
        cf = logSa + mp.log(lam) - lam * (D - a) + 0.5 * lam * lam * s * s + mp.log(dphi)
        worst_cf = max(worst_cf, abs(float(cf - ref)))
        got = float(_log_unknown_interval(torch.tensor(D, dtype=torch.float64), torch.tensor(a, dtype=torch.float64),
                                          torch.tensor(b, dtype=torch.float64), torch.tensor(logSa, dtype=torch.float64),
                                          torch.tensor(lam, dtype=torch.float64), torch.tensor(s, dtype=torch.float64)))
        err = abs(got - float(ref))
        worst = max(worst, err)
        print(f"       D={D} [{a},{b}] lambda={lam} s={s}: log I {got:.6f} vs {float(ref):.6f}")
    check("[A3] the quadrature reference agrees with the 50-digit closed form (reference sanity)",
          worst_cf < 1e-8, f"worst {worst_cf:.1e}")
    check("[A3] float64 log unknown integral within 1e-6 of 50-digit quadrature, incl. tails", worst < 1e-6,
          f"worst {worst:.1e}")


def test_gradients():
    print("\n[A4] autodiff vs finite differences of -log L")
    from estimator.first_return_reference import first_return_ray
    g = torch.Generator().manual_seed(3)
    worst = 0.0
    for trial in range(6):
        k = 1 + trial
        z0 = torch.sort(0.45 + 0.6 * torch.rand(k, generator=g, dtype=torch.float64)).values
        a0 = 0.05 + 0.9 * torch.rand(k, generator=g, dtype=torch.float64)
        D = torch.tensor(float(z0[trial % k]) + 0.01, dtype=torch.float64)

        def E(z, a):
            return -first_return_ray(D, z, a, torch.full_like(z, 0.008), z_near=0.3, z_far=1.3,
                                     lambda_unknown=1.2, sigma_unknown=0.007).log_valid_depth_likelihood
        z = z0.clone().requires_grad_(True)
        a = a0.clone().requires_grad_(True)
        gz, ga = torch.autograd.grad(E(z, a), (z, a))
        h = 1e-6
        for i in range(k):
            dz = torch.zeros(k, dtype=torch.float64); dz[i] = h
            fz = (E(z0 + dz, a0) - E(z0 - dz, a0)) / (2 * h)
            fa = (E(z0, a0 + dz) - E(z0, a0 - dz)) / (2 * h)
            worst = max(worst, abs(float(fz - gz[i])) / max(1.0, abs(float(fz))),
                        abs(float(fa - ga[i])) / max(1.0, abs(float(fa))))
    check("[A4] dz_i and dalpha_i of -log L match central FD (6 rays, 1..6 contributors)", worst < 1e-4,
          f"worst rel {worst:.1e}")
    rays = [(0.7, [0.6, 0.71, 0.9], [0.3, 0.6, 0.5]), (0.95, [0.75, 0.9], [0.9, 0.2]), (0.5, [0.8], [0.4])]
    zs = [torch.tensor(z, dtype=torch.float64, requires_grad=True) for _, z, _ in rays]
    tot = sum(-first_return_ray(torch.tensor(D, dtype=torch.float64), zz, torch.tensor(a, dtype=torch.float64),
                                torch.full_like(zz, 0.005), z_near=0.3, z_far=1.3, lambda_unknown=0.9,
                                sigma_unknown=0.005).log_valid_depth_likelihood
              for (D, _, a), zz in zip(rays, zs))
    grads = torch.autograd.grad(tot, zs)
    ok = True
    for r_i, ((D, z, a), gr) in enumerate(zip(rays, grads)):
        z0 = torch.tensor(z, dtype=torch.float64)
        for i in range(len(z)):
            dz = torch.zeros_like(z0); dz[i] = 1e-6

            def E1(zz):
                return -first_return_ray(torch.tensor(D, dtype=torch.float64), zz,
                                         torch.tensor(a, dtype=torch.float64), torch.full_like(zz, 0.005),
                                         z_near=0.3, z_far=1.3, lambda_unknown=0.9,
                                         sigma_unknown=0.005).log_valid_depth_likelihood
            fd = float((E1(z0 + dz) - E1(z0 - dz)) / 2e-6)
            ok &= abs(fd - float(gr[i])) < 1e-4 * max(1.0, abs(fd))
    check("[A4] multi-ray sum: per-ray gradients unaffected by the other rays", ok)


def test_prior_schema():
    print("\n[A5] prior schema")
    from estimator.first_return_types import content_id, load_prior, write_prior
    cams = {"cam_000": {"z_near_m": 0.2, "z_far_m": 2.0, "lambda_per_m": 0.7,
                        "unknown_depth_sigma_model": {"kind": "sensor"}},
            "cam_001": {"z_near_m": 0.25, "z_far_m": 3.0, "lambda_per_m": 0.0,
                        "unknown_depth_sigma_model": {"kind": "constant", "sigma_m": 0.004}}}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "prior.json"
        write_prior(p, cams, "assumed", {"note": "test"})
        pr = load_prior(p, ["cam_000", "cam_001"])
        check("[A5] round trip: per-camera near/far, hazard and sigma model",
              pr.camera("cam_001").z_far == 3.0 and pr.camera("cam_000").lam == 0.7
              and pr.camera("cam_001").sigma_kind == "constant")
        raw = json.loads(p.read_text())

        def refused(mut, cams_needed=None):
            r = json.loads(json.dumps(raw))
            mut(r)
            if "model_id" in r and r.get("_keep_id") is None:
                r["model_id"] = content_id(r)
            q = Path(td) / "bad.json"
            q.write_text(json.dumps(r))
            try:
                load_prior(q, cams_needed)
                return False
            except ValueError:
                return True
        res = {
            "missing camera": refused(lambda r: None, ["cam_000", "cam_009"]),
            "bad units": refused(lambda r: r.__setitem__("units", {"depth": "mm", "hazard": "1/m"})),
            "negative hazard": refused(lambda r: r["cameras"]["cam_000"]["unknown_hazard_segments"][0]
                                       .__setitem__("lambda_per_m", -1.0)),
            "piecewise hazard": refused(lambda r: r["cameras"]["cam_000"].__setitem__(
                "unknown_hazard_segments", [{"z0_m": 0.2, "z1_m": 1.0, "lambda_per_m": 0.5},
                                            {"z0_m": 1.0, "z1_m": 2.0, "lambda_per_m": 0.9}])),
            "impossible support": refused(lambda r: r["cameras"]["cam_000"].__setitem__("z_far_m", 0.1)),
            "schema": refused(lambda r: r.__setitem__("schema", 2)),
            "no provenance": refused(lambda r: r.__setitem__("provenance", {})),
        }
        r = json.loads(json.dumps(raw)); r["cameras"]["cam_000"]["unknown_hazard_segments"][0]["lambda_per_m"] = 9.0
        (Path(td) / "edited.json").write_text(json.dumps(r))
        try:
            load_prior(Path(td) / "edited.json"); res["edited content"] = False
        except ValueError:
            res["edited content"] = True
        check("[A5] refused: " + ", ".join(res), all(res.values()), str(res))
    from estimator.first_return_observation import FirstReturnObservation
    try:
        FirstReturnObservation("depth", None); ok1 = False
    except ValueError:
        ok1 = True
    try:
        FirstReturnObservation("rgbd", pr); ok2 = False
    except ValueError:
        ok2 = True
    check("[A5] depth without a prior, and the unimplemented rgbd mode, are refused", ok1 and ok2)


def main() -> int:
    for t in (test_reference, test_normalization, test_quadrature, test_gradients, test_prior_schema):
        try:
            t()
        except Exception as ex:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{t.__name__} raised", False, f"{type(ex).__name__}: {ex}"[:200])
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
