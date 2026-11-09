#!/usr/bin/env python
"""Gate D toy physics on the CUDA operators (one camera looking down +z).

Scene: a 'door' sheet of Gaussians at z = 2.0 m (opacity 0.95) in front of a 'cavity'
sheet at z = 2.3 m that covers the same pixels.

  [O1] door observed (D = 2.0): the door carries the depth information; the hidden
       cavity has no direct depth score and no visibility score
  [O2] the observed door moved 1 cm deeper: the door's depth score points toward it
  [O3] cavity observed (D = 2.3) through a WRONG modelled door of effective alpha a:
       log L drops by ~log(1 - a) against the same scene without the door; the
       contradiction is reported (front map, per-Gaussian front sums) -- it is not
       hidden by the unknown branch
  [O4] a new unmodelled surface (pixels with no contributors) contributes exactly zero
       score to every Gaussian
  [O5] lambda_U = 0: the unknown branch is off and the false-front penalty is total

    python tests/test_first_return_occlusion.py
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CHECKS, FAILURES = 0, []
SD = 0.004


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def sheet(z, opacity, n=5, span=0.12, seed=0):
    from estimator.state import GaussianMotionState
    xs = torch.linspace(-span, span, n)
    xy = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    m = xy.shape[0]
    mu = torch.cat([xy, torch.full((m, 1), z)], 1)
    zz = torch.zeros(m, 3)
    return GaussianMotionState(mu=mu.cuda(), R=torch.eye(3).repeat(m, 1, 1).cuda(), v=zz.cuda(), omega=zz.cuda(),
                               colors=torch.full((m, 3), 0.5).cuda(), opacity=torch.full((m, 1), opacity).cuda(),
                               scales=torch.tensor([0.04, 0.04, 0.004]).repeat(m, 1).cuda())


def evaluate(st, cam, D, valid, lam=0.5):
    from estimator import first_return_operators as fo
    from estimator.renderer_bridge import build_linearization
    v = build_linearization(st, [cam]).views[0]
    sd = torch.full_like(D, SD)
    args = fo.camera_args(v.cache, D, valid, sd, sd.clone(), lam, 0.2, 6.0)
    r = fo.fr_eval(args)
    return v, args, r


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from estimator import first_return_operators as fo
    from tests.test_iekf_end_to_end import camera
    from tests.test_visibility_constrained_iekf_phase1 import cat
    cam = camera(width=64, height=64, fov=0.8)
    door, cav = sheet(2.0, 0.95), sheet(2.3, 0.95, seed=1)
    both = cat(door, cav)
    nd = door.mu.shape[0]
    v, _a, _r = evaluate(both, cam, torch.zeros(64, 64, device="cuda"), torch.zeros(64, 64, dtype=torch.bool, device="cuda"))
    covered = v.alpha > 0.9

    print("\n[O1] door observed, cavity hidden")
    D = torch.where(covered, torch.full_like(v.depth, 2.0), torch.zeros_like(v.depth))
    _v, args, r = evaluate(both, cam, D, covered)
    sm = fo.fr_vjp(args, r["logL"], covered.float(), fo.PART_MATCH)
    sv = fo.fr_vjp(args, r["logL"], covered.float(), fo.PART_VIS)
    door_m, cav_m = float(sm[:nd].norm()), float(sm[nd:].norm())
    door_all = float((sm[:nd] + sv[:nd]).norm()); cav_all = float((sm[nd:] + sv[nd:]).norm())
    check("[O1] the hidden cavity has no direct depth score", cav_m < 1e-6 * max(door_m, 1.0) or cav_m < 1e-3,
          f"|match| door {door_m:.2e} cavity {cav_m:.2e}")
    check("[O1] ...and no visibility score of comparable size", cav_all < 1e-3 * max(door_all, 1.0),
          f"|full| door {door_all:.2e} cavity {cav_all:.2e}")
    check("[O1] door pixels are explained by modelled hits", float(r["gamma_unknown"][covered].mean()) < 0.05,
          f"mean gamma_U {float(r['gamma_unknown'][covered].mean()):.3f}")

    print("\n[O2] observed door 1 cm deeper")
    D2 = torch.where(covered, torch.full_like(v.depth, 2.01), torch.zeros_like(v.depth))
    _v, args2, r2 = evaluate(both, cam, D2, covered)
    s2 = fo.fr_vjp(args2, r2["logL"], covered.float(), fo.PART_ALL)
    ez = fo.depth_axis(_v.cache).cuda()
    toward = float((s2[:nd, :3] @ ez).sum())
    check("[O2] the door's score pushes it deeper (toward the observation)", toward > 0, f"sum e_z . score {toward:.2e}")

    print("\n[O3] cavity observed through a wrong modelled door")
    D3 = torch.where(covered, torch.full_like(v.depth, 2.3), torch.zeros_like(v.depth))
    _vw, args_w, rw = evaluate(both, cam, D3, covered)
    _vc, args_c, rc = evaluate(cav, cam, D3, covered)
    dump = fo.fr_dump(args_w)
    off, gid, al = dump["offsets"].cpu(), dump["gid"].cpu(), dump["alpha"].double().cpu()
    errs = []
    for pix in torch.nonzero(covered.reshape(-1)).reshape(-1).tolist()[::7]:
        front = [float(al[e]) for e in range(off[pix], off[pix + 1]) if int(gid[e]) < nd]
        pen = sum(math.log1p(-a) for a in front)
        dl = float(rw["logL"].reshape(-1)[pix] - rc["logL"].reshape(-1)[pix])
        errs.append(abs(dl - pen))
    check("[O3] log L drops by sum log(1 - alpha_front) (the survival factor), not hidden by unknown",
          max(errs) < 0.05, f"max |dlogL - log(1-a)| {max(errs):.3f} over {len(errs)} pixels")
    # per-pixel contradiction -log(1 - alpha_eff Phi((D - z)/sigma)), alpha_eff the footprint opacity
    # at the pixel: the door is wholly in front of D (Phi ~ 1), the cavity lies ON the observed
    # surface (Phi = 1/2), so per pixel the door must dominate
    pg, pc = rw["front_sum"], rw["pixel_count"].clamp_min(1)
    hit = rw["pixel_count"] > 0
    dpp = float((pg[:nd][hit[:nd]] / pc[:nd][hit[:nd]]).median())
    cpp = float((pg[nd:][hit[nd:]] / pc[nd:][hit[nd:]]).median())
    check("[O3] the false front is reported per Gaussian: door >> cavity on the observed surface",
          cpp < -math.log(1 - 0.95 / 2) * 1.05 and dpp > 2 * cpp and float(pg[:nd].sum()) > 2 * float(pg[nd:].sum()),
          f"median front per pixel: door {dpp:.2f}, cavity {cpp:.2f} (cavity on the surface: {-math.log(1 - 0.475):.2f})")

    print("\n[O4] a new unmodelled surface")
    empty = torch.zeros_like(covered)
    empty[:4, :4] = True                         # an image corner with no contributors
    vv, args4, r4 = evaluate(both, cam, torch.where(empty, torch.full_like(v.depth, 3.0), torch.zeros_like(v.depth)), empty)
    nsup = int(r4["n_support"][empty].sum())
    s4 = fo.fr_vjp(args4, r4["logL"], empty.float(), fo.PART_ALL)
    check("[O4] pixels with no contributors give exactly zero score to every Gaussian",
          nsup == 0 and float(s4.abs().max()) == 0.0 and float(r4["gamma_unknown"][empty].min()) > 0.999,
          f"support {nsup}, max |score| {float(s4.abs().max()):.1e}")

    print("\n[O5] lambda_U = 0")
    _a5, _b5, r5 = evaluate(door, cam, D3, covered, lam=0.0)
    _a6, _b6, r6 = evaluate(door, cam, D3, covered, lam=0.5)
    l0, l5 = float(r5["logL"][covered].mean()), float(r6["logL"][covered].mean())
    check("[O5] door only, D = 2.3: with lambda_U = 0 the unmatched deeper return is (nearly) impossible",
          l0 < l5 - 100, f"mean log L {l0:.1f} (lambda 0) vs {l5:.2f} (lambda 0.5)")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
