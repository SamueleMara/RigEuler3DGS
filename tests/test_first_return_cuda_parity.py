#!/usr/bin/env python
"""Gates B and C: the first-return CUDA operators against the CPU oracle and themselves.

  [B1] the FULL contributor dump, composited with the production T_STOP rule,
       reproduces the production render (RGB, A, Z/(A+eps))
  [B2] the dump continues past T_STOP (a deeper contributor is never dropped)
  [C1] CUDA log L == float64 CPU oracle on every valid pixel
  [C2] score (VJP) == central FD of sum log L at the nonlinear render, translation
       and left rotation (best step of a ladder, away from sorting swaps)
  [C3] <J v, w> == <v, J^T w> for the match, visibility and full parts
  [C4] Fisher-vis blocks == explicit sum_p g_p g_p^T restricted to each Gaussian
  [C5] the normal operator is PSD and equals sum W (e.v)^2 + sum_p (g_p . v)^2
  [C6] invalid / mask-zero pixels have exactly zero effect, whatever depth they hold
  [C7] runtime and peak memory on a 640 x 480 x 3-camera scene (reported)

    python tests/test_first_return_cuda_parity.py
"""
from __future__ import annotations

import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "observation_diagnostics"))
CHECKS, FAILURES = 0, []
LAM, ZN, ZF, SD = 0.6, 0.2, 5.0, 0.004


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def scene(H=48, W=48, ncam=3):
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    from tests.test_visibility_constrained_iekf_phase1 import blob, cat
    from estimator.manifold import so3_exp
    cams = make_cameras(n_cam=ncam, H=H, W=W, radius=2.2)
    st = cat(blob(n=40, seed=0), blob(n=25, seed=1, centre=(0.05, 0.02, -0.12)))
    # anisotropic footprints in random orientations: an isotropic Gaussian is rotation
    # invariant, so a rotation finite difference would compare two zeros
    g = torch.Generator().manual_seed(7)
    n = st.mu.shape[0]
    sc = torch.tensor([0.05, 0.025, 0.012]).repeat(n, 1) * (0.7 + 0.6 * torch.rand(n, 1, generator=g))
    R = so3_exp(torch.randn(n, 3, generator=g)).float()
    return cams, dataclasses.replace(st, scales=sc.cuda(), R=R.cuda().contiguous())


def observation(st, cams, shift=(0.01, 0.0, 0.0), noise=0.001, seed=0):
    from estimator.renderer_bridge import build_linearization
    moved = dataclasses.replace(st, mu=st.mu + torch.tensor(shift, device="cuda"))
    lin = build_linearization(moved, cams)
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for v in lin.views:
        valid = v.alpha > 0.5
        D = v.depth + noise * torch.randn(v.depth.shape, device="cuda", generator=g)
        out.append((torch.where(valid, D, torch.zeros_like(D)), valid))
    return out


def args_for(view, D, valid):
    from estimator import first_return_operators as fo
    sd = torch.full_like(D, SD)
    return fo.camera_args(view.cache, D, valid, sd, sd.clone(), LAM, ZN, ZF)


def total_logL(st, cams, obs):
    from estimator import first_return_operators as fo
    from estimator.renderer_bridge import build_linearization
    lin = build_linearization(st, cams)
    tot = 0.0
    for v, (D, valid) in zip(lin.views, obs):
        tot += float(fo.fr_eval(args_for(v, D, valid), diagnostics=False)["logL"][valid].sum())
    return tot


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from estimator import first_return_operators as fo
    from estimator.first_return_reference import first_return_ray
    from estimator.renderer_bridge import build_linearization, embed_pose_to_full_state
    from estimator.state import boxplus
    from first_return_audit import renderer_parity
    cams, st = scene()
    obs = observation(st, cams)
    lin = build_linearization(st, cams)

    print("\n[B] contributor dump vs the production renderer")
    worst = {"alpha_max_abs": 0.0, "rgb_max_abs": 0.0, "depth_max_abs_where_A>0.5": 0.0}
    past = 0
    for v, (D, valid) in zip(lin.views, obs):
        a = args_for(v, D, valid)
        dump = fo.fr_dump(a)
        par = renderer_parity(v, dump, v.cache.colors, v.cache.background, v.cache.eps_depth)
        for k in worst:
            worst[k] = max(worst[k], par[k])
        cnt = (dump["offsets"][1:] - dump["offsets"][:-1]).reshape(v.depth.shape)
        past += int((cnt > v.n_contrib.to(cnt.dtype)).sum())
    check("[B1] dump composited with T_STOP == production (RGB, A, D)",
          worst["alpha_max_abs"] < 1e-5 and worst["rgb_max_abs"] < 1e-5 and worst["depth_max_abs_where_A>0.5"] < 1e-4,
          str({k: f"{x:.1e}" for k, x in worst.items()}))
    # a dense opaque stack: 8 Gaussians of opacity 0.99 one behind the other, so the
    # production renderer stops at T_STOP after the second one
    from tests.test_visibility_constrained_iekf_phase1 import blob
    stk = blob(n=8, seed=3)
    stk = dataclasses.replace(stk, mu=torch.stack([torch.tensor([0.0, 0.0, -0.3 + 0.05 * i]) for i in range(8)]).cuda(),
                              opacity=torch.full((8, 1), 0.99, device="cuda"))
    vs = build_linearization(stk, cams[:1]).views[0]
    Ds = torch.full_like(vs.depth, 2.0); vals = vs.alpha > 0.5
    dump = fo.fr_dump(args_for(vs, Ds, vals))
    cnt = (dump["offsets"][1:] - dump["offsets"][:-1]).reshape(vs.depth.shape)
    past += int((cnt > vs.n_contrib.to(cnt.dtype)).sum())
    check("[B2] the first-return list continues past T_STOP (dense opaque stack)", past > 0,
          f"{past} pixels with contributors the renderer dropped")

    print("\n[C1] CUDA log L vs the float64 CPU oracle")
    err = 0.0; npx = 0
    for v, (D, valid) in zip(lin.views, obs):
        a = args_for(v, D, valid)
        r = fo.fr_eval(a)
        dump = fo.fr_dump(a)
        off = dump["offsets"].cpu(); al = dump["alpha"].double().cpu(); z = dump["z"].double().cpu()
        Dv = D.reshape(-1).double().cpu(); lg = r["logL"].reshape(-1).cpu()
        for p in torch.nonzero(valid.reshape(-1)).reshape(-1).cpu().tolist():
            zz, aa = z[off[p]:off[p + 1]], al[off[p]:off[p + 1]]
            keep = (zz > ZN) & (zz <= ZF)
            ref = first_return_ray(Dv[p], zz[keep], aa[keep], torch.full((int(keep.sum()),), SD, dtype=torch.float64),
                                   z_near=ZN, z_far=ZF, lambda_unknown=LAM, sigma_unknown=SD)
            err = max(err, abs(float(ref.log_valid_depth_likelihood) - float(lg[p])))
            npx += 1
    check("[C1] CUDA log L == CPU oracle on every valid pixel", err < 1e-6, f"{npx} pixels, max |err| {err:.1e}")

    print("\n[C2] score vs finite differences of the nonlinear sum log L")
    score = None
    for v, (D, valid) in zip(lin.views, obs):
        a = args_for(v, D, valid)
        r = fo.fr_eval(a, diagnostics=False)
        g = fo.fr_vjp(a, r["logL"], valid.float(), fo.PART_ALL)
        score = g if score is None else score + g
    gen = torch.Generator(device="cpu").manual_seed(5)
    for name, sel in (("translation", slice(0, 3)), ("rotation", slice(3, 6))):
        base = torch.zeros((st.mu.shape[0], 6), dtype=torch.float64)
        base[:, sel] = torch.randn((st.mu.shape[0], 3), generator=gen, dtype=torch.float64)
        best = float("inf")
        for eps in (1e-3, 3e-4, 1e-4, 3e-5):
            vdir = (base * eps).float().cuda()
            f = embed_pose_to_full_state(vdir)
            fd = (total_logL(boxplus(st, f), cams, obs) - total_logL(boxplus(st, -f), cams, obs)) / 2
            an = float((score.double() * vdir.double()).sum())
            best = min(best, abs(fd - an) / max(abs(fd), 1e-12))
        check(f"[C2] d(sum log L) along a random {name}: VJP == central FD", best < 2e-2, f"best rel {best:.1e}")

    print("\n[C3] adjoint identity <J v, w> = <v, J^T w>")
    for parts, nm in ((fo.PART_MATCH, "match"), (fo.PART_VIS, "visibility"), (fo.PART_ALL, "full")):
        worst = 0.0
        for v, (D, valid) in zip(lin.views, obs):
            a = args_for(v, D, valid)
            r = fo.fr_eval(a, diagnostics=False)
            dv = torch.randn((st.mu.shape[0], 6), device="cuda")
            w = torch.randn(D.shape, device="cuda") * valid.float()
            jv = fo.fr_jvp(a, r["logL"], dv, parts)
            jtw = fo.fr_vjp(a, r["logL"], w, parts)
            lhs, rhs = float((jv.double() * w.double()).sum()), float((dv.double() * jtw.double()).sum())
            worst = max(worst, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12))
        check(f"[C3] adjoint identity, {nm} part", worst < 1e-3, f"rel {worst:.1e}")

    print("\n[C4] Fisher-vis blocks vs explicit per-Gaussian outer products")
    v0, (D0, valid0) = lin.views[0], obs[0]
    a0 = args_for(v0, D0, valid0)
    r0 = fo.fr_eval(a0, diagnostics=False)
    F = fo.sym21_to_dense(fo.fr_fisher_vis(a0, r0["logL"]))
    N = st.mu.shape[0]
    worst = 0.0
    for gidx in range(N):
        cols = []
        for b in range(6):
            e = torch.zeros((N, 6), device="cuda"); e[gidx, b] = 1.0
            cols.append(fo.fr_jvp(a0, r0["logL"], e, fo.PART_VIS).reshape(-1).double())
        G = torch.stack(cols, 1)
        Fe = G.T @ G
        worst = max(worst, float((Fe - F[gidx].double()).abs().max() / Fe.abs().max().clamp_min(1e-12)))
    check("[C4] fr_fisher_vis == sum_p g_p g_p^T per Gaussian (6 JVPs per Gaussian)", worst < 1e-3, f"rel {worst:.1e}")

    print("\n[C5] normal operator")
    from estimator.first_return_observation import _Cam, FirstReturnEvaluation
    cam_objs = []
    for v, (D, valid) in zip(lin.views, obs):
        a = args_for(v, D, valid)
        r = fo.fr_eval(a)
        cam_objs.append(_Cam("c", a, r["logL"], valid, r["w_match"], fo.depth_axis(v.cache).cuda()))
    ev = FirstReturnEvaluation(0.0, cam_objs, "POSTERIOR_UNCERTIFIED", {"cameras": []})
    worst_neg, worst_eq = 0.0, 0.0
    for t in range(6):
        vv = torch.randn((N, 6), device="cuda")
        Bv = ev.normal_pose(vv)
        q = float((vv.double() * Bv.double()).sum())
        ref = 0.0
        for c in cam_objs:
            ref += float((c.w_match.double() * (vv[:, :3] @ c.axis).double() ** 2).sum())
            ref += float((fo.fr_jvp(c.args, c.logL, vv, fo.PART_VIS).double() * c.valid.double()).pow(2).sum())
        worst_neg = min(worst_neg, q / max(abs(ref), 1e-12))
        worst_eq = max(worst_eq, abs(q - ref) / max(abs(ref), 1e-12))
    check("[C5] v^T B v >= 0 (PSD) on random directions", worst_neg >= -1e-6, f"min ratio {worst_neg:.1e}")
    check("[C5] v^T B v == sum W (e.v)^2 + sum_p (g_p.v)^2", worst_eq < 1e-3, f"rel {worst_eq:.1e}")
    Fb = ev.fisher_pose()
    ev2 = torch.linalg.eigvalsh(0.5 * (Fb + Fb.transpose(1, 2)).double().cpu())
    check("[C5] Fisher blocks PSD", float(ev2.min()) >= -1e-6 * float(ev2.abs().max()))

    print("\n[C6] invalid / mask-zero pixels")
    D, valid = obs[0]
    a = args_for(v0, D, valid)
    r = fo.fr_eval(a)
    Dg = torch.where(valid, D, torch.full_like(D, 1e6))
    Dg[0, 0] = float("nan")
    ag = args_for(v0, Dg, valid)
    rg = fo.fr_eval(ag)
    g1, g2 = fo.fr_vjp(a, r["logL"], valid.float()), fo.fr_vjp(ag, rg["logL"], valid.float())
    # float atomics: the score is reproducible only to accumulation order
    same = torch.equal(r["logL"], rg["logL"]) and float((g1 - g2).abs().max()) <= 1e-5 * float(g1.abs().max())
    az = args_for(v0, D, torch.zeros_like(valid))
    rz = fo.fr_eval(az)
    zero = float(rz["logL"].abs().sum()) == 0.0 and float(fo.fr_vjp(az, rz["logL"], torch.ones_like(D)).abs().sum()) == 0.0
    check("[C6] depth outside the valid set (1e6 / NaN) changes nothing; no valid pixel -> exactly zero",
          same and zero)

    print("\n[C7] runtime / memory, 640 x 480 x 3 cameras (reported)")
    camsL, stL = scene(H=480, W=640, ncam=3)
    stL = dataclasses.replace(stL, scales=stL.scales * 0.3)
    obsL = observation(stL, camsL)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    linL = build_linearization(stL, camsL)
    torch.cuda.synchronize(); t0 = time.time()
    rs = []
    for v, (D, valid) in zip(linL.views, obsL):
        aL = args_for(v, D, valid)
        rs.append((aL, fo.fr_eval(aL, diagnostics=True), valid))
    torch.cuda.synchronize(); t_eval = time.time() - t0
    t0 = time.time()
    for aL, r, valid in rs:
        fo.fr_vjp(aL, r["logL"], valid.float())
    torch.cuda.synchronize(); t_vjp = time.time() - t0
    t0 = time.time()
    vv = torch.randn((stL.mu.shape[0], 6), device="cuda")
    for aL, r, valid in rs:
        s = fo.fr_jvp(aL, r["logL"], vv, fo.PART_VIS)
        fo.fr_vjp(aL, r["logL"], s * valid.float(), fo.PART_VIS)
    torch.cuda.synchronize(); t_norm = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() - m0) / 2 ** 20
    print(f"       eval {t_eval * 1000:.1f} ms, score {t_vjp * 1000:.1f} ms, one normal-operator product "
          f"{t_norm * 1000:.1f} ms, peak extra memory {peak:.0f} MiB")
    check("[C7] runtime / memory measured", True)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
