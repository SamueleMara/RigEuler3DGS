#!/usr/bin/env python
"""Audit: the EXISTING IEKF renderer's background derivative (no code changed).

C_p = sum_i w_i c_i + T_end B,  A = 1 - T_end,  so with B and c_i fixed
    dC_p = sum_i (c_i - B) dw_i = sum_i c_i dw_i - B dA.

Checks on the real CUDA IEKF operators (diff_gaussian_rasterization._C):
  [1] RGB JVP vs central finite differences of the actual renderer, for
      nonzero / non-grey backgrounds, on pixels whose contributor branch is
      unchanged by the perturbation (excluded pixels are counted)
  [2] the background identity  H_C(B) - H_C(0) = -B (x) H_A  holds exactly
      (H_A obtained here only as a TEST device: colours 1, background 0)
  [3] a splat whose colour equals the background has an RGB JVP of exactly 0
      (photometric camouflage is a property of the measurement, not a bug)
  [4] VJP adjoint identity <Hv, z> = <v, H^T z> with nonzero background
  [5] per-Gaussian Fisher blocks equal J_i^T W J_i built from explicit JVP
      columns, with nonzero background
  [6] reference outputs saved for a bitwise parity check after any rebuild

    python tests/test_background_compositing_audit.py [--save-reference DIR | --check-reference DIR]
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CHECKS, FAILURES = 0, []
BACKGROUNDS = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.3, 0.7, 0.1), (0.9, 0.2, 0.55)]


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def scene(seed=0):
    """Overlapping, rotated, anisotropic splats with distinct colours, one of
    them coloured like the 4th background, plus an opaque stack that drives
    transmittance below T_STOP (termination branch)."""
    from estimator.manifold import so3_exp
    from tests.test_fb_pixel_overlap_core import plate, state
    g = torch.Generator().manual_seed(seed)
    n = 40
    mu = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([0.5, 0.4, 0.3])
    R = so3_exp(torch.randn(n, 3, generator=g).double()).float()
    sc = 0.01 + 0.06 * torch.rand(n, 3, generator=g)
    op = 0.3 + 0.69 * torch.rand(n, generator=g)
    col = torch.rand(n, 3, generator=g)
    col[0] = torch.tensor(BACKGROUNDS[3])               # c = B for one splat
    stack = [plate(0.05, 0.0, z, 0.12, 0.12, 4, (0.8, 0.1, 0.3), opacity=0.99) for z in (0.30, 0.27, 0.24)]
    mu = torch.cat([mu] + [s[0] for s in stack])
    col = torch.cat([col] + [s[1] for s in stack])
    op = torch.cat([op] + [s[2] for s in stack])
    sc = torch.cat([sc] + [s[3] for s in stack])
    R = torch.cat([R, torch.eye(3).expand(3 * 16, 3, 3)])
    return state(mu, col, op, sc, R=R)


def lin_of(st, cam, bg, colors=None):
    from estimator.renderer_bridge import build_linearization
    s = st if colors is None else dataclasses.replace(st, colors=colors)
    return build_linearization(s, [cam], background=torch.tensor(bg, device="cuda", dtype=torch.float32)).views[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-reference", default=None)
    ap.add_argument("--check-reference", default=None)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from diff_gaussian_rasterization import iekf_apply_H, iekf_apply_HT, iekf_build_fisher, iekf_fisher_to_dense
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    cam = make_cameras(n_cam=16, H=48, W=64, radius=2.0)[0]      # the same camera the overlap suite used
    st = scene()
    P = st.mu.shape[0]
    g = torch.Generator(device="cuda").manual_seed(3)
    v = torch.randn(P, 6, device="cuda", generator=g) * torch.tensor([1, 1, 1, 0.5, 0.5, 0.5], device="cuda")
    ref = {}

    print("\n[1-3] RGB JVP with nonzero backgrounds")
    from observation_diagnostics.renderer_access import pixel_contributions

    def signature(L):
        cn = pixel_contributions(L.cache)
        HW = cn.H * cn.W
        k = torch.arange(cn.gid.numel(), device="cuda") - cn.offsets[:-1].repeat_interleave(cn.counts)
        s1 = torch.zeros(HW, dtype=torch.float64, device="cuda").index_add_(
            0, cn.pixel, (cn.gid.double() + 1) * (k.double() + 1))
        return cn.counts, s1
    vt = v.clone(); vt[:, 3:] = 0.0
    fd_err = {}
    for bg in BACKGROUNDS:
        L = lin_of(st, cam, bg)
        dC, dD = iekf_apply_H(L.cache, v)
        ref[f"render_{bg}"] = (L.rgb, L.depth, L.alpha, L.transmittance, L.n_contrib)
        ref[f"jvp_{bg}"] = (dC, dD)
        dCt, _ = iekf_apply_H(L.cache, vt)
        c0, s0 = signature(L)
        best = (float("inf"), None, 0, 0)
        for eps in (1e-2, 3e-3, 1e-3, 3e-4):
            Lp = lin_of(dataclasses.replace(st, mu=st.mu + eps * vt[:, :3]), cam, bg)
            Lm = lin_of(dataclasses.replace(st, mu=st.mu - eps * vt[:, :3]), cam, bg)
            frozen = torch.ones_like(c0, dtype=torch.bool)
            for Lx in (Lp, Lm):
                c_, s_ = signature(Lx)
                frozen &= (c_ == c0) & (s_ == s0)
            fm = frozen.reshape(L.depth.shape) & (dCt.abs().sum(0) > 0)
            fd = (Lp.rgb - Lm.rgb) / (2 * eps)
            rel = float((dCt - fd)[:, fm].norm() / fd[:, fm].norm().clamp_min(1e-12))
            if rel < best[0]:
                best = (rel, eps, int(fm.sum()), int((~frozen).sum()))
        fd_err[bg] = best[0]
        check(f"[1] B={bg}: translation JVP vs central FD (frozen contributor lists)", best[0] < 3e-2,
              f"rel {best[0]:.2e} at eps {best[1]:g} on {best[2]} px; {best[3]} branch-changing px excluded")
    spread = max(fd_err.values()) / max(min(fd_err.values()), 1e-12)
    check("[1] the FD error does not depend on the background (no missing B term)", spread < 1.5,
          f"rel errors {[round(x, 4) for x in fd_err.values()]}")
    L0 = lin_of(st, cam, (0.0, 0.0, 0.0))
    dC0, _ = iekf_apply_H(L0.cache, v)
    LA = lin_of(st, cam, (0.0, 0.0, 0.0), colors=torch.ones_like(st.colors))
    dA = iekf_apply_H(LA.cache, v)[0][0]                     # test device only: sum_i dw_i
    for bg in BACKGROUNDS[1:]:
        LB = lin_of(st, cam, bg)
        dCB, _ = iekf_apply_H(LB.cache, v)
        B = torch.tensor(bg, device="cuda")[:, None, None]
        err = float((dCB - dC0 + B * dA[None]).abs().max())
        scale = float(dC0.abs().max())
        check(f"[2] B={bg}: H_C(B) - H_C(0) = -B dA", err <= 1e-5 * max(scale, 1.0), f"max {err:.2e} (|H_C| {scale:.2e})")
    one = torch.zeros(P, dtype=torch.bool); one[0] = True
    only = dataclasses.replace(st, **{f: getattr(st, f)[one.cuda()] for f in ("mu", "R", "v", "omega", "colors", "opacity", "scales")})
    L1 = lin_of(only, cam, BACKGROUNDS[3])
    d1, _ = iekf_apply_H(L1.cache, torch.ones(1, 6, device="cuda"))
    check("[3] a splat coloured exactly like the background: RGB JVP = 0",
          float(d1.abs().max()) < 1e-6 and float(L1.alpha.max()) > 0.05,
          f"max |dC| {float(d1.abs().max()):.1e} with alpha up to {float(L1.alpha.max()):.2f}")

    print("\n[4-5] VJP and Fisher with nonzero background")
    bg = BACKGROUNDS[2]
    L = lin_of(st, cam, bg)
    gz = torch.Generator(device="cuda").manual_seed(11)   # seeded: the VJP output is a saved reference
    zc = torch.randn(L.rgb.shape, device="cuda", generator=gz); zd = torch.randn(L.depth.shape, device="cuda", generator=gz)
    dC, dD = iekf_apply_H(L.cache, v)
    HT = iekf_apply_HT(L.cache, zc, zd, L.transmittance, L.alpha, L.depth, L.n_contrib)
    lhs = float((dC.double() * zc.double()).sum() + (dD.double() * zd.double()).sum())
    rhs = float((v.double() * HT.double()).sum())
    gap = abs(lhs - rhs) / (float(torch.sqrt((dC.double() ** 2).sum() + (dD.double() ** 2).sum()))
                            * float(torch.sqrt((zc.double() ** 2).sum() + (zd.double() ** 2).sum())))
    check("[4] <Hv,z> = <v,H^T z>", gap < 1e-5, f"normalised gap {gap:.1e}")
    pc = torch.rand(L.rgb.shape, device="cuda", generator=gz); pd = torch.rand(L.depth.shape, device="cuda", generator=gz)
    F = iekf_fisher_to_dense(iekf_build_fisher(L.cache, pc, pd, L.transmittance, L.alpha, L.depth, L.n_contrib))
    ref["vjp"] = HT; ref["fisher"] = F
    worst = 0.0
    for i in (0, 5, 17, P - 1):
        cols = []
        for k in range(6):
            e = torch.zeros(P, 6, device="cuda"); e[i, k] = 1.0
            a, b = iekf_apply_H(L.cache, e)
            cols.append(torch.cat([a.reshape(-1), b.reshape(-1)]))
        J = torch.stack(cols, 1).double()
        Wd = torch.cat([pc.reshape(-1), pd.reshape(-1)]).double()
        Fe = J.T @ (Wd[:, None] * J)
        worst = max(worst, float((Fe - F[i].double()).abs().max() / Fe.abs().max().clamp_min(1e-12)))
    check("[5] Fisher block = J_i^T W J_i (explicit JVP columns)", worst < 1e-4, f"max rel {worst:.1e}")

    if args.save_reference:
        Path(args.save_reference).mkdir(parents=True, exist_ok=True)
        torch.save({k: tuple(t.cpu() for t in v) if isinstance(v, tuple) else v.cpu() for k, v in ref.items()},
                   Path(args.save_reference) / "iekf_reference.pt")
        print(f"\n  [6] reference saved to {args.save_reference}")
    if args.check_reference:
        old = torch.load(Path(args.check_reference) / "iekf_reference.pt")
        pairs = [(k, a, b.cpu()) for k in ref for a, b in
                 zip(old[k] if isinstance(old[k], tuple) else (old[k],),
                     ref[k] if isinstance(ref[k], tuple) else (ref[k],))]
        same = all(torch.equal(a, b) for _k, a, b in pairs)
        rel = {k: float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30))
               for k, a, b in pairs}
        worst = max(rel.values()) if rel else 0.0
        # the VJP/Fisher kernels accumulate with atomics, so a run of the SAME
        # binary is not bitwise reproducible; bitwise is reported, the check
        # is "no difference above float32 accumulation round-off"
        check("[6] parity with the saved reference (float32 round-off)", worst < 1e-5,
              f"bitwise {same}, worst relative diff {worst:.1e} ({max(rel, key=rel.get) if rel else '-'})")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
