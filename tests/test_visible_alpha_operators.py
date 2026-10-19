#!/usr/bin/env python
"""Native visible-alpha operators (diff_gaussian_rasterization._C_vis) on the real renderer.

  [1] full-empty rays (mode 1): A_vis equals the renderer's own accumulated
      alpha and its last-contributor index, pixel for pixel
  [2] alpha JVP vs central finite differences of the ACTUAL native renderer,
      translations and anisotropic rotations, on frozen contributor lists
  [3] front-of-surface rays (mode 2): the gate uses each splat's OWN depth --
      a splat behind the measured surface contributes nothing, one in front
      does, and the composited normalized depth plays no role
  [4] overlapping splats compose (1 - prod(1 - alpha)), not sum
  [5] colour independence: A_vis and dA_vis do not change with the colours
      (a black splat on a black background cannot hide)
  [6] adjoint identity <H_A v, z> = <v, H_A^T z> for mixed ray modes, and the
      combined RGB-D + alpha operator is still a transpose pair
  [7] front-alpha JVP vs FD inside one gate branch (no splat crosses z_obs)

    python tests/test_visible_alpha_operators.py
"""
from __future__ import annotations

import dataclasses
import sys
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


def lin_of(st, cam, bg=(0.0, 0.0, 0.0)):
    from estimator.renderer_bridge import build_linearization
    return build_linearization(st, [cam], background=torch.tensor(bg, device="cuda", dtype=torch.float32)).views[0]


def signature(L):
    from observation_diagnostics.renderer_access import pixel_contributions
    cn = pixel_contributions(L.cache)
    k = torch.arange(cn.gid.numel(), device="cuda") - cn.offsets[:-1].repeat_interleave(cn.counts)
    s1 = torch.zeros(cn.H * cn.W, dtype=torch.float64, device="cuda").index_add_(
        0, cn.pixel, (cn.gid.double() + 1) * (k.double() + 1))
    return cn.counts, s1


def perturbed(st, v, eps):
    from estimator.manifold import so3_exp
    R = so3_exp((eps * v[:, 3:]).double()).float().cuda() @ st.R
    return dataclasses.replace(st, mu=st.mu + eps * v[:, :3], R=R.contiguous())


def fd_check(st, cam, v, mode, zobs, dz, label):
    from diff_gaussian_rasterization import iekf_vis_apply_H, iekf_visible_alpha
    L = lin_of(st, cam)
    jv = iekf_vis_apply_H(L.cache, mode, zobs, dz, v)
    c0, s0 = signature(L)
    best = (float("inf"), None, 0)
    for eps in (3e-3, 1e-3, 3e-4):
        Lp, Lm = lin_of(perturbed(st, v, eps), cam), lin_of(perturbed(st, v, -eps), cam)
        fr = torch.ones_like(c0, dtype=torch.bool)
        for Lx in (Lp, Lm):
            c_, s_ = signature(Lx)
            fr &= (c_ == c0) & (s_ == s0)
        ap = iekf_visible_alpha(Lp.cache, mode, zobs, dz)
        am = iekf_visible_alpha(Lm.cache, mode, zobs, dz)
        same_gate = (ap[3] == iekf_visible_alpha(L.cache, mode, zobs, dz)[3]) & (am[3] == ap[3])
        m = fr.reshape(L.depth.shape) & same_gate & (jv.abs() > 0)
        fd = (ap[0] - am[0]) / (2 * eps)
        rel = float((jv - fd)[m].norm() / fd[m].norm().clamp_min(1e-12)) if bool(m.any()) else float("nan")
        if rel < best[0]:
            best = (rel, eps, int(m.sum()))
    check(label, best[0] < 3e-2, f"rel {best[0]:.2e} at eps {best[1]} on {best[2]} px")


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from diff_gaussian_rasterization import (iekf_apply_H, iekf_apply_HT, iekf_vis_apply_H,
                                             iekf_vis_apply_HT, iekf_visible_alpha)
    from tests.test_background_compositing_audit import scene
    from tests.test_fb_pixel_overlap_core import cams_pair, plate, state
    cam = cams_pair()[0]
    st = scene()
    P = st.mu.shape[0]
    L = lin_of(st, cam)
    H, W = L.depth.shape
    full = torch.ones(H, W, dtype=torch.int32, device="cuda")
    zero_z = torch.zeros(H, W, device="cuda")

    print("\n[1] full-empty rays")
    A, Tf, nc, nf, nb = iekf_visible_alpha(L.cache, full, zero_z, 0.0)
    check("[1] A_vis = renderer alpha", float((A - L.alpha).abs().max()) < 1e-6, f"max {float((A - L.alpha).abs().max()):.1e}")
    check("[1] same last-contributor index as the renderer", torch.equal(nc.to(L.n_contrib.dtype), L.n_contrib))

    print("\n[2] alpha JVP vs FD of the native renderer")
    g = torch.Generator(device="cuda").manual_seed(5)
    vt = torch.zeros(P, 6, device="cuda"); vt[:, :3] = torch.randn(P, 3, device="cuda", generator=g)
    vr = torch.zeros(P, 6, device="cuda"); vr[:, 3:] = torch.randn(P, 3, device="cuda", generator=g)
    fd_check(st, cam, vt, full, zero_z, 0.0, "[2] translations")
    fd_check(st, cam, vr, full, zero_z, 0.0, "[2] anisotropic rotations")

    print("\n[3-4] front-of-surface gate and composition")
    a = plate(0.0, 0.0, 0.2, 0.2, 0.2, 5, (0.0, 0.0, 0.0), opacity=0.6)     # nearer the camera (z = 2 looks down -z)
    b = plate(0.0, 0.0, -0.2, 0.2, 0.2, 5, (0.0, 0.0, 0.0), opacity=0.6)    # farther
    two = state(torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]]), torch.cat([a[2], b[2]]), torch.cat([a[3], b[3]]))
    sa = state(*a); sb = state(*b)
    L2, La, Lb = lin_of(two, cam), lin_of(sa, cam), lin_of(sb, cam)
    da_ = float(La.cache.depth.mean()); db_ = float(Lb.cache.depth.mean())
    z_mid = torch.full((H, W), 0.5 * (da_ + db_), device="cuda")
    front = torch.full((H, W), 2, dtype=torch.int32, device="cuda")
    A2 = iekf_visible_alpha(L2.cache, front, z_mid, 0.01)
    Aa = iekf_visible_alpha(La.cache, full, zero_z, 0.0)[0]
    check("[3] with the surface between the plates, A_vis is the FRONT plate's alpha only",
          float((A2[0] - Aa).abs().max()) < 1e-5, f"max {float((A2[0] - Aa).abs().max()):.1e}")
    check("[3] the behind plate is counted as behind, not as a violation", int(A2[4].sum()) > 0 and int(A2[3].sum()) > 0)
    Dcomp = L2.depth[L2.alpha > 0.5]
    check("[3] (the composited normalized depth lies between the plates and is not used)",
          bool(((Dcomp > da_) & (Dcomp < db_)).any()))
    A_full2 = iekf_visible_alpha(L2.cache, full, zero_z, 0.0)[0]
    Ab = iekf_visible_alpha(Lb.cache, full, zero_z, 0.0)[0]
    comp = 1 - (1 - Aa) * (1 - Ab)
    m = (Aa > 0.05) & (Ab > 0.05)
    check("[4] overlapping splats compose as 1 - (1-A_a)(1-A_b), not A_a + A_b",
          float((A_full2 - comp)[m].abs().max()) < 1e-4 and float((Aa + Ab)[m].max()) > 1.0 + 1e-3,
          f"max dev {float((A_full2 - comp)[m].abs().max()):.1e}, sum reaches {float((Aa + Ab)[m].max()):.2f}")

    print("\n[5] colour independence")
    v = vt + vr
    st_black = dataclasses.replace(st, colors=torch.zeros_like(st.colors))
    Lk = lin_of(st_black, cam)
    j1 = iekf_vis_apply_H(L.cache, full, zero_z, 0.0, v)
    j2 = iekf_vis_apply_H(Lk.cache, full, zero_z, 0.0, v)
    check("[5] A_vis and dA_vis are identical for black splats on a black background",
          torch.equal(iekf_visible_alpha(Lk.cache, full, zero_z, 0.0)[0], A) and torch.equal(j1, j2)
          and float(j1.abs().max()) > 0)

    print("\n[6] adjoint identities")
    mix = torch.randint(0, 3, (H, W), device="cuda", dtype=torch.int32, generator=g)
    zmix = 1.5 + 1.0 * torch.rand(H, W, device="cuda", generator=g)
    Am, Tfm, ncm, _, _ = iekf_visible_alpha(L.cache, mix, zmix, 0.01)
    za = torch.randn(H, W, device="cuda", generator=g)
    Hv = iekf_vis_apply_H(L.cache, mix, zmix, 0.01, v)
    HTz = iekf_vis_apply_HT(L.cache, mix, zmix, 0.01, za, Tfm, ncm)
    lhs = float((Hv.double() * za.double()).sum()); rhs = float((v.double() * HTz.double()).sum())
    gap = abs(lhs - rhs) / (float(Hv.double().norm()) * float(za.double().norm()))
    check("[6] <H_A v, z> = <v, H_A^T z> (mixed modes 0/1/2)", gap < 1e-5, f"normalised gap {gap:.1e}")
    zc = torch.randn_like(L.rgb); zd = torch.randn_like(L.depth)
    dC, dD = iekf_apply_H(L.cache, v)
    HTr = iekf_apply_HT(L.cache, zc, zd, L.transmittance, L.alpha, L.depth, L.n_contrib)
    lhs2 = lhs + float((dC.double() * zc.double()).sum() + (dD.double() * zd.double()).sum())
    rhs2 = rhs + float((v.double() * HTr.double()).sum())
    nrm = float(torch.sqrt(Hv.double().norm() ** 2 + dC.double().norm() ** 2 + dD.double().norm() ** 2)
                * torch.sqrt(za.double().norm() ** 2 + zc.double().norm() ** 2 + zd.double().norm() ** 2))
    check("[6] combined RGB-D + alpha operator is a transpose pair", abs(lhs2 - rhs2) / nrm < 1e-5,
          f"normalised gap {abs(lhs2 - rhs2) / nrm:.1e}")

    print("\n[7] front-alpha JVP inside one gate branch")
    fd_check(two, cam, torch.randn(two.mu.shape[0], 6, device="cuda", generator=g) * torch.tensor(
        [1, 1, 0.2, 0.5, 0.5, 0.5], device="cuda"), front, z_mid, 0.01, "[7] mode 2, translations + rotations")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
