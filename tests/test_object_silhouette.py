#!/usr/bin/env python
"""Object masks in the RGB-D measurement (estimator/silhouette.py), on the real CUDA renderer.

  [1] the silhouette view is C = 1 - A exactly, with the object geometry
  [2] its JVP is the alpha Jacobian (central finite difference, frozen pixels)
  [3] the two ways a background target fails inside a frozen-appearance EKF:
      a dark stray Gaussian against a black RGB target has (almost) no
      gradient (camouflage); a plain alpha target outside the mask has no
      translation gradient for a footprint wholly outside it (flat landscape)
  [4] the distance-weighted silhouette residual gives a stray Gaussian, dark
      or bright, a descent direction toward the mask, and a step along it
      brings the Gaussian closer
  [5] occluder pixels (recorded surface in front of the rendered object) are
      allowed: a Gaussian hidden behind the occluder gets no silhouette pull

    python tests/test_object_silhouette.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.test_fb_pixel_overlap_core import cams_pair, plate, state  # noqa: E402

CHECKS, FAILURES = 0, []


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def scene(stray_colour, stray_x=0.45):
    obj = plate(0.0, 0.0, 0.0, 0.3, 0.3, 8, (0.8, 0.5, 0.2))
    mu = torch.cat([obj[0], torch.tensor([[stray_x, 0.0, 0.0]])])
    col = torch.cat([obj[1], torch.tensor([stray_colour])])
    op = torch.cat([obj[2], torch.tensor([0.9])])
    sc = torch.cat([obj[3], torch.tensor([[0.04, 0.04, 0.01]])])
    return state(mu, col, op, sc), obj[0].shape[0]


def grad_translation(lin, prec_res, k):
    """Descent direction of 0.5 sum p r^2 for Gaussian k: +H^T (p r), translation part."""
    adj = [(p * r, pd * rd) for (p, pd, r, rd) in prec_res]
    g = lin.apply_HT(adj)
    return g[k, :3].double().cpu()


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from estimator.renderer_bridge import build_linearization
    from estimator.silhouette import ObjectSilhouette
    cam = cams_pair()[0]
    st_obj = state(*plate(0.0, 0.0, 0.0, 0.3, 0.3, 8, (0.8, 0.5, 0.2)))
    mask = (build_linearization(st_obj, [cam]).views[0].alpha > 0.5).cpu()
    H, W = mask.shape
    sil = ObjectSilhouette([[mask]], d0_px=5.0, band_px=1.0)
    no_depth = torch.zeros(H, W)

    print("\n[1-2] silhouette view")
    st, n_obj = scene((0.9, 0.9, 0.9))
    k = n_obj
    lin = sil.augment(build_linearization(st, [cam]), st, [cam])
    vo, vs = lin.views
    check("[1] silhouette colour is 1 - alpha (every channel)",
          float((vs.rgb - (1.0 - vo.alpha)[None]).abs().max()) < 1e-5,
          f"max {float((vs.rgb - (1.0 - vo.alpha)[None]).abs().max()):.1e}")
    keep = torch.zeros(st.mu.shape[0], dtype=torch.bool)
    keep[k] = True
    only = dataclasses.replace(st, **{f: getattr(st, f)[keep.cuda()] for f in
                                      ("mu", "R", "v", "omega", "colors", "opacity", "scales")})
    a_stray = build_linearization(only, [cam]).views[0].alpha
    check("[1] the stray Gaussian is in view and wholly outside the mask",
          float(a_stray.max()) > 0.3 and float((a_stray * mask.cuda()).sum()) < 1e-3 * float(a_stray.sum()),
          f"peak alpha {float(a_stray.max()):.2f}, share inside mask "
          f"{float((a_stray * mask.cuda()).sum()) / float(a_stray.sum()):.1e}")
    from diff_gaussian_rasterization import iekf_apply_H
    d = torch.zeros(st.mu.shape[0], 6, device="cuda")
    d[k, 0] = 1.0
    jvp_rgb, _ = iekf_apply_H(vs.cache, d)
    eps = 1e-3
    a = []
    for s in (+1, -1):
        mu = st.mu.clone()
        mu[k, 0] += s * eps
        a.append(build_linearization(dataclasses.replace(st, mu=mu), [cam]).views[0].alpha)
    fd = (a[0] - a[1]) / (2 * eps)
    rel = float((-jvp_rgb[0] - fd).norm() / fd.norm().clamp_min(1e-12))
    check("[2] silhouette JVP = -dA (central FD)", rel < 0.05, f"relative error {rel:.3f}")

    print("\n[3] why a background target is not enough in a frozen-appearance EKF")
    st_dark, _ = scene((0.0, 0.0, 0.0))
    lin_d = build_linearization(st_dark, [cam])
    v = lin_d.views[0]
    target_black = torch.zeros_like(v.rgb)
    p = torch.full_like(v.rgb, 100.0) * (~mask.cuda())[None]
    g_cam = grad_translation(lin_d, [(p, torch.zeros_like(v.depth), target_black - v.rgb,
                                      torch.zeros_like(v.depth))], k)
    lin_s = sil.augment(build_linearization(st, [cam]), st, [cam])
    vo, vs = lin_s.views
    p_flat = torch.zeros_like(vs.rgb)
    p_flat[0] = 100.0 * (~mask.cuda())
    g_flat = grad_translation(build_linearization(dataclasses.replace(st, colors=torch.zeros_like(st.colors)),
                                                  [cam], background=torch.ones(3, device="cuda")),
                              [(p_flat, torch.zeros_like(vs.depth), 1.0 - vs.rgb, torch.zeros_like(vs.depth))], k)
    p_s, pd_s, r_s, rd_s = sil.terms(0, vo, vs, no_depth, None)
    g_sil = grad_translation(lin_s, [(torch.zeros_like(vo.rgb), torch.zeros_like(vo.depth),
                                      torch.zeros_like(vo.rgb), torch.zeros_like(vo.depth)),
                                     (p_s, pd_s, r_s, rd_s)], k)
    check("[3] camouflage: a black stray Gaussian on a black target has ~no gradient",
          float(g_cam.norm()) < 1e-3 * max(float(g_sil.norm()), 1e-12),
          f"|g| {float(g_cam.norm()):.2e} vs silhouette {float(g_sil.norm()):.2e}")
    check("[3] flat landscape: a plain alpha target gives a wholly-outside footprint ~no translation gradient",
          float(g_flat[:2].norm()) < 0.05 * float(g_sil[:2].norm()),
          f"|g_xy| {float(g_flat[:2].norm()):.2e} vs silhouette {float(g_sil[:2].norm()):.2e}")

    print("\n[4] distance-weighted silhouette: a pull back to the mask")
    for name, colour in (("dark", (0.0, 0.0, 0.0)), ("bright", (0.9, 0.9, 0.9))):
        s_, _ = scene(colour)
        L = sil.augment(build_linearization(s_, [cam]), s_, [cam])
        vo, vs = L.views
        terms = sil.terms(0, vo, vs, no_depth, None)
        z = (torch.zeros_like(vo.rgb), torch.zeros_like(vo.depth), torch.zeros_like(vo.rgb), torch.zeros_like(vo.depth))
        g = grad_translation(L, [z, terms], k)
        toward = -s_.mu[k].double().cpu()          # the mask is around the origin
        check(f"[4] {name}: the descent direction points to the mask", float(g[:2] @ toward[:2]) > 0,
              f"g {g[:2].numpy().round(3)}")

        def energy(x):
            mu = s_.mu.clone()
            mu[k, :2] = torch.as_tensor(x, device="cuda", dtype=mu.dtype)
            s2 = dataclasses.replace(s_, mu=mu)
            L2 = sil.augment(build_linearization(s2, [cam]), s2, [cam])
            p2, _, r2, _ = sil.terms(0, L2.views[0], L2.views[1], no_depth, None)
            return float(0.5 * (p2 * r2 * r2).sum())
        x0 = s_.mu[k, :2].double().cpu()
        dirn = g[:2] / g[:2].norm()
        step = next((h for h in (0.08, 0.04, 0.02, 0.01) if energy((x0 + h * dirn).tolist()) < energy(x0.tolist())), None)
        check(f"[4] {name}: a step along it lowers the energy and moves the Gaussian closer",
              step is not None and float((x0 + step * dirn).norm()) < float(x0.norm()),
              f"step {step}")

    print("\n[5] occluders are allowed")
    vo, vs = lin.views
    occ_depth = torch.zeros(H, W, device="cuda")
    stray_px = (vo.alpha > 0.05) & (~mask.cuda())
    occ_depth[stray_px] = 0.5                       # something in front of the stray Gaussian (rendered at ~2 m)
    p5, _, r5, _ = sil.terms(0, vo, vs, occ_depth, occ_depth > 0)
    w_stray = float(p5[0][stray_px].sum())
    p0, _, _, _ = sil.terms(0, vo, vs, no_depth, None)
    check("[5] behind an occluder the stray Gaussian gets no silhouette weight",
          w_stray == 0.0 and float(p0[0][stray_px].sum()) > 0, f"{w_stray} vs {float(p0[0][stray_px].sum()):.1f}")
    s = sil.summary()
    check("[5] the GT-free check reports alpha outside the mask", s["max_alpha_outside_mask_fraction"] is not None)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
