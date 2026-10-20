#!/usr/bin/env python
"""Masked-RGB / foreground-valid-depth observation (estimator/masked_observation.py),
CODEX_IMPLEMENT_EXACT_MASKED_RGB_VALID_DEPTH_IEKF_2026-10-20.md section 6, on the real
CUDA renderer (toy scenes; the full-scene benchmark L is a separate run).

  [A] prepared RGB: P = M I, Y = P + (1-M) b; M=1 -> I, M=0 -> b; camera/frame
      association by ORIGINAL frame incl. reverse processing order; strict loader
  [B] coverage-correct 2x2 -> 1x1 resize = area mean of the native composite, any b
  [C] depth outside the mask set to 0 / NaN / 1e6 / noise: energy identical, RHS and
      the IEKF update identical to the CUDA floor
  [D] C = F + (1-A) b in the real renderer; dC(b) - dC(0) = -b dA (renderer JVP vs
      the full-ray alpha JVP); RGB JVP vs central FD with b != 0 incl. rotation
  [E] exact background expectation vs the 8-point moment-matching cubature:
      energy (diagonal and cross-channel W), gradient and normal operator
  [F] alpha operators: A = 1 - T_end, full-ray A == v.alpha, FD (translation,
      rotation), adjoint, alpha Fisher == explicit J^T W J (new kernel), PSD
  [G] stage lifts (Phase 1 supernode, Phase 2 rigid, Phase 3 displacement):
      (C, D, A) JVP vs central FD of the whole stage-to-render map
  [H] measurement_operators vs explicitly built dense Jacobian columns:
      normal = H_C^T W H_C + H_A^T K H_A + H_D^T W_D H_D, RHS, Fisher blocks
  [I] true acceptance: a candidate that lowers the RGB energy but raises the total
      exact objective is rejected; the objective is fixed across rejected trials
  [K] the masked mode runs no QP / constraint / silhouette code and renders exactly
      once per camera per nonlinear evaluation
  [M] conflicting flags are refused
  [N] identifiability: an isolated Gaussian in a uniform negative region has a
      nonzero background cost, no RGB gradient, and a gradient that does not depend on
      where (or whether) the object mask is -- i.e. no attraction toward the mask
  [P] end to end: masked Phase 1 frame, rigid Phase 2 window and fixed-joint Phase 3
      window move toward the observation, joint geometry bitwise unchanged

    python tests/test_masked_rgb_observation.py
"""
from __future__ import annotations

import copy
import dataclasses
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
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


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def prepared_from(gt, masks):
    from estimator.masked_observation import prepare
    return [prepare(g[0], g[1], g[2], m, f"cam_{i:03d}", 1) for i, (g, m) in enumerate(zip(gt, masks))]


def zero_prior(st):
    return torch.zeros((st.mu.shape[0], 12, 12), device="cuda")


def frozen_mask(L0, Lp, Lm):
    """Pixels whose contributor list is identical at the base and both FD points."""
    from observation_diagnostics.renderer_access import pixel_contributions

    def sig(L):
        cn = pixel_contributions(L.cache)
        k = torch.arange(cn.gid.numel(), device="cuda") - cn.offsets[:-1].repeat_interleave(cn.counts)
        return cn.counts, torch.zeros(cn.H * cn.W, dtype=torch.float64, device="cuda").index_add_(
            0, cn.pixel, (cn.gid.double() + 1) * (k.double() + 1))
    c0, s0 = sig(L0)
    fr = torch.ones_like(c0, dtype=torch.bool)
    for L in (Lp, Lm):
        c_, s_ = sig(L)
        fr &= (c_ == c0) & (s_ == s0)
    return fr.reshape(L0.depth.shape)


def best_fd(jv_fn, render_pm, eps_list=(1e-3, 3e-4, 1e-4)):
    """min over eps of ||Jv - FD|| / ||FD|| on frozen-branch pixels (per output)."""
    best = {}
    jv = jv_fn()
    for eps in eps_list:
        L0, Lp, Lm = render_pm(eps)
        m = frozen_mask(L0, Lp, Lm)
        for name, (j, fp, fm) in jv(Lp, Lm, eps).items():
            mm = m if j.dim() == 2 else m[None].expand_as(j)
            fd = (fp - fm) / (2 * eps)
            sel = mm & (fd.abs() + j.abs() > 0)
            if bool(sel.any()):
                best[name] = min(best.get(name, float("inf")), rel(j[sel], fd[sel]))
    return best


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from estimator.masked_observation import (MaskedObservationModel, MaskedSequence, area_downsample,
                                              energy_terms, load_object_mask, load_object_masks,
                                              model_from_args, phase1_terms, prepare)
    from estimator.renderer_bridge import build_linearization, embed_pose_to_full_state, extract_pose
    from estimator.sparse_iekf import IEKFConfig, evaluate_true_map_energy, iekf_frame_step, measurement_operators
    from estimator.state import boxplus
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    from tests.test_visibility_constrained_iekf_phase1 import blob, cat, cov, observe

    cams = make_cameras(n_cam=3, H=96, W=96, radius=2.2)
    obj = blob()
    cfg = IEKFConfig(max_iters=8, pcg_max_iters=30)
    model_q = MaskedObservationModel("masked-rgb-expected-quad", (0.5, 0.5, 0.5), "uniform01", cfg.rgb_sigma)
    kappa_exact = 1.0 / (4.0 * cfg.rgb_sigma ** 2)

    # ------------------------------------------------------------------ A
    print("\n[A] prepared RGB and frame/camera association")
    g = torch.Generator(device="cuda").manual_seed(0)
    I = torch.rand(3, 16, 16, device="cuda", generator=g)
    M = torch.rand(16, 16, device="cuda", generator=g) > 0.5
    d = torch.rand(16, 16, device="cuda", generator=g) + 0.5
    ob = prepare(I, d, d > 0, M)
    b = torch.tensor([0.2, 0.7, 0.4], device="cuda")
    Y = ob.target(b)
    check("[A] M=1 pixels: target = recorded RGB; M=0 pixels: target = b exactly",
          torch.equal(Y[:, M], I[:, M]) and torch.equal(Y[:, ~M], b[:, None].expand(3, int((~M).sum()))))
    check("[A] P = M I and V_used = V_orig AND M", torch.equal(ob.rgb_foreground_premult, I * M[None].float())
          and torch.equal(ob.depth_valid, M & (d > 0)))
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        from PIL import Image
        for c in range(2):
            for f in range(4):
                a = np.zeros((8, 8), np.uint8)
                a[f, c] = 255                          # the pixel (frame, camera) marks the file
                (root / f"cam_{c:03d}").mkdir(exist_ok=True)
                Image.fromarray(a).save(root / f"cam_{c:03d}" / f"frame_{f:06d}.png")
        orig = [3, 2, 1, 0]                            # backward processing order
        ms = load_object_masks(root, ["cam_000", "cam_001"], orig, 8, 8)
        ok = all(bool(ms[k][c][orig[k], c]) and int(ms[k][c].sum()) == 1 for k in range(4) for c in range(2))
        check("[A] masks matched by camera and ORIGINAL frame in reverse processing order", ok)
        bad = np.zeros((8, 8), np.uint8); bad[0, 0] = 128
        Image.fromarray(bad).save(root / "nb.png")
        Image.fromarray(np.zeros((9, 8), np.uint8)).save(root / "size.png")
        refused = []
        for name, fn in (("non-binary", lambda: load_object_mask(root / "nb.png", 8, 8)),
                         ("wrong size", lambda: load_object_mask(root / "size.png", 8, 8)),
                         ("missing", lambda: load_object_masks(root, ["cam_000"], [9], 8, 8))):
            try:
                fn(); refused.append(False)
            except SystemExit:
                refused.append(True)
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(root / "cam_000" / "frame_000000.jpg")
        try:
            load_object_masks(root, ["cam_000"], [0], 8, 8); refused.append(False)
        except SystemExit:
            refused.append(True)
        check("[A] non-binary, wrong-size, missing and ambiguous (png+jpg) masks are refused", all(refused),
              str(refused))
    seq = MaskedSequence(model_q, [[torch.ones(4, 4, dtype=torch.bool)] * 2], ["a", "b"], [0])
    try:
        seq.prepare_frame(0, [(torch.zeros(3, 4, 4), torch.zeros(4, 4), torch.zeros(4, 4, dtype=torch.bool))])
        check("[A] observation/mask count mismatch is refused", False)
    except AssertionError:
        check("[A] observation/mask count mismatch is refused", True)

    # ------------------------------------------------------------------ B
    print("\n[B] coverage-correct resizing")
    I2 = torch.tensor([[[1.0, 0.0], [0.2, 0.9]], [[0.3, 0.6], [0.0, 1.0]], [[0.5, 0.5], [0.8, 0.1]]])
    M2 = torch.tensor([[True, False], [True, True]])
    Pr, Mr = area_downsample(I2, M2, 2)
    errs, halo = [], []
    for bb in ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.3, 0.9, 0.2]):
        bt = torch.tensor(bb)
        native = I2 * M2.float()[None] + (1 - M2.float())[None] * bt.reshape(3, 1, 1)
        want = native.mean(dim=(1, 2))
        got = Pr[:, 0, 0] + (1 - Mr[0, 0]) * bt
        naive = I2.mean(dim=(1, 2)) * M2.float().mean() + (1 - M2.float().mean()) * bt
        errs.append(float((got - want).abs().max()))
        halo.append(float((naive - want).abs().max()))
    check("[B] R(M I) + (1 - R(M)) b = area mean of the native composite for every b", max(errs) < 1e-7,
          f"max err {max(errs):.1e}; resizing I and M separately is off by up to {max(halo):.3f}")

    # ------------------------------------------------------------------ C
    print("\n[C] depth outside the mask is never read")
    truth = dataclasses.replace(obj, mu=obj.mu + torch.tensor([0.012, 0.0, 0.0], device="cuda"))
    gt, masks = observe(truth, cams, wall_depth=4.0)       # valid depth OUTSIDE the mask too
    base = prepared_from(gt, masks)
    variants = {}
    for name, fill in (("wall", None), ("zero", 0.0), ("nan", float("nan")), ("1e6", 1e6), ("noise", -1.0)):
        obs_v = []
        for (rgb, dz, v), m in zip(gt, masks):
            d2 = dz.clone()
            if fill is not None:
                d2[~m] = fill if fill != -1.0 else 0.0
                if fill == -1.0:
                    d2[~m] = torch.rand(int((~m).sum()), device="cuda") * 9
            obs_v.append((rgb, d2, v if fill is None else torch.isfinite(d2) & (d2 > 0)))
        variants[name] = prepared_from(obs_v, masks)
    L = build_linearization(obj, cams, background=model_q.background_tensor("cuda"))
    E = {k: energy_terms(*phase1_terms(L.views, v, cfg, model_q)) for k, v in variants.items()}
    check("[C] the energy terms are bitwise identical for every out-of-mask depth value",
          all(E[k] == E["wall"] for k in E), str({k: round(v["depth"], 6) for k, v in E.items()}))
    rh = {k: measurement_operators(L, *phase1_terms(L.views, v, cfg, model_q))[0] for k, v in variants.items()}
    floor = rel(measurement_operators(L, *phase1_terms(L.views, variants["wall"], cfg, model_q))[0], rh["wall"])
    worst = max(rel(rh[k], rh["wall"]) for k in rh)
    check("[C] the RHS is identical up to the float atomic floor", worst <= max(4 * floor, 1e-6),
          f"worst {worst:.1e}, repeat floor {floor:.1e}")
    finite = all(bool(torch.isfinite(r).all()) for r in rh.values())
    check("[C] no NaN reaches the RHS (validity-indexed residual)", finite)
    outs = {}
    for k in ("wall", "nan", "1e6"):
        outs[k] = iekf_frame_step(obj, cov(obj.mu.shape[0]), variants[k], cams, 1 / 30, cfg,
                                  observation_model=model_q)[0].mu
    rep = iekf_frame_step(obj, cov(obj.mu.shape[0]), variants["wall"], cams, 1 / 30, cfg,
                          observation_model=model_q)[0].mu
    fl = float((rep - outs["wall"]).abs().max())
    dif = max(float((outs[k] - outs["wall"]).abs().max()) for k in outs)
    check("[C] the accepted IEKF state is identical up to the CUDA floor", dif <= max(4 * fl, 1e-6),
          f"max |dmu| across fills {dif:.1e}, repeat floor {fl:.1e}")

    # ------------------------------------------------------------------ D
    print("\n[D] compositing on an arbitrary background in the real renderer")
    st = obj
    bD = torch.tensor([0.3, 0.7, 0.1], device="cuda")
    Lb = build_linearization(st, cams, background=bD)
    L0 = build_linearization(st, cams)
    e1 = max(float((vb.rgb - v0.rgb - (1 - v0.alpha)[None] * bD[:, None, None]).abs().max())
             for vb, v0 in zip(Lb.views, L0.views))
    check("[D] C(b) = F + (1 - A) b", e1 < 1e-5, f"max {e1:.1e}")
    gv = torch.Generator(device="cuda").manual_seed(2)
    v = torch.randn(st.mu.shape[0], 6, device="cuda", generator=gv)
    dCb, dC0, dA = Lb.apply_H(v), L0.apply_H(v), L0.apply_H_alpha(v)
    e2 = max(rel(cb[0] - c0[0], -bD[:, None, None] * a[None]) for cb, c0, a in zip(dCb, dC0, dA))
    check("[D] H_C(b) - H_C(0) = -b H_A (renderer JVP vs full-ray alpha JVP)", e2 < 1e-4, f"rel {e2:.1e}")

    def jvD():
        def f(Lp, Lm, eps):
            return {"rgb": (dCb[0][0], Lp.views[0].rgb, Lm.views[0].rgb)}
        return f

    def pmD(eps):
        s_p = boxplus(st, embed_pose_to_full_state(eps * v))
        s_m = boxplus(st, embed_pose_to_full_state(-eps * v))
        return Lb, build_linearization(s_p, cams, background=bD), build_linearization(s_m, cams, background=bD)
    # frozen_mask works per camera: camera 0
    def pm0(eps):
        a, bp, bm = pmD(eps)
        return (type("V", (), {"cache": a.views[0].cache, "depth": a.views[0].depth})(),
                type("V", (), {"cache": bp.views[0].cache, "depth": bp.views[0].depth, "views": bp.views})(),
                type("V", (), {"cache": bm.views[0].cache, "depth": bm.views[0].depth, "views": bm.views})())
    fdD = best_fd(jvD, pm0)
    check("[D] RGB JVP vs central FD with b != 0 (translation + rotation)", fdD.get("rgb", 1) < 3e-2,
          f"rel {fdD.get('rgb', float('nan')):.2e}")

    # ------------------------------------------------------------------ E
    print("\n[E] exact background expectation vs the 8-point cubature")
    # anisotropic, randomly oriented splats, so rotations change the render
    from estimator.manifold import so3_exp as _so3
    ga = torch.Generator(device="cuda").manual_seed(13)
    st_e = dataclasses.replace(
        obj, mu=obj.mu + torch.tensor([-0.02, 0.01, 0.0], device="cuda"),
        scales=(0.012 + 0.05 * torch.rand(obj.mu.shape[0], 3, device="cuda", generator=ga)).contiguous(),
        R=_so3(torch.randn(obj.mu.shape[0], 3, generator=torch.Generator().manual_seed(3)).double()).float().cuda())
    gt_e, m_e = observe(truth, cams)
    pe = prepared_from(gt_e, m_e)
    h = 1.0 / math.sqrt(12.0)
    corners = [torch.tensor([0.5 + sx * h, 0.5 + sy * h, 0.5 + sz * h], device="cuda")
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    w_diag = torch.tensor([60.0, 100.0, 140.0], device="cuda")         # per-channel W
    Lbar = build_linearization(st_e, cams, background=torch.full((3,), 0.5, device="cuda"))
    kap = w_diag.sum() / 12.0
    Ec, Eq = 0.0, 0.0
    for vbar, ob in zip(Lbar.views, pe):
        ebar = vbar.rgb - ob.target(torch.full((3,), 0.5, device="cuda"))
        Ec += 0.5 * float((w_diag[:, None, None] * ebar.double() ** 2).sum()) \
            + 0.5 * float(kap) * float(((vbar.alpha - ob.object_coverage).double() ** 2).sum())
    Ls = [build_linearization(st_e, cams, background=bc) for bc in corners]
    for Lc, bc in zip(Ls, corners):
        for vc, ob in zip(Lc.views, pe):
            e = vc.rgb - ob.target(bc)
            Eq += 0.5 * float((w_diag[:, None, None] * e.double() ** 2).sum()) / 8
    check("[E] closed-form energy = cubature mean (diagonal W)", abs(Ec - Eq) / Eq < 1e-6,
          f"{Ec:.6f} vs {Eq:.6f}")
    Wf = torch.tensor([[80.0, 20.0, -10.0], [20.0, 120.0, 15.0], [-10.0, 15.0, 90.0]], device="cuda", dtype=torch.float64)
    Ec2, Eq2 = 0.0, 0.0
    for vbar, ob in zip(Lbar.views, pe):
        ebar = (vbar.rgb - ob.target(torch.full((3,), 0.5, device="cuda"))).double().reshape(3, -1)
        Ec2 += 0.5 * float((ebar * (Wf @ ebar)).sum()) + 0.5 * float(torch.trace(Wf) / 12) * float(
            ((vbar.alpha - ob.object_coverage).double() ** 2).sum())
    for Lc, bc in zip(Ls, corners):
        for vc, ob in zip(Lc.views, pe):
            e = (vc.rgb - ob.target(bc)).double().reshape(3, -1)
            Eq2 += 0.5 * float((e * (Wf @ e)).sum()) / 8
    check("[E] closed-form energy = cubature mean (cross-channel W, kappa = trace(W)/12)",
          abs(Ec2 - Eq2) / Eq2 < 1e-6, f"{Ec2:.6f} vs {Eq2:.6f}")
    zero_d = [torch.zeros_like(vv.depth) for vv in Lbar.views]
    prec_bar = [(w_diag[:, None, None].expand_as(vv.rgb).contiguous(), zd,
                 ob.target(torch.full((3,), 0.5, device="cuda")) - vv.rgb, zd)
                for vv, ob, zd in zip(Lbar.views, pe, zero_d)]
    alpha_bar = [(torch.full_like(vv.alpha, float(kap)), ob.object_coverage - vv.alpha) for vv, ob in zip(Lbar.views, pe)]
    rhs_c, normal_c, _ = measurement_operators(Lbar, prec_bar, alpha_bar)
    rhs_q = None
    pv = torch.randn(st_e.mu.shape[0], 6, device="cuda", generator=gv)
    nq = None
    for Lc, bc in zip(Ls, corners):
        pr = [(w_diag[:, None, None].expand_as(vv.rgb).contiguous(), torch.zeros_like(vv.depth),
               ob.target(bc) - vv.rgb, torch.zeros_like(vv.depth)) for vv, ob in zip(Lc.views, pe)]
        r_, n_, _ = measurement_operators(Lc, pr, None)
        rhs_q = r_ / 8 if rhs_q is None else rhs_q + r_ / 8
        nq = n_(pv) / 8 if nq is None else nq + n_(pv) / 8
    check("[E] closed-form gradient = cubature mean of the per-background gradients", rel(rhs_c, rhs_q) < 1e-4,
          f"rel {rel(rhs_c, rhs_q):.1e}")
    check("[E] H_C(b_bar)^T W H_C(b_bar) + H_A^T K H_A = cubature mean of the normal operators",
          rel(normal_c(pv), nq) < 1e-4, f"rel {rel(normal_c(pv), nq):.1e}")

    # ------------------------------------------------------------------ F
    print("\n[F] accumulated-alpha operators and the alpha Fisher kernel")
    from diff_gaussian_rasterization import iekf_visible_alpha
    Lf = build_linearization(st_e, cams, background=torch.full((3,), 0.5, device="cuda"))
    vf = Lf.views[0]
    one = torch.ones_like(vf.alpha, dtype=torch.int32)
    Av, Tf, nc, _f, _b = iekf_visible_alpha(vf.cache, one, torch.zeros_like(vf.alpha), 0.0)
    check("[F] A = 1 - T_end and the full-ray alpha equals v.alpha, T and n_contrib",
          float((vf.alpha - (1 - vf.transmittance)).abs().max()) < 1e-6 and float((Av - vf.alpha).abs().max()) < 1e-6
          and float((Tf - vf.transmittance).abs().max()) < 1e-6 and torch.equal(nc.int(), vf.n_contrib.int()),
          f"|A - (1-T)| {float((vf.alpha - (1 - vf.transmittance)).abs().max()):.1e}, |A_vis - A| {float((Av - vf.alpha).abs().max()):.1e}")
    for nm, mask6 in (("translation", torch.tensor([1, 1, 1, 0, 0, 0.0])), ("rotation", torch.tensor([0, 0, 0, 1, 1, 1.0]))):
        vv = torch.randn(st_e.mu.shape[0], 6, device="cuda", generator=gv) * mask6.cuda()
        dA0 = Lf.apply_H_alpha(vv)[0]

        def jvF(dA0=dA0):
            def f(Lp, Lm, eps):
                return {"A": (dA0, Lp.views[0].alpha, Lm.views[0].alpha)}
            return f

        def pmF(eps, vv=vv):
            sp = build_linearization(boxplus(st_e, embed_pose_to_full_state(eps * vv)), cams[:1],
                                     background=torch.full((3,), 0.5, device="cuda"))
            sm = build_linearization(boxplus(st_e, embed_pose_to_full_state(-eps * vv)), cams[:1],
                                     background=torch.full((3,), 0.5, device="cuda"))
            o = type("V", (), {"cache": vf.cache, "depth": vf.depth})()
            p_ = type("V", (), {"cache": sp.views[0].cache, "depth": sp.views[0].depth, "views": sp.views})()
            m_ = type("V", (), {"cache": sm.views[0].cache, "depth": sm.views[0].depth, "views": sm.views})()
            return o, p_, m_
        r_ = best_fd(jvF, pmF)
        check(f"[F] dA/dx vs central FD ({nm})", r_.get("A", 1) < 3e-2, f"rel {r_.get('A', float('nan')):.2e}")
    za = [torch.randn(vw.alpha.shape, device="cuda", generator=gv) for vw in Lf.views]
    vv = torch.randn(st_e.mu.shape[0], 6, device="cuda", generator=gv)
    lhs = sum(float((a.double() * z.double()).sum()) for a, z in zip(Lf.apply_H_alpha(vv), za))
    rhs_ = float((vv.double() * Lf.apply_HT_alpha(za).double()).sum())
    check("[F] <H_A v, z> = <v, H_A^T z>", abs(lhs - rhs_) / max(abs(lhs), 1e-30) < 1e-4,
          f"rel gap {abs(lhs - rhs_) / max(abs(lhs), 1e-30):.1e}")
    wts = [torch.rand(vw.alpha.shape, device="cuda", generator=gv) * 30 for vw in Lf.views]
    Fk = Lf.build_alpha_fisher(wts)
    worst, psd = 0.0, True
    for i in (0, 7, 23, 39):
        cols = []
        for k in range(6):
            e = torch.zeros(st_e.mu.shape[0], 6, device="cuda"); e[i, k] = 1.0
            cols.append(Lf.apply_H_alpha(e))
        Fi = torch.zeros(6, 6, dtype=torch.float64, device="cuda")
        for c in range(len(cams)):
            J = torch.stack([cols[k][c].reshape(-1).double() for k in range(6)], 1)
            Fi += J.T @ (wts[c].reshape(-1, 1).double() * J)
        worst = max(worst, rel(Fk[i].double(), Fi))
        psd &= float(torch.linalg.eigvalsh(Fk[i].double()).min()) > -1e-6 * float(Fi.abs().max())
    check("[F] alpha Fisher block = explicit J^T W J (new kernel) and PSD", worst < 1e-4 and psd, f"max rel {worst:.1e}")

    # ------------------------------------------------------------------ G
    print("\n[G] stage lifts: (C, D, A) JVP vs FD of the whole stage-to-render map")
    from estimator.supernodes import GroupedObservation, build_supernodes
    bbar = torch.full((3,), 0.5, device="cuda")
    part = build_supernodes(st_e.mu, target_size=8)
    grp = GroupedObservation(None, part)
    vg = torch.randn(part.num_supernodes, 6, device="cuda", generator=gv)
    pose_g = grp.expand(vg)
    LG = build_linearization(st_e, cams[:1], background=bbar)
    (dCg, dDg), dAg = LG.apply_H(pose_g)[0], LG.apply_H_alpha(pose_g)[0]

    def jvG():
        def f(Lp, Lm, eps):
            return {"rgb": (dCg, Lp.views[0].rgb, Lm.views[0].rgb), "depth": (dDg, Lp.views[0].depth, Lm.views[0].depth),
                    "alpha": (dAg, Lp.views[0].alpha, Lm.views[0].alpha)}
        return f

    def pmG(eps):
        sp = build_linearization(boxplus(st_e, embed_pose_to_full_state(eps * pose_g)), cams[:1], background=bbar)
        sm = build_linearization(boxplus(st_e, embed_pose_to_full_state(-eps * pose_g)), cams[:1], background=bbar)
        o = type("V", (), {"cache": LG.views[0].cache, "depth": LG.views[0].depth})()
        return (o, type("V", (), {"cache": sp.views[0].cache, "depth": sp.views[0].depth, "views": sp.views})(),
                type("V", (), {"cache": sm.views[0].cache, "depth": sm.views[0].depth, "views": sm.views})())
    rG = best_fd(jvG, pmG)
    check("[G] Phase 1 supernode lift: RGB, depth, alpha JVP = FD", max(rG.values()) < 3e-2,
          " ".join(f"{k} {v:.1e}" for k, v in rG.items()))
    from estimator.joint_process import apply_correction
    from estimator.joint_renderer_bridge import joint_apply_H, joint_apply_HT
    from estimator.phase2_shared import render_state
    from estimator.rigid_part_runtime import RigidIEKFConfig, RigidState
    from estimator.rigid_part_se3 import RigidPart
    from estimator.manifold import so3_exp
    from tests.test_visibility_constrained_iekf_phase23 import scene as scene23
    st2, members = scene23()
    mu0, R0 = st2.mu.double(), st2.R.double()
    rs = RigidState(RigidPart(members, num_frames=2), RigidIEKFConfig(max_iters=6))
    rs.parts[0].R[1] = so3_exp(torch.tensor([[0.0, 0.0, 0.07]], dtype=torch.float64))[0]

    def lin2(s_, t):
        mu, Rg = s_.gaussians(t, mu0, R0)
        return render_state(st2, mu, Rg, cams[:1], cfg, build_linearization, background=bbar)
    from estimator import phase2_prismatic as mp
    from estimator.joint_displacement import DisplacementOnlyState
    from estimator.phase2_model_comparison import prismatic_state
    ppart = mp.PrismaticPart(members, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
                             torch.tensor([0.0, 0.004], dtype=torch.float64))
    ds = DisplacementOnlyState(prismatic_state(ppart))
    for nm, state in (("Phase 2 rigid SE(3)", rs), ("Phase 3 fixed-joint displacement", ds)):
        dv = torch.randn(state.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        L_ = lin2(state, 1)
        dC_, dD_, dA_ = joint_apply_H(L_, state, 1, dv.cuda(), mu0, R0, alpha=True)[0]

        def jv2(dC_=dC_, dD_=dD_, dA_=dA_):
            def f(Lp, Lm, eps):
                return {"rgb": (dC_, Lp.views[0].rgb, Lm.views[0].rgb), "depth": (dD_, Lp.views[0].depth, Lm.views[0].depth),
                        "alpha": (dA_, Lp.views[0].alpha, Lm.views[0].alpha)}
            return f

        def pm2(eps, state=state, dv=dv, L_=L_):
            sp = apply_correction(copy.deepcopy(state), 1, eps * dv, transport=False)[0]
            sm = apply_correction(copy.deepcopy(state), 1, -eps * dv, transport=False)[0]
            Lp, Lm = lin2(sp, 1), lin2(sm, 1)
            o = type("V", (), {"cache": L_.views[0].cache, "depth": L_.views[0].depth})()
            return (o, type("V", (), {"cache": Lp.views[0].cache, "depth": Lp.views[0].depth, "views": Lp.views})(),
                    type("V", (), {"cache": Lm.views[0].cache, "depth": Lm.views[0].depth, "views": Lm.views})())
        r2 = best_fd(jv2, pm2)
        check(f"[G] {nm} lift: RGB, depth, alpha JVP = FD", max(r2.values()) < 3e-2,
              " ".join(f"{k} {v:.1e}" for k, v in r2.items()))
        za_ = [(torch.randn_like(L_.views[0].rgb), torch.randn_like(L_.views[0].depth), torch.randn_like(L_.views[0].alpha))]
        lhs = float((dC_.double() * za_[0][0].double()).sum() + (dD_.double() * za_[0][1].double()).sum()
                    + (dA_.double() * za_[0][2].double()).sum())
        rhs2 = float((dv.double() * joint_apply_HT(L_, state, 1, za_, mu0, R0).double().cpu()).sum())
        check(f"[G] {nm}: <H v, z> = <v, H^T z> with the alpha row", abs(lhs - rhs2) / max(abs(lhs), 1e-30) < 1e-4,
              f"rel gap {abs(lhs - rhs2) / max(abs(lhs), 1e-30):.1e}")

    # ------------------------------------------------------------------ H
    print("\n[H] matrix-free normal operator vs explicit Jacobian columns (tiny scene)")
    tiny = blob(n=6, seed=11)
    cams_t = make_cameras(n_cam=2, H=32, W=32, radius=2.2)
    gt_t, m_t = observe(dataclasses.replace(tiny, mu=tiny.mu + 0.01), cams_t)
    pt = prepared_from(gt_t, m_t)
    Lt = build_linearization(tiny, cams_t, background=bbar)
    prec_t, alpha_t = phase1_terms(Lt.views, pt, cfg, model_q)
    rhs_t, normal_t, fisher_t = measurement_operators(Lt, prec_t, alpha_t)
    P6 = tiny.mu.shape[0] * 6
    cols = []
    for j in range(P6):
        e = torch.zeros(P6, device="cuda"); e[j] = 1.0
        e = e.reshape(-1, 6)
        hc = Lt.apply_H(e); ha = Lt.apply_H_alpha(e)
        cols.append(torch.cat([torch.cat([c.reshape(-1), d_.reshape(-1), a.reshape(-1)]) for (c, d_), a in zip(hc, ha)]).double())
    J = torch.stack(cols, 1)
    wv = torch.cat([torch.cat([p.reshape(-1), pd.reshape(-1), k.reshape(-1)])
                    for (p, pd, _, _), (k, _) in zip(prec_t, alpha_t)]).double()
    rv = torch.cat([torch.cat([rc.reshape(-1), rd.reshape(-1), ra.reshape(-1)])
                    for (_, _, rc, rd), (_, ra) in zip(prec_t, alpha_t)]).double()
    Nd = J.T @ (wv[:, None] * J)
    bd = J.T @ (wv * rv)
    x = torch.randn(P6, device="cuda", generator=gv, dtype=torch.float64)
    nmv = normal_t(x.float().reshape(-1, 6)).reshape(-1).double()
    check("[H] normal(v) = (H_C^T W H_C + H_A^T K H_A + H_D^T W_D H_D) v", rel(nmv, Nd @ x) < 1e-4,
          f"rel {rel(nmv, Nd @ x):.1e}")
    check("[H] RHS = H_C^T W r_C + H_A^T K r_A + H_D^T W_D r_D", rel(rhs_t.reshape(-1), bd) < 1e-4,
          f"rel {rel(rhs_t.reshape(-1), bd):.1e}")
    Fd = torch.stack([Nd[6 * i:6 * i + 6, 6 * i:6 * i + 6] for i in range(tiny.mu.shape[0])])
    check("[H] Fisher = the 6x6 diagonal blocks of the same normal matrix (alpha included)",
          rel(fisher_t().double(), Fd) < 1e-4, f"rel {rel(fisher_t().double(), Fd):.1e}")
    check("[H] kappa = 1/(4 sigma_rgb^2) on every pixel", all(torch.allclose(k, torch.full_like(k, kappa_exact)) for k, _ in alpha_t),
          f"kappa {kappa_exact}")

    # ------------------------------------------------------------------ I
    print("\n[I] acceptance on the exact objective")
    red = blob(n=1, seed=21, centre=(0.45, 0.0, 0.0), colour=(1.0, 0.0, 0.0))
    far = blob(n=1, seed=22, centre=(0.0, 30.0, 0.0), colour=(0.5, 0.5, 0.5))   # outside every frustum
    far = dataclasses.replace(far, scales=torch.full((1, 3), 0.09, device="cuda"))
    c0 = cat(obj, red, far)
    cams_i = cams[:1]                       # the grey blob sits in front of the red stray for camera 0
    gt_i, m_i = observe(obj, cams_i)
    pi = prepared_from(gt_i, m_i)
    cam_dir = -torch.tensor(cams[0].viewmatrix[:3, 2].tolist(), device="cuda")
    c1 = dataclasses.replace(c0, mu=c0.mu.clone())
    c1.mu[-1] = red.mu[0] + 0.25 * cam_dir / cam_dir.norm()        # a big grey blob in front of the red stray
    LP = zero_prior(c0)
    e0, p0, _ = evaluate_true_map_energy(c0, c0, LP, pi, cams_i, cfg, model_q.background_tensor("cuda"), None, model_q)
    e1, p1, _ = evaluate_true_map_energy(c1, c0, LP, pi, cams_i, cfg, model_q.background_tensor("cuda"), None, model_q)
    check("[I] the candidate lowers the RGB energy but raises the exact total", p1["rgb_energy"] < p0["rgb_energy"] and e1 > e0,
          f"RGB {p0['rgb_energy']:.1f} -> {p1['rgb_energy']:.1f}, alpha {p0['alpha_energy']:.1f} -> {p1['alpha_energy']:.1f}, "
          f"total {e0:.1f} -> {e1:.1f}")
    from estimator.trust_region import TrustRegionConfig, adapt_radius
    _r, acc = adapt_radius(1.0, (e0 - e1) / 1.0, False, e0 - e1, TrustRegionConfig())
    check("[I] the trust-region acceptance rejects it", not acc)
    e0b, _, _ = evaluate_true_map_energy(c0, c0, LP, pi, cams_i, cfg, model_q.background_tensor("cuda"), None, model_q)
    check("[I] the objective at a fixed state is reproducible (RGB precision fixed)", e0b == e0, f"{e0} vs {e0b}")
    _s, _P, dg = iekf_frame_step(obj, cov(obj.mu.shape[0]), prepared_from(*observe(truth, cams)), cams, 1 / 30, cfg,
                                 observation_model=model_q)
    rows = [r for r in dg.rows if r.get("event") in ("initial", "accepted", "rejected")]
    held = all(rows[k]["energy"] == rows[k - 1]["energy"] for k in range(1, len(rows)) if rows[k]["event"] == "rejected")
    check("[I] rejected trials leave the evaluated objective unchanged", held,
          f"{sum(r['event'] == 'accepted' for r in rows)} accepted, {sum(r['event'] == 'rejected' for r in rows)} rejected")

    # ------------------------------------------------------------------ K
    print("\n[K] no QP, no constraint, no silhouette, one render per camera per evaluation")
    import estimator.constrained_qp as cq
    import estimator.visibility_constraint as vc
    import estimator.renderer_bridge as rb
    import estimator.sparse_iekf as si
    import diff_gaussian_rasterization as dgr
    calls = {"render": 0, "energy": 0}
    saved = {}

    def forbid(name):
        def f(*a, **k):
            raise AssertionError(f"masked mode called {name}")
        return f
    for mod, name in ((cq, "solve_qp_projection"), (cq, "solve_qp_alm"), (vc, "constrained_step"),
                      (vc, "classify_persistent"), (dgr, "iekf_visible_alpha")):
        saved[(mod, name)] = getattr(mod, name)
        setattr(mod, name, forbid(f"{mod.__name__}.{name}"))
    orig_render, orig_energy = rb.iekf_render, si.evaluate_true_map_energy

    def count_render(*a, **k):
        calls["render"] += 1
        return orig_render(*a, **k)

    def count_energy(*a, **k):
        calls["energy"] += 1
        return orig_energy(*a, **k)
    sil_before = "estimator.silhouette" in sys.modules
    obs_k = prepared_from(*observe(truth, cams))          # rendered BEFORE the counters are installed
    rb.iekf_render, si.evaluate_true_map_energy = count_render, count_energy
    try:
        iekf_frame_step(obj, cov(obj.mu.shape[0]), obs_k, cams, 1 / 30, cfg,
                        observation_model=model_q)
        ran = True
    except AssertionError as ex:
        ran = False
        print("   ", ex)
    finally:
        for (mod, name), f in saved.items():
            setattr(mod, name, f)
        rb.iekf_render, si.evaluate_true_map_energy = orig_render, orig_energy
    check("[K] the masked IEKF frame never calls the QP, the constraint step or the visible-alpha forward", ran)
    check("[K] exactly one render per camera per nonlinear evaluation", calls["render"] == len(cams) * calls["energy"],
          f"{calls['render']} renders, {calls['energy']} evaluations x {len(cams)} cameras")
    check("[K] estimator.silhouette is not imported", sil_before or "estimator.silhouette" not in sys.modules)

    # ------------------------------------------------------------------ M
    print("\n[M] conflicting / incomplete flags")
    import argparse
    bad = []
    for kw in ({"observation_mode": "masked-rgb-expected-quad", "object_mask_constraint": "visible-negative-hard"},
               {"observation_mode": "masked-rgb-constant"},
               {"observation_mode": "masked-rgb-expected-quad", "rgb_background_constant": [0.1, 0.1, 0.1]},
               {"observation_mode": "original", "rgb_background_distribution": "uniform01"}):
        ns = argparse.Namespace(**{"object_mask_constraint": "off", "rgb_background_constant": None,
                                   "rgb_background_distribution": None, **kw})
        try:
            model_from_args(ns, cfg); bad.append(False)
        except SystemExit:
            bad.append(True)
    check("[M] hard-constraint + masked mode, missing/misplaced background options are refused", all(bad), str(bad))
    try:
        iekf_frame_step(obj, cov(obj.mu.shape[0]), pi, cams_i, 1 / 30, cfg, observation_model=model_q,
                        visibility=object())
        check("[M] iekf_frame_step refuses observation_model together with visibility", False)
    except ValueError:
        check("[M] iekf_frame_step refuses observation_model together with visibility", True)

    # ------------------------------------------------------------------ N
    print("\n[N] identifiability: an isolated Gaussian in a uniform negative region")
    stray = blob(n=1, seed=31, centre=(0.0, 0.62, 0.0), colour=(0.5, 0.5, 0.5))
    sN = cat(obj, stray)
    gt_n, m_n = observe(obj, cams[:1])
    pn = prepared_from(gt_n, m_n)
    LN = build_linearization(sN, cams[:1], background=bbar)
    prec_n, alpha_n = phase1_terms(LN.views, pn, cfg, model_q)
    terms = energy_terms(prec_n, alpha_n)
    rhs_n = measurement_operators(LN, prec_n, alpha_n)[0][-1]
    Rc = torch.tensor(cams[0].viewmatrix[:3, :3].tolist(), device="cuda").T       # rows = camera axes in world
    g_cam = Rc @ rhs_n[:3]
    inplane, along = float(g_cam[:2].norm()), float(g_cam[2].abs())
    rgb_part = float(LN.apply_HT([(p * r, torch.zeros_like(pd)) for p, pd, r, _ in prec_n])[-1, :3].norm())
    check("[N] the stray carries a nonzero background cost", terms["alpha"] > 0, f"alpha energy {terms['alpha']:.2f}")
    check("[N] the stray gets no RGB gradient (camouflaged against the mean background)",
          rgb_part < 1e-6 * max(along, 1e-12), f"RGB part {rgb_part:.1e}")
    from estimator.masked_observation import PreparedRGBD
    no_mask = [PreparedRGBD(torch.zeros_like(o.rgb_foreground_premult), torch.zeros_like(o.object_coverage),
                            o.depth_observed, torch.zeros_like(o.depth_valid)) for o in pn]
    rhs_free = measurement_operators(LN, *phase1_terms(LN.views, no_mask, cfg, model_q))[0][-1]
    check("[N] its gradient is identical with the object mask removed: no attraction toward the mask",
          rel(rhs_free, rhs_n) < 1e-5,
          f"rel {rel(rhs_free, rhs_n):.1e}; |g| along the ray {along:.2e} (shrink the footprint), in the image "
          f"plane {inplane:.2e} (perspective: the footprint size depends on the off-axis position)")

    # ------------------------------------------------------------------ P
    print("\n[P] end to end on the toy scenes")
    g7 = torch.Generator(device="cuda").manual_seed(7)
    tex = [torch.rand(3, 96, 96, device="cuda", generator=g7) for _ in cams]
    gt_p, m_p = observe(truth, cams, texture=tex)
    err = lambda s: float((s.mu - truth.mu).norm(dim=1).mean())          # noqa: E731
    out_m, _, dm = iekf_frame_step(obj, cov(obj.mu.shape[0]), prepared_from(gt_p, m_p), cams, 1 / 30, cfg,
                                   observation_model=model_q)
    out_o, _, _ = iekf_frame_step(obj, cov(obj.mu.shape[0]), gt_p, cams, 1 / 30, cfg)
    leak = lambda s: sum(float(v.alpha[~m].sum()) for v, m in zip(build_linearization(s, cams).views, m_p))  # noqa: E731
    check("[P] Phase 1 masked frame moves the object toward the observation, without background leakage",
          err(out_m) < err(obj) and leak(out_m) < leak(out_o),
          f"centre error {err(obj):.4f} -> masked {err(out_m):.4f} (original {err(out_o):.4f}); alpha on negative "
          f"pixels {leak(obj):.1f} -> masked {leak(out_m):.1f} (original {leak(out_o):.1f})")
    from estimator.phase2_shared import effective_precision_masked
    from estimator.rigid_part_runtime import run_rigid_window
    lid = members.cuda()
    Rm = so3_exp(torch.tensor([[0.0, 0.0, 0.15]], dtype=torch.float64))[0].float().cuda()
    pivot = torch.tensor([0.02, 0.0, 0.0], device="cuda")
    mu1 = st2.mu.clone(); mu1[lid] = (st2.mu[lid] - pivot) @ Rm.T + pivot
    R1 = st2.R.clone(); R1[lid] = Rm @ st2.R[lid]
    t1 = dataclasses.replace(st2, mu=mu1, R=R1)
    obs_w = []
    for tr in (st2, t1):
        gw, mw = observe(tr, cams, texture=tex)
        obs_w.append(prepared_from(gw, mw))
    ctr = {"t": 0}

    def linw(s_, t):
        ctr["t"] = t
        mu, Rg = s_.gaussians(t, mu0, R0)
        return render_state(st2, mu, Rg, cams, cfg, build_linearization, background=bbar)

    def precw(Lx):
        return effective_precision_masked(Lx, obs_w[ctr["t"]], list(range(len(cams))), cfg, None)
    outw = run_rigid_window(RigidState(RigidPart(members, num_frames=2), RigidIEKFConfig(max_iters=6)), mu0, R0,
                            obs_w, linw, precw, cfg=RigidIEKFConfig(max_iters=6), observation_model=model_q)
    ang = float(torch.arccos(((torch.diagonal(outw["part"].R[1]).sum() - 1) / 2).clamp(-1, 1)))
    r1 = outw["records"][1]
    check("[P] Phase 2 masked rigid window: the part rotates toward the observation (true 0.15 rad)",
          0.05 < ang < 0.25 and r1.energy_after < r1.energy_before,
          f"angle {ang:.3f} rad, energy {r1.energy_before:.1f} -> {r1.energy_after:.1f}")
    from estimator.joint_displacement import geometry_snapshot, geometry_unchanged
    from estimator.joint_iekf import JointIEKFConfig
    from estimator.phase2_model_comparison import fit_model
    tp = dataclasses.replace(st2, mu=torch.where(torch.isin(torch.arange(st2.mu.shape[0], device="cuda"), lid)[:, None],
                                                 st2.mu + torch.tensor([0.015, 0.0, 0.0], device="cuda"), st2.mu))
    obs_w = []
    for tr in (st2, tp):
        gw, mw = observe(tr, cams, texture=tex)
        obs_w.append(prepared_from(gw, mw))
    ds2 = DisplacementOnlyState(prismatic_state(mp.PrismaticPart(members, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
                                                                 torch.tensor([0.0, 0.004], dtype=torch.float64))))
    snap = geometry_snapshot(ds2.parts[0])
    fit = fit_model(ds2, mu0, obs_w, linw, precw, JointIEKFConfig(max_iters=6), label="prismatic",
                    observation_model=model_q)
    s1 = float(ds2.parts[0].s[1])
    check("[P] Phase 3 masked fixed-joint window: displacement toward 0.015 m, geometry bitwise unchanged",
          abs(s1 - 0.015) < abs(0.004 - 0.015) and geometry_unchanged(snap, ds2.parts[0]),
          f"s 0.004 -> {s1:.4f} (true 0.015), energy {fit['records'][1].energy_before:.1f} -> {fit['records'][1].energy_after:.1f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
