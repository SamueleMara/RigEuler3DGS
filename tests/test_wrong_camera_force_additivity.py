#!/usr/bin/env python
"""Wrong-camera audit, tests 2, 3, 8, 9: per-camera x per-channel decomposition of the
production masked operator (RGB, depth, alpha) on a toy CUDA scene.

  additivity   sum_{c,ch} b == production rhs_pose; sum_{c,ch} A(v) == production normal(v)
  adjoint      <H_c v, z> == <v, H_c^T z> per camera for the RGB-D and alpha operators
  FD           H_c v == central FD of the camera's render (best step of a ladder)
  symmetry/PSD <v, A_c w> == <A_c v, w>,  v^T A_c v >= 0 per camera and channel
  mask oracle  a camera with no valid depth has exactly zero b_depth and A_depth, while its
               masked RGB / alpha terms continue

    python tests/test_wrong_camera_force_additivity.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tests.wrong_camera_fixtures import check_factory, finish, frozen, masked_scene  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from estimator import wrong_camera_audit as wa
    from estimator.renderer_bridge import build_linearization, embed_pose_to_full_state
    from estimator.state import boxplus
    check, st_ = check_factory()
    cams, st, truth, obs, gt, masks, model, cfg = masked_scene()
    lin, terms, prec, alpha_terms, rhs, normal = frozen(st, cams, obs, model, cfg)

    print("\n[additivity / symmetry / PSD]")
    ck, b = wa.additivity_and_symmetry(lin, terms, rhs, normal, n_probe=3)
    check("sum_c,ch b == production rhs_pose", ck["rhs_additivity_rel_err"] < 1e-5, f"{ck['rhs_additivity_rel_err']:.1e}")
    check("sum_c,ch A(v) == production normal(v)", ck["normal_additivity_rel_err_max"] < 1e-5,
          f"{ck['normal_additivity_rel_err_max']:.1e}")
    check("alpha channel present and nonzero", float(sum(b[(c, 'alpha')].norm() for c in range(3))) > 0)
    check("<v, A_c w> == <A_c v, w> for every camera and channel", max(ck["symmetry_rel_err"].values()) < 1e-4,
          f"max {max(ck['symmetry_rel_err'].values()):.1e}")
    check("v^T A_c v >= 0 for every camera and channel", min(ck["min_vAv_over_norms"].values()) > -1e-6,
          f"min {min(ck['min_vAv_over_norms'].values()):.1e}")

    print("\n[adjoint per camera]")
    g = torch.Generator(device="cpu").manual_seed(3)
    n = st.mu.shape[0]
    worst = 0.0
    for c, v in enumerate(lin.views):
        dv = torch.randn((n, 6), generator=g).cuda()
        zr = torch.randn(v.rgb.shape, generator=g).cuda()
        zd = torch.randn(v.depth.shape, generator=g).cuda()
        za = torch.randn(v.alpha.shape, generator=g).cuda()
        dC, dD = wa.view_H(v, dv)
        lhs = float((dC.double() * zr.double()).sum() + (dD.double() * zd.double()).sum())
        rhs_ = float((dv.double() * wa.view_HT(v, zr, zd).double()).sum())
        dA = wa.view_H_alpha(v, dv)
        lhs_a = float((dA.double() * za.double()).sum())
        rhs_a = float((dv.double() * wa.view_HT_alpha(v, za).double()).sum())
        worst = max(worst, abs(lhs - rhs_) / max(abs(lhs), 1e-30), abs(lhs_a - rhs_a) / max(abs(lhs_a), 1e-30))
    check("<H_c v, z> == <v, H_c^T z> (RGB-D and alpha, every camera)", worst < 1e-4, f"{worst:.1e}")

    print("\n[per-view operator == production per-camera operator; FD sanity]")
    dv = torch.randn((n, 6), generator=g).cuda()
    prod = lin.apply_H(dv)
    same = all(torch.equal(wa.view_H(v, dv)[0], prod[c][0]) and torch.equal(wa.view_H(v, dv)[1], prod[c][1])
               for c, v in enumerate(lin.views))
    check("view_H(c) is bit-identical to the production FrameLinearization.apply_H camera c", same)
    # FD on a 48x48 toy crosses alpha-skip / tile boundaries at large steps and float32
    # round-off at tiny ones: the best step of a ladder is reported, per motion type
    for part, sl in (("translation", slice(0, 3)), ("rotation", slice(3, 6))):
        base = torch.zeros((n, 6), dtype=torch.float64)
        base[:, sl] = torch.randn((n, 3), generator=g, dtype=torch.float64)
        best = float("inf")
        for eps in (1e-3, 1e-4, 1e-5, 1e-6):
            d = (base * eps).float().cuda()
            f = embed_pose_to_full_state(d)
            Lp = build_linearization(boxplus(st, f), cams, eps_depth=cfg.eps_depth, background=model.background_tensor("cuda"))
            Lm = build_linearization(boxplus(st, -f), cams, eps_depth=cfg.eps_depth, background=model.background_tensor("cuda"))
            e = max(float((wa.view_H(v, d)[0] - 0.5 * (Lp.views[c].rgb - Lm.views[c].rgb)).norm()
                          / (0.5 * (Lp.views[c].rgb - Lm.views[c].rgb)).norm().clamp_min(1e-30))
                    for c, v in enumerate(lin.views))
            best = min(best, e)
        check(f"H_c v == central FD of the render along a random {part} (best of the step ladder)", best < 5e-2,
              f"best rel {best:.1e}")

    print("\n[mask / invalid-depth oracle]")
    obs2 = list(obs)
    obs2[0] = dataclasses.replace(obs[0], depth_valid=torch.zeros_like(obs[0].depth_valid),
                                  depth_observed=torch.full_like(obs[0].depth_observed, 1e6))
    lin2, terms2, _p, _a, rhs2, normal2 = frozen(st, cams, obs2, model, cfg)
    b2 = wa.camera_channel_rhs(lin2, terms2)
    v = torch.randn((n, 6), generator=g).cuda()
    A2 = wa.camera_channel_normal(lin2, terms2, v)
    check("camera without valid depth: b_depth == 0 and A_depth(v) == 0 exactly",
          float(b2[(0, "depth")].abs().max()) == 0.0 and float(A2[(0, "depth")].abs().max()) == 0.0)
    check("...while its masked RGB and alpha terms continue",
          float(b2[(0, "rgb")].norm()) > 0 and float(b2[(0, "alpha")].norm()) > 0)
    ck2, _ = wa.additivity_and_symmetry(lin2, terms2, rhs2, normal2, n_probe=1)
    check("additivity still exact", ck2["rhs_additivity_rel_err"] < 1e-5 and ck2["normal_additivity_rel_err_max"] < 1e-5)
    return finish(st_)


if __name__ == "__main__":
    raise SystemExit(main())
