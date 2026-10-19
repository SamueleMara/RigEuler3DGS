#!/usr/bin/env python
"""Phase 1 free-Gaussian IEKF with the hard negative-ray constraint (toy scenes, real CUDA).

Spec tests (CODEX_IMPLEMENT_ANALYTIC_BACKGROUND_VISIBILITY_CONSTRAINED_IEKF_2026-10-19.md sec. 7):
  [1] a black / bright / background-coloured stray Gaussian outside the mask:
      the RGB term carries no information for the black one, but the
      visible-alpha audit detects all three identically
  [2] object one frame behind its observation (1.2 cm), textured (real-like)
      background: mask=off lets the RGB term fit the background (object opacity
      on negative rays); mask=on creates no new violation at any accepted step
      (full nonlinear audit), does not increase the pre-existing violation, and
      still moves the object toward the data
  [3] a Gaussian stranded far outside the mask: no restoring direction is
      invented -- the frame is NOT reported feasible, the Gaussian is not
      dragged, and no new violation is created
  [4] known-nonobject-depth: a Gaussian genuinely behind a measured occluder
      carries no penalty; the same Gaussian in front of it does (individual
      splat depth, not composited depth)
  [9] camera / mask / observation counts must match (no silent zip truncation)
 [10] every accepted iteration carries a full nonlinear audit with ok = True

    python tests/test_visibility_constrained_iekf_phase1.py
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


def blob(n=40, seed=0, centre=(0.0, 0.0, 0.0), colour=None):
    from estimator.state import GaussianMotionState
    g = torch.Generator().manual_seed(seed)
    mu = (torch.rand(n, 3, generator=g) - 0.5) * 0.3 + torch.tensor(centre)
    col = torch.rand(n, 3, generator=g) * 0.6 + 0.2 if colour is None else torch.tensor(colour).expand(n, 3)
    z = torch.zeros(n, 3)
    return GaussianMotionState(mu=mu.cuda(), R=torch.eye(3).repeat(n, 1, 1).cuda(), v=z.cuda(), omega=z.cuda(),
                               colors=col.contiguous().cuda(), opacity=torch.full((n, 1), 0.85).cuda(),
                               scales=torch.full((n, 3), 0.035).cuda())


def cat(*states):
    return dataclasses.replace(states[0], **{f: torch.cat([getattr(s, f) for s in states]).contiguous()
                                             for f in ("mu", "R", "v", "omega", "colors", "opacity", "scales")})


def cov(n):
    P = torch.zeros((n, 12, 12), device="cuda")
    P[:, range(12), range(12)] = torch.tensor([1e-2] * 3 + [1e-2] * 3 + [1e-1] * 3 + [1e-1] * 3, device="cuda")
    return P


def observe(truth, cams, texture=None, wall_depth=None):
    """(rgbd_gt, masks): the truth rendered over a fixed background image, its
    silhouette as the mask, object depth inside it, wall depth outside."""
    from estimator.renderer_bridge import build_linearization
    lin = build_linearization(truth, cams)
    gt, masks = [], []
    for k, v in enumerate(lin.views):
        M = v.alpha > 0.5
        rgb = v.rgb.clone()
        if texture is not None:
            rgb = rgb + (1.0 - v.alpha)[None] * texture[k]
        d = torch.where(M, v.depth, torch.zeros_like(v.depth))
        if wall_depth is not None:
            d = torch.where(M, d, torch.full_like(d, wall_depth))
        gt.append((rgb, d, d > 0))
        masks.append(M)
    return gt, masks


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import IEKFConfig, iekf_frame_step
    from estimator.visibility_constraint import FrameVisibility, MaskPolicy
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    cams = make_cameras(n_cam=3, H=96, W=96, radius=2.2)
    pol = MaskPolicy("trusted-full-ray", epsilon=0.05, boundary_px=1, depth_noise_m=None)
    obj = blob()
    gt0, masks0 = observe(obj, cams)

    print("\n[1] detection does not depend on the stray Gaussian's colour")
    counts = {}
    for name, col in (("black", (0.0, 0.0, 0.0)), ("bright", (0.95, 0.95, 0.95)), ("bg-coloured", (0.0, 0.0, 0.0))):
        s = cat(obj, blob(n=1, seed=1, centre=(0.45, 0.0, 0.0), colour=col))
        vis = FrameVisibility(masks0, [g[1] for g in gt0], [g[2] for g in gt0], pol)
        L = build_linearization(s, cams)
        a = vis.row_values(vis.evaluate(L))
        counts[name] = int((a > pol.epsilon).sum())
    check("[1] the stray Gaussian violates the negative rays for every colour, identically",
          counts["black"] > 0 and len(set(counts.values())) == 1, str(counts))

    print("\n[2] object one step behind its observation, textured background")
    truth = dataclasses.replace(obj, mu=obj.mu + torch.tensor([0.012, 0.0, 0.0], device="cuda"))
    g = torch.Generator(device="cuda").manual_seed(7)
    tex = [torch.rand(3, 96, 96, device="cuda", generator=g) for _ in cams]
    gt, masks = observe(truth, cams, texture=tex)
    cfg = IEKFConfig(max_iters=8, pcg_max_iters=30)
    vis = FrameVisibility(masks, [x[1] for x in gt], [x[2] for x in gt], pol)
    a0 = vis.row_values(vis.evaluate(build_linearization(obj, cams)))
    out_off, _, _ = iekf_frame_step(obj, cov(obj.mu.shape[0]), gt, cams, 1 / 30, cfg)
    a_off = vis.row_values(vis.evaluate(build_linearization(out_off, cams)))
    out_on, _, diag = iekf_frame_step(obj, cov(obj.mu.shape[0]), gt, cams, 1 / 30, cfg, visibility=vis)
    a_on = vis.row_values(vis.evaluate(build_linearization(out_on, cams)))
    ev = next(r for r in diag.rows if r.get("event") == "visibility")
    err = lambda s: float((s.mu - truth.mu).norm(dim=1).mean())
    check("[2] mask=off: the RGB term puts object opacity on negative rays (background fit)",
          int((a_off > pol.epsilon).sum()) > int((a_on > pol.epsilon + pol.audit_tol).sum()),
          f"violating rays: off {int((a_off > pol.epsilon).sum())}, on {int((a_on > pol.epsilon + pol.audit_tol).sum())}, "
          f"start {int((a0 > pol.epsilon).sum())}")
    fa = ev["final_audit"]
    check("[2] mask=on: no new violation, pre-existing total violation not increased",
          fa["new_violations"] == 0 and fa["preexisting_total_violation_after"]
          <= fa["preexisting_total_violation_before"] + 1e-6,
          f"{ev['status']}: violating {fa['violating_now']}, pre-existing total "
          f"{fa['preexisting_total_violation_before']:.3f} -> {fa['preexisting_total_violation_after']:.3f}")
    check("[2] mask=on still moves the object toward the observation", err(out_on) < err(obj),
          f"mean centre error {err(obj):.4f} -> {err(out_on):.4f} (off: {err(out_off):.4f})")
    acc = [r for r in ev["iterations"] if r["accepted"]]
    check("[10] every accepted iteration has a passing full nonlinear audit",
          len(acc) > 0 and all(r["audit"] is not None and r["audit"]["ok"] for r in acc),
          f"{len(acc)} accepted, {ev['rejected_by_audit']} rejected by the audit, {ev['constrained_steps']} constrained")

    print("\n[3] a Gaussian stranded far outside the mask")
    stray = blob(n=1, seed=2, centre=(0.5, 0.25, 0.0), colour=(0.0, 0.0, 0.0))
    s3 = cat(obj, stray)
    vis3 = FrameVisibility(masks0, [x[1] for x in gt0], [x[2] for x in gt0], pol)
    out3, _, d3 = iekf_frame_step(s3, cov(s3.mu.shape[0]), gt0, cams, 1 / 30, cfg, visibility=vis3)
    e3 = next(r for r in d3.rows if r.get("event") == "visibility")
    moved = float((out3.mu[-1] - s3.mu[-1]).norm())
    check("[3] the frame is NOT reported feasible", e3["status"] != "FEASIBLE",
          f"{e3['status']}, violating {e3['final_audit']['violating_now']}")
    check("[3] the stranded Gaussian is not dragged by an invented direction", moved < 0.01, f"moved {moved:.4f} m")
    check("[3] no new violation is created", e3["final_audit"]["new_violations"] == 0
          and e3["final_audit"]["preexisting_total_violation_after"] <= e3["final_audit"]["preexisting_total_violation_before"] + 1e-6)

    print("\n[4] occluded vs in front, individual splat depth")
    polz = MaskPolicy("known-nonobject-depth", epsilon=0.05, boundary_px=0, depth_noise_m=0.01)
    L = build_linearization(obj, cams)
    v0 = L.views[0]
    d_obj = float(v0.cache.depth[v0.cache.radii > 0].min())
    occ = torch.zeros(96, 96, dtype=torch.bool, device="cuda")
    occ[30:66, 30:66] = True                                  # an occluder covering part of the object
    M = (v0.alpha > 0.5) & ~occ
    z_occ = torch.where(occ, torch.full((96, 96), d_obj - 0.3, device="cuda"), torch.zeros(96, 96, device="cuda"))
    visz = FrameVisibility([M], [z_occ], [occ], polz)
    hid = visz.evaluate(build_linearization(obj, [cams[0]]))
    check("[4] object Gaussians behind the measured occluder: no penalty",
          float(hid[0][0][occ].max()) == 0.0 and int(hid[0][4][occ].sum()) > 0,
          f"max A_vis on occluder rays {float(hid[0][0][occ].max()):.3f}, behind-count {int(hid[0][4][occ].sum())}")
    cam_dir = -torch.tensor(cams[0].viewmatrix[:3, 2].tolist(), device="cuda")
    front = dataclasses.replace(obj, mu=obj.mu + 0.5 * cam_dir / cam_dir.norm())
    infr = visz.evaluate(build_linearization(front, [cams[0]]))
    check("[4] the same Gaussians placed in front of the occluder: penalty",
          float(infr[0][0][occ].max()) > 0.3, f"max A_vis on occluder rays {float(infr[0][0][occ].max()):.3f}")

    print("\n[9] counts must match")
    try:
        FrameVisibility(masks0[:2], [x[1] for x in gt0], [x[2] for x in gt0], pol)
        check("[9] fewer masks than cameras is refused", False)
    except AssertionError:
        check("[9] fewer masks than cameras is refused", True)
    try:
        bad = FrameVisibility(masks0[:2], [x[1] for x in gt0[:2]], [x[2] for x in gt0[:2]], pol)
        iekf_frame_step(obj, cov(obj.mu.shape[0]), gt0, cams, 1 / 30, cfg, visibility=bad)
        check("[9] a 2-camera constraint on a 3-camera frame is refused", False)
    except AssertionError:
        check("[9] a 2-camera constraint on a 3-camera frame is refused", True)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
