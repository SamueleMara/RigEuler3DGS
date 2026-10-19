#!/usr/bin/env python
"""Phase 2 (rigid SE(3)) and Phase 3 (fixed joint, displacement) with the hard
negative-ray constraint: stage lifts C_s = J_A J_g,s and constrained windows (real CUDA).

  [P2-a] C_2 = H_A J_rigid vs central FD of the full rigid-pose -> render -> A_vis
         pipeline (translation + rotation of the part, frozen contributor lists)
  [P2-b] a constrained rigid window: every accepted step passes the full
         nonlinear audit, the part moves toward the observation
  [P2-c] spec test 6: a misowned STATIONARY Gaussian inside the moving part's
         support: the constraint does not hold the part back to hide it (pose
         within noise of the unconstrained estimate) and the frame reports the
         conflict (released rays / not FEASIBLE)
  [P3-a] C_3 = H_A J_joint vs FD for a fixed prismatic joint (displacement only)
  [P3-b] a constrained fixed-joint window: audited steps, displacement moves
         toward the observation, joint geometry bitwise unchanged

    python tests/test_visibility_constrained_iekf_phase23.py
"""
from __future__ import annotations

import copy
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


def scene():
    from tests.test_visibility_constrained_iekf_phase1 import blob, cat
    base = blob(n=30, seed=0, centre=(-0.12, 0.0, 0.0))
    lid = blob(n=20, seed=3, centre=(0.16, 0.0, 0.0))
    return cat(base, lid), torch.arange(30, 50)


def setup(st, members, truth_states, cams, pol, tex=None):
    """mu0/R0, obs per frame, masks, closures shared by the Phase 2/3 runners."""
    from estimator.phase2_shared import effective_precision, render_state
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import IEKFConfig
    from estimator.visibility_constraint import SequenceVisibility
    from tests.test_visibility_constrained_iekf_phase1 import observe
    cfg = IEKFConfig()
    obs, masks = [], []
    for tr in truth_states:
        gt, m = observe(tr, cams, texture=tex)
        obs.append(gt)
        masks.append(m)
    seq = SequenceVisibility(pol, [[x.cpu() for x in m] for m in masks],
                             [[g[1].cpu() for g in gt] for gt in obs], [[g[2].cpu() for g in gt] for gt in obs])
    mu0, R0 = st.mu.double(), st.R.double()
    cam_ids = list(range(len(cams)))
    ctr = {"t": 0}

    def lin(s_, t):
        ctr["t"] = t
        mu, Rg = s_.gaussians(t, mu0, R0)
        return render_state(st, mu, Rg, cams, cfg, build_linearization)

    def prec(L):
        return effective_precision(L, obs[ctr["t"]], cam_ids, cfg, None)
    obs_pairs = [[(o[0], o[1]) for o in gt] for gt in obs]
    return mu0, R0, lin, prec, obs_pairs, seq


def frozen_fd(L0, Lp, Lm, ap, am, jv):
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
    m = fr.reshape(L0.depth.shape) & (jv.abs() > 0)
    return m


def lift_fd(state, frame, lin_fn, vis, eps_list=(1e-3, 3e-4, 1e-4)):
    """C v (native) vs central FD of A_vis through the stage's own retraction."""
    from estimator.joint_process import apply_correction
    from diff_gaussian_rasterization import iekf_vis_apply_H, iekf_visible_alpha
    g = torch.Generator().manual_seed(1)
    v = torch.randn(state.dim, generator=g, dtype=torch.float64)
    L0 = lin_fn(state, frame).views[0]
    md, z = vis.modes[0], vis.zobs[0]
    pose = state.jvp(frame, v.cuda(), state._mu0, state._R0).float()
    jv = iekf_vis_apply_H(L0.cache, torch.ones_like(md), z, 0.0, pose.contiguous())
    best = float("inf")
    for eps in eps_list:
        sp = apply_correction(copy.deepcopy(state), frame, eps * v, transport=False)[0]
        sm = apply_correction(copy.deepcopy(state), frame, -eps * v, transport=False)[0]
        Lp, Lm = lin_fn(sp, frame).views[0], lin_fn(sm, frame).views[0]
        ap = iekf_visible_alpha(Lp.cache, torch.ones_like(md), z, 0.0)[0]
        am = iekf_visible_alpha(Lm.cache, torch.ones_like(md), z, 0.0)[0]
        m = frozen_fd(L0, Lp, Lm, ap, am, jv)
        fd = (ap - am) / (2 * eps)
        if bool(m.any()):
            best = min(best, float((jv - fd)[m].norm() / fd[m].norm().clamp_min(1e-12)))
    return best


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from estimator.rigid_part_runtime import RigidIEKFConfig, RigidState, run_rigid_window
    from estimator.rigid_part_se3 import RigidPart
    from estimator.visibility_constraint import MaskPolicy
    from estimator.manifold import so3_exp
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    cams = make_cameras(n_cam=3, H=96, W=96, radius=2.2)
    pol = MaskPolicy("trusted-full-ray", epsilon=0.05, boundary_px=1, depth_noise_m=None)
    st, members = scene()
    lid = members.cuda()
    Rm = so3_exp(torch.tensor([[0.0, 0.0, 0.15]], dtype=torch.float64))[0].float().cuda()
    t_m = torch.tensor([0.0, 0.012, 0.0], device="cuda")
    pivot = torch.tensor([0.02, 0.0, 0.0], device="cuda")
    mu1 = st.mu.clone()
    mu1[lid] = (st.mu[lid] - pivot) @ Rm.T + pivot + t_m
    R1 = st.R.clone(); R1[lid] = Rm @ st.R[lid]
    truth1 = dataclasses.replace(st, mu=mu1, R=R1)
    g = torch.Generator(device="cuda").manual_seed(4)
    tex = [torch.rand(3, 96, 96, device="cuda", generator=g) for _ in cams]

    print("\n[P2] rigid SE(3)")
    mu0, R0, lin, prec, obs_pairs, seq = setup(st, members, [st, truth1], cams, pol, tex)
    rs = RigidState(RigidPart(members, num_frames=2), RigidIEKFConfig(max_iters=6))
    rs._mu0, rs._R0 = mu0, R0
    rs.parts[0].R[1] = so3_exp(torch.tensor([[0.0, 0.0, 0.07]], dtype=torch.float64))[0]
    fv = seq.for_frame(1)
    rel = lift_fd(rs, 1, lin, fv)
    check("[P2-a] C_2 = H_A J_rigid vs FD of the rigid pipeline", rel < 3e-2, f"rel {rel:.2e}")

    def run_rigid(vis, members_):
        s0 = RigidState(RigidPart(members_, num_frames=2), RigidIEKFConfig(max_iters=6))
        out = run_rigid_window(s0, mu0, R0, obs_pairs, lin, prec, cfg=RigidIEKFConfig(max_iters=6), visibility=vis)
        return out
    out_on = run_rigid(seq, members)
    out_off = run_rigid(None, members)
    rec = out_on["records"][1]
    vsum = rec.observability.get("visibility")
    p_on, p_off = out_on["part"], out_off["part"]
    ang = lambda R: float(torch.arccos(((torch.diagonal(R, dim1=-2, dim2=-1).sum() - 1) / 2).clamp(-1, 1)))
    acc = [r for r in vsum["iterations"] if r["accepted"]]
    check("[P2-b] every accepted rigid step passes the full nonlinear audit",
          all(r["audit"] is not None and r["audit"]["ok"] for r in acc), f"{len(acc)} accepted, status {vsum['status']}")
    check("[P2-b] the part rotates toward the observation (true 0.15 rad)", ang(p_on.R[1]) > 0.02,
          f"on {ang(p_on.R[1]):.3f} rad, off {ang(p_off.R[1]):.3f} rad")

    print("\n[P2-c] a misowned stationary Gaussian in the support (spec test 6)")
    from tests.test_visibility_constrained_iekf_phase1 import blob, cat
    out_g = blob(n=1, seed=9, centre=(-0.12, 0.17, 0.0))           # base top edge, truly static
    st6 = cat(st, out_g)
    mem6 = torch.cat([members, torch.tensor([st.mu.shape[0]])])
    mu1b = torch.cat([mu1, out_g.mu]); R1b = torch.cat([R1, out_g.R])
    truth6 = dataclasses.replace(st6, mu=mu1b, R=R1b)
    mu0, R0, lin, prec, obs_pairs, seq6 = setup(st6, mem6, [st6, truth6], cams, pol, tex)
    on6 = run_rigid_window(RigidState(RigidPart(mem6, num_frames=2), RigidIEKFConfig(max_iters=6)), mu0, R0,
                           obs_pairs, lin, prec, cfg=RigidIEKFConfig(max_iters=6), visibility=seq6)
    off6 = run_rigid_window(RigidState(RigidPart(mem6, num_frames=2), RigidIEKFConfig(max_iters=6)), mu0, R0,
                            obs_pairs, lin, prec, cfg=RigidIEKFConfig(max_iters=6), visibility=None)
    v6 = on6["records"][1].observability["visibility"]
    dang = abs(ang(on6["part"].R[1]) - ang(off6["part"].R[1]))
    released = max([r.get("audit", {}).get("released_rays", 0) if r.get("audit") else 0 for r in v6["iterations"]] or [0])
    check("[P2-c] the part is not held back to hide the outlier", dang < 0.02,
          f"angle on {ang(on6['part'].R[1]):.3f} vs off {ang(off6['part'].R[1]):.3f} rad")
    check("[P2-c] the conflict is reported (released rays or not FEASIBLE)",
          released > 0 or v6["status"] != "FEASIBLE", f"status {v6['status']}, released rays {released}")

    print("\n[P3] fixed prismatic joint, displacement only")
    from estimator import phase2_prismatic as mp
    from estimator.joint_displacement import DisplacementOnlyState, geometry_snapshot, geometry_unchanged
    from estimator.joint_iekf import JointIEKFConfig
    from estimator.phase2_model_comparison import fit_model, prismatic_state
    e = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    truth_p = dataclasses.replace(st, mu=torch.where(torch.isin(torch.arange(st.mu.shape[0], device="cuda"), lid)[:, None],
                                                     st.mu + torch.tensor([0.015, 0.0, 0.0], device="cuda"), st.mu))
    mu0, R0, lin, prec, obs_pairs, seqp = setup(st, members, [st, truth_p], cams, pol, tex)
    part = mp.PrismaticPart(members, e, torch.tensor([0.0, 0.004], dtype=torch.float64))
    ds = DisplacementOnlyState(prismatic_state(part))
    ds._mu0, ds._R0 = mu0, R0
    rel3 = lift_fd(ds, 1, lin, seqp.for_frame(1))
    check("[P3-a] C_3 = H_A J_joint vs FD (prismatic displacement)", rel3 < 3e-2, f"rel {rel3:.2e}")
    part = mp.PrismaticPart(members, e, torch.tensor([0.0, 0.0], dtype=torch.float64))
    ds = DisplacementOnlyState(prismatic_state(part))
    snap = geometry_snapshot(ds.parts[0])
    fit = fit_model(ds, mu0, obs_pairs, lin, prec, JointIEKFConfig(max_iters=5), label="p3", visibility=seqp)
    v3 = fit["records"][1].observability["visibility"]
    acc3 = [r for r in v3["iterations"] if r["accepted"]]
    check("[P3-b] every accepted step passes the audit; geometry unchanged",
          all(r["audit"]["ok"] for r in acc3) and geometry_unchanged(snap, ds.parts[0]),
          f"{len(acc3)} accepted, status {v3['status']}")
    check("[P3-b] the displacement moves toward the observation (true 0.015 m)", float(ds.parts[0].s[1]) > 0.003,
          f"s = {float(ds.parts[0].s[1]):.4f}")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
