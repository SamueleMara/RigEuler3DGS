#!/usr/bin/env python
"""Deterministic IEKF reductions (estimator/deterministic_ops.py) on a toy CUDA scene.

  [D1] each of the six reductions equals the production float-atomic result to float32
       accumulation accuracy (RGB-D H^T, RGB-D Fisher, alpha H^T, alpha Fisher, group
       adjoint, group Fisher)
  [D2] repeated calls are bitwise identical
  [D3] a full production Phase-1 replay (4 frames, canonical partition, masked mode) is
       bitwise identical across two runs

    python tests/test_deterministic_ops.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from tests.wrong_camera_fixtures import NAMES, check_factory, finish, frozen, masked_scene  # noqa: E402


def outputs(lin, terms, part, n, seed=0):
    from estimator import wrong_camera_audit as wa
    from estimator.supernodes import GroupedObservation
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    v = lin.views[0]
    t = terms[0]
    out["rgbd_HT"] = wa.view_HT(v, t.p_rgb * t.r_rgb, t.p_d * t.r_d).clone()
    out["rgbd_fisher"] = lin.build_block_fisher([(x.p_rgb, x.p_d) for x in terms]).clone()
    out["alpha_HT"] = wa.view_HT_alpha(v, t.kappa * t.r_a).clone()
    out["alpha_fisher"] = lin.build_alpha_fisher([x.kappa for x in terms]).clone()
    go = GroupedObservation(None, part)
    adj = torch.randn((n, 6), generator=g).cuda()
    out["group_adjoint"] = go.reduce(adj).clone()
    go.lin = lin
    out["group_fisher"] = go.block_fisher([(x.p_rgb, x.p_d) for x in terms]).clone()
    return out


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from estimator.supernodes import build_supernodes
    check, st_ = check_factory()
    cams, st, truth, obs, gt, masks, model, cfg = masked_scene()
    lin, terms, *_ = frozen(st, cams, obs, model, cfg)
    n = st.mu.shape[0]
    part = build_supernodes(st.mu, target_size=8, seed=0)
    prod = outputs(lin, terms, part, n)
    from estimator import deterministic_ops as d
    d.enable()
    det1 = outputs(lin, terms, part, n)
    det2 = outputs(lin, terms, part, n)
    print("\n[D1] deterministic == production (float32 accumulation accuracy)")
    for k in prod:
        rel = float((det1[k].double() - prod[k].double()).norm() / prod[k].double().norm().clamp_min(1e-30))
        check(f"[D1] {k}", rel < 1e-5, f"rel {rel:.1e}")
    print("\n[D2] bitwise repeatability")
    check("[D2] every reduction is bitwise identical across calls", all(torch.equal(det1[k], det2[k]) for k in det1))
    print("\n[D3] Phase-1 replay bitwise identical")
    from audit_phase1_wrong_camera import replay
    from estimator.masked_observation import MaskedSequence
    window = [(gt, {"original_frame_index": k}) for k in range(4)]
    mseq = MaskedSequence(model, [[m.cpu() for m in masks]] * 4, list(NAMES), [0, 1, 2, 3])
    ns = argparse.Namespace(supernode_size=8, seed=0, bootstrap_iters=4, dt=1 / 25.0, visibility_constraint=None,
                            masked_observation=mseq, phase1_partition="canonical")
    r1, r2 = replay(st, cams, window, cfg, ns), replay(st, cams, window, cfg, ns)
    same = all(torch.equal(a, b) for a, b in zip(r1["positions"], r2["positions"])) and all(
        torch.equal(fa[k], fb[k]) for fa, fb in zip(r1["capture"].frames, r2["capture"].frames)
        for k in ("mu", "v", "omega", "P_post"))
    check("[D3] positions, velocities and covariances bitwise identical across two replays", same)
    return finish(st_)


if __name__ == "__main__":
    raise SystemExit(main())
