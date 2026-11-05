#!/usr/bin/env python
"""Wrong-camera audit, tests 4-6: grouped frozen leave-one-out solves.

  [G]  <G u, v> == <u, G^T v> for the production GroupedObservation (pose and velocity)
  [L1] tiny real scene (3 cameras, 4 canonical groups, nonzero prior offset): the dense
       grouped matrix G^T A G is built column by column from the matrix-free operator;
       PCG delta_all and every leave-one-camera / -channel solve match dense solves; the
       exact identity I_k = A^-1 [b_k - A_k(delta_-k)] holds
  [L2] near-singular oracle (dense, designed): a camera adding a tiny contribution along a
       direction the prior barely constrains has a LARGE influence there, which vanishes
       when the prior curvature along it grows -- no gate involved

    python tests/test_wrong_camera_grouped_loo.py
"""
from __future__ import annotations

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
    from estimator.covariance import apply_prior, invert_blocks
    from estimator.renderer_bridge import embed_pose_to_full_state
    from estimator.supernodes import GroupedObservation, build_supernodes
    check, st_ = check_factory()
    cams, st, truth, obs, gt, masks, model, cfg = masked_scene(n=24, H=40, W=40)
    lin, terms, prec, alpha_terms, rhs, normal = frozen(st, cams, obs, model, cfg)
    n = st.mu.shape[0]
    part = build_supernodes(st.mu, target_size=6, seed=0)
    grouped = GroupedObservation(None, part)
    C = part.num_supernodes

    print("\n[G] group map adjoint")
    g = torch.Generator().manual_seed(1)
    u = torch.randn((C, 6), generator=g).cuda(); vv = torch.randn((n, 6), generator=g).cuda()
    lhs = float((grouped.expand(u).double() * vv.double()).sum()); rhs_ = float((u.double() * grouped.reduce(vv).double()).sum())
    check("<G u, v> == <u, G^T v>", abs(lhs - rhs_) / max(abs(lhs), 1e-30) < 1e-5, f"{abs(lhs - rhs_) / abs(lhs):.1e}")

    print("\n[L1] dense reference")
    P = torch.zeros((n, 12, 12), device="cuda")
    P[:, range(12), range(12)] = torch.tensor([1e-4] * 3 + [1e-3] * 3 + [1e-2] * 6, device="cuda")
    Lam, _ = invert_blocks(P.double())
    offset = 1e-3 * torch.randn((n, 12), generator=g, dtype=torch.float64).cuda()
    b12 = embed_pose_to_full_state(rhs.double()) - apply_prior(Lam, offset)
    sys_ = wa.FrozenSystem(lin, terms, Lam.float(), b12, normal, 1e-3, grouped, None)
    D = 12 * C
    eye = torch.eye(D, dtype=torch.float64, device="cuda")

    def dense(keys):
        cols = [sys_.GT(sys_.A12_minus(sys_.G(eye[k].reshape(C, 12)), keys)).reshape(-1) for k in range(D)]
        return torch.stack(cols, 1)
    Afull = dense(())
    sym = float((Afull - Afull.T).abs().max() / Afull.abs().max())
    check("dense grouped operator is symmetric", sym < 1e-4, f"{sym:.1e}")
    bk = wa.camera_channel_rhs(lin, terms)
    bg = sys_.GT(b12).reshape(-1)
    d_ref = torch.linalg.solve(0.5 * (Afull + Afull.T), bg)
    d_all, dg_all, info = sys_.solve((), tol=1e-10, max_iters=2000)
    e_all = float((dg_all.reshape(-1) - d_ref).norm() / d_ref.norm())
    check("PCG delta_all == dense solve", e_all < 1e-4, f"{e_all:.1e}, {info['iterations']} it")
    worst, worst_id = 0.0, 0.0
    from estimator.renderer_bridge import extract_pose
    for keys in ([(0, "rgb"), (0, "depth"), (0, "alpha")], [(1, "depth")], [(2, "alpha")], [(1, "rgb")]):
        Ak = dense(keys)
        bkk = sys_.GT(b12 - embed_pose_to_full_state(sum(bk[k] for k in keys))).reshape(-1)
        dk_ref = torch.linalg.solve(0.5 * (Ak + Ak.T), bkk)
        _, dgk, _ = sys_.solve(keys, tol=1e-10, max_iters=2000)
        worst = max(worst, float((dgk.reshape(-1) - dk_ref).norm() / dk_ref.norm()))
        I = d_ref - dk_ref
        d_k12 = sys_.G(dk_ref.reshape(C, 12))
        Akd = wa.camera_channel_normal(lin, terms, extract_pose(d_k12), keys=set(keys))
        r = sys_.GT(embed_pose_to_full_state(sum(bk[k] for k in keys) - sum(Akd[k] for k in keys))).reshape(-1)
        y = torch.linalg.solve(0.5 * (Afull + Afull.T), r)
        worst_id = max(worst_id, float((y - I).norm() / I.norm().clamp_min(1e-30)))
    check("PCG leave-one-out solves == dense solves (camera and channel arms)", worst < 1e-4, f"{worst:.1e}")
    check("exact identity I_k = A^-1 [b_k - A_k(delta_-k)] (float32 operators)", worst_id < 1e-4, f"{worst_id:.1e}")
    loo = wa.leave_one_out(sys_, 3, tol=1e-9, max_iters=2000)
    ids = [v["identity_rel_err"] for v in loo["arms"].values() if "identity_rel_err" in v]
    check("leave_one_out(): all arms converged, identity holds", all(v["solver"]["converged"] for v in loo["arms"].values())
          and max(ids) < 1e-4, f"max identity err {max(ids):.1e}, {len(loo['arms'])} arms")

    print("\n[L2] near-singular oracle (dense, designed)")
    torch.manual_seed(0)
    Dm = 6
    Q, _ = torch.linalg.qr(torch.randn(Dm, Dm, dtype=torch.float64))
    weak = Q[:, 0]
    A1 = Q[:, 1:] @ torch.diag(torch.linspace(1.0, 5.0, Dm - 1, dtype=torch.float64)) @ Q[:, 1:].T   # camera 1: blind along `weak`
    A2 = 1e-3 * torch.outer(weak, weak)                                     # camera 2: a tiny contribution along it
    b1 = A1 @ torch.randn(Dm, dtype=torch.float64)
    b2 = 1e-3 * weak * 0.5                                                   # small, biased force along the weak axis
    res = {}
    for prior in (1e-6, 10.0):
        A = A1 + A2 + prior * torch.eye(Dm, dtype=torch.float64)
        d_all = torch.linalg.solve(A, b1 + b2)
        d_m2 = torch.linalg.solve(A - A2, b1)
        I2 = d_all - d_m2
        x, info = wa._pcg64(lambda v: A @ v, b1 + b2, tol=1e-12, max_iters=200)
        res[prior] = (float((I2 @ weak).abs()), float((x - d_all).norm() / d_all.norm()))
    check("weak prior: the tiny camera-2 term has a large influence along the weak axis", res[1e-6][0] > 0.3,
          f"|I_2 . weak| {res[1e-6][0]:.3f}")
    check("strong prior along that axis: the influence vanishes", res[10.0][0] < 1e-3, f"{res[10.0][0]:.1e}")
    check("_pcg64 reproduces the dense solve", max(r[1] for r in res.values()) < 1e-8)
    return finish(st_)


if __name__ == "__main__":
    raise SystemExit(main())
