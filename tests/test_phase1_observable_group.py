"""Gate I.D: the observation-supported grouped update (estimator/observable_group.py).

RigEuler3DGS_PHASE1_COHERENCE_CODEX_IMPLEMENTATION_SPEC_2026-11-16.md section I.D tests 1-8.

    python tests/test_phase1_observable_group.py

CPU dense oracle (no renderer):
 [1] adjoint <B u, y> == <u, B^T y> and dense B^T A B parity, with pose-velocity correlation
 [2] 2-member group, one observed and one hidden: the hidden member's update is exactly 0
 [3] partially observable member: the update lies in P_ee a (the measured direction only)
 [4] ungrouped full-rank measurement: the restricted solve == the exact Kalman MAP
 [5] zero information: mean stays at x^-, the block posterior keeps P^- exactly
 [6] Joseph covariance of the restricted linear update vs the block approximation
     (reported; hidden member asserted exact)
CUDA (real renderer):
 [7] a moving visible part and a static part hidden behind a front occluder, one supernode:
     legacy drags / pushes the hidden part; the new likelihood + observable update keeps it
     at the prediction while the visible part is still re-acquired; accepted steps lower
     the new objective; hidden members keep P^-; the real GroupedObservation adjoint holds
 [8] deterministic mode: two identical frame solves are bitwise equal
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


class DenseGroups:
    """The rigid group tangent of supernodes.GroupedObservation as a dense CPU map:
    dmu_i = dt_c + dphi_c x (mu_i - c_c), dphi_i = dphi_c."""

    def __init__(self, mu, sid):
        self.mu, self.sid = mu.double(), sid
        self.num_supernodes = int(sid.max()) + 1
        cent = torch.zeros((self.num_supernodes, 3), dtype=torch.float64)
        cnt = torch.zeros(self.num_supernodes, dtype=torch.float64)
        cent.index_add_(0, sid, self.mu); cnt.index_add_(0, sid, torch.ones_like(sid, dtype=torch.float64))
        self.cent = cent / cnt[:, None]
        n = mu.shape[0]
        G = torch.zeros((6 * n, 6 * self.num_supernodes), dtype=torch.float64)
        for i in range(n):
            c = int(sid[i]); r = self.mu[i] - self.cent[c]
            G[6 * i:6 * i + 3, 6 * c:6 * c + 3] = torch.eye(3, dtype=torch.float64)
            G[6 * i:6 * i + 3, 6 * c + 3:6 * c + 6] = -torch.tensor([[0, -r[2], r[1]], [r[2], 0, -r[0]],
                                                                       [-r[1], r[0], 0]], dtype=torch.float64)
            G[6 * i + 3:6 * i + 6, 6 * c + 3:6 * c + 6] = torch.eye(3, dtype=torch.float64)
        self.G = G

    def expand(self, du):
        return (self.G @ du.double().reshape(-1)).reshape(-1, 6).to(du.dtype)

    def reduce(self, y):
        return (self.G.T @ y.double().reshape(-1)).reshape(-1, 6).to(y.dtype)


def random_prior(n, seed):
    """Per-Gaussian 12x12 covariance with a genuine pose-velocity cross block (the shape
    one constant-twist prediction produces: P = F P0 F^T + Q)."""
    g = torch.Generator().manual_seed(seed)
    P = torch.zeros((n, 12, 12), dtype=torch.float64)
    for i in range(n):
        A = torch.randn(12, 12, generator=g, dtype=torch.float64) * 0.1
        P0 = A @ A.T + torch.diag(torch.tensor([1e-2] * 6 + [1e-1] * 6, dtype=torch.float64))
        F = torch.eye(12, dtype=torch.float64); F[:6, 6:] = torch.eye(6, dtype=torch.float64) / 30
        P[i] = F @ P0 @ F.T + torch.diag(torch.tensor([1e-4] * 6 + [1e-3] * 6, dtype=torch.float64))
    return P


def blockdiag(blocks):
    return torch.block_diag(*list(blocks))


def dense_problem(n, observed_rows, seed=0):
    """H: m x 12n acting on the pose part of the listed Gaussians; W = I."""
    g = torch.Generator().manual_seed(seed + 100)
    rows = []
    for i, r in observed_rows.items():
        if r == 0:
            continue
        h = torch.zeros((r, 12 * n), dtype=torch.float64)
        h[:, 12 * i:12 * i + 6] = torch.randn(r, 6, generator=g, dtype=torch.float64) * 30
        rows.append(h)
    H = torch.cat(rows) if rows else torch.zeros((0, 12 * n), dtype=torch.float64)
    y = torch.randn(H.shape[0], generator=g, dtype=torch.float64)
    return H, y


def fisher_blocks(H, n):
    F = torch.zeros((n, 6, 6), dtype=torch.float64)
    HtH = H.T @ H
    for i in range(n):
        F[i] = HtH[12 * i:12 * i + 6, 12 * i:12 * i + 6]
    return F


def restricted_solve(P, H, y, grouped, kind="smooth"):
    """Dense B, then du = (B^T A B)^+ B^T H^T y, dx = B du; also the Joseph covariance."""
    from estimator.observable_group import build_map, operators
    n = P.shape[0]
    omap = build_map(P, fisher_blocks(H, n), kind)
    Bf, BTf = operators(omap, grouped)
    k = 6 * (grouped.num_supernodes if grouped is not None else n)
    Bm = torch.stack([Bf(e.reshape(-1, 6)).reshape(-1) for e in torch.eye(k, dtype=torch.float64)], dim=1)
    Lam = torch.linalg.inv(blockdiag(P))
    A = Lam + H.T @ H
    AB = Bm.T @ A @ Bm
    du = torch.linalg.lstsq(AB, Bm.T @ (H.T @ y)).solution
    dx = Bm @ du
    KB = Bm @ torch.linalg.pinv(AB) @ Bm.T @ H.T
    I = torch.eye(12 * n, dtype=torch.float64)
    J = (I - KB @ H) @ blockdiag(P) @ (I - KB @ H).T + KB @ KB.T
    return dx.reshape(n, 12), J, Bm, BTf, Bf, omap


def main() -> int:
    from estimator.covariance import approximate_posterior_information
    torch.manual_seed(0)

    print("\n[1] adjoint and dense B^T A B parity (CPU)")
    n = 5
    mu = torch.randn(n, 3, dtype=torch.float64) * 0.1
    grp = DenseGroups(mu, torch.tensor([0, 0, 1, 1, 1]))
    P = random_prior(n, 1)
    H, y = dense_problem(n, {0: 4, 1: 0, 2: 6, 3: 2, 4: 0})
    dx, J, Bm, BTf, Bf, omap = restricted_solve(P, H, y, grp)
    u = torch.randn(grp.num_supernodes, 6, dtype=torch.float64)
    yy = torch.randn(n, 12, dtype=torch.float64)
    lhs, rhs = float((Bf(u) * yy).sum()), float((u * BTf(yy)).sum())
    check("[1] <Bu,y> == <u,B^T y>", abs(lhs - rhs) < 1e-10 * max(1.0, abs(lhs)), f"{lhs:.12e} vs {rhs:.12e}")
    A = torch.linalg.inv(blockdiag(P)) + H.T @ H
    mf = BTf((A @ Bf(u).reshape(-1)).reshape(n, 12)).reshape(-1)
    check("[1] matrix-free B^T A B u == dense", float((mf - Bm.T @ A @ Bm @ u.reshape(-1)).abs().max()) < 1e-8)
    check("[1] prior has a genuine pose-velocity cross block", float(P[:, 6:, :6].abs().max()) > 1e-4)

    print("\n[2] 2-member group: observed + hidden")
    P2 = random_prior(2, 2)
    grp2 = DenseGroups(torch.tensor([[0.0, 0, 0], [0.1, 0.05, 0]], dtype=torch.float64), torch.tensor([0, 0]))
    H2, y2 = dense_problem(2, {0: 8, 1: 0})
    dx2, J2, *_ = restricted_solve(P2, H2, y2, grp2)
    check("[2] hidden member: dx (pose AND velocity) exactly 0", float(dx2[1].abs().max()) == 0.0,
          f"max {float(dx2[1].abs().max()):.1e}")
    check("[2] observed member updates", float(dx2[0, :6].norm()) > 1e-4, f"|deta| {float(dx2[0, :6].norm()):.3e}")
    check("[2] observed member's velocity correction = K deta (conditional mean)",
          float((dx2[0, 6:] - (P2[0, 6:, :6] @ torch.linalg.inv(P2[0, :6, :6]) @ dx2[0, :6])).abs().max()) < 1e-10)
    # legacy rigid expansion of the same problem drags the hidden member
    Gd = grp2.G
    Lam2 = torch.linalg.inv(blockdiag(P2)); A2 = Lam2 + H2.T @ H2
    G12 = torch.zeros((24, 12), dtype=torch.float64)
    for i in range(2):
        G12[12 * i:12 * i + 6, :6] = Gd[6 * i:6 * i + 6]; G12[12 * i + 6:12 * i + 12, 6:] = Gd[6 * i:6 * i + 6]
    du_leg = torch.linalg.solve(G12.T @ A2 @ G12, G12.T @ H2.T @ y2)
    check("[2] (contrast) the legacy 12-DoF group expansion moves the hidden member",
          float((G12 @ du_leg).reshape(2, 12)[1, :6].norm()) > 1e-4,
          f"legacy |deta_hidden| {float((G12 @ du_leg).reshape(2, 12)[1, :6].norm()):.3e}")

    print("\n[3] partially observable member (rank-1 measurement)")
    P3 = random_prior(1, 3)
    a = torch.randn(6, dtype=torch.float64)
    H3 = torch.zeros((1, 12), dtype=torch.float64); H3[0, :6] = a * 30
    y3 = torch.tensor([0.7], dtype=torch.float64)
    for kind in ("smooth", "rank"):
        dx3, *_ = restricted_solve(P3, H3, y3, None, kind)
        d = P3[0, :6, :6] @ a
        cosang = float(dx3[0, :6] @ d) / (float(dx3[0, :6].norm()) * float(d.norm()))
        check(f"[3] {kind}: the update is along P_ee a only", abs(abs(cosang) - 1) < 1e-10, f"|cos| {abs(cosang):.12f}")

    print("\n[4] ungrouped full-rank: restricted solve == exact Kalman MAP")
    P4 = random_prior(3, 4)
    H4, y4 = dense_problem(3, {0: 8, 1: 9, 2: 7}, seed=4)
    dx4, *_ = restricted_solve(P4, H4, y4, None)
    exact = torch.linalg.solve(torch.linalg.inv(blockdiag(P4)) + H4.T @ H4, H4.T @ y4).reshape(3, 12)
    check("[4] max |dx - dx_Kalman| < 1e-9", float((dx4 - exact).abs().max()) < 1e-9,
          f"{float((dx4 - exact).abs().max()):.2e}")

    print("\n[5] zero information")
    H5 = torch.zeros((0, 24), dtype=torch.float64)
    dx5, J5, *_ = restricted_solve(P2, H5, torch.zeros(0, dtype=torch.float64), grp2)
    Lp = approximate_posterior_information(torch.linalg.inv(P2), torch.zeros((2, 6, 6), dtype=torch.float64))
    check("[5] mean stays at x^- (dx == 0)", float(dx5.abs().max()) == 0.0)
    check("[5] block posterior == P^- (rel 1e-10)",
          float((torch.linalg.inv(Lp) - P2).abs().max()) < 1e-10 * float(P2.abs().max()))
    check("[5] Joseph covariance == P^-", float((J5 - blockdiag(P2)).abs().max()) < 1e-12)

    print("\n[6] Joseph covariance of the restricted update vs the production block approximation")
    Pfull = blockdiag(P)
    Fb = fisher_blocks(H, n)
    approx = torch.linalg.inv(approximate_posterior_information(torch.linalg.inv(P), Fb))
    exact_post = torch.linalg.inv(torch.linalg.inv(Pfull) + H.T @ H)
    hidden = [1, 4]
    for i in range(n):
        Jb = J[12 * i:12 * i + 12, 12 * i:12 * i + 12]
        Eb = exact_post[12 * i:12 * i + 12, 12 * i:12 * i + 12]
        rel = float((approx[i] - Jb).norm() / Jb.norm())
        tr = float(torch.trace(approx[i][:6, :6]) / torch.trace(Jb[:6, :6]))
        trE = float(torch.trace(Eb[:6, :6]) / torch.trace(Jb[:6, :6]))
        print(f"      member {i} ({'hidden' if i in hidden else 'observed'}): ||block - Joseph||/||Joseph|| {rel:.3f}, "
              f"pose trace block/Joseph {tr:.3f}, unrestricted-Kalman/Joseph {trE:.3f}")
        if i in hidden:
            check(f"[6] hidden member {i}: Joseph block == P^- and block approximation == P^-",
                  float((Jb - P[i]).abs().max()) < 1e-12 and float((approx[i] - P[i]).abs().max()) < 1e-10)
    check("[6] Joseph covariance is symmetric PSD", float(torch.linalg.eigvalsh(0.5 * (J + J.T)).min()) > -1e-12)
    print("      -> the block approximation is NOT the restricted update's covariance on observed members; the "
          "posterior is reported as uncertified")

    if not torch.cuda.is_available():
        print("\nNOT_RUN: [7] [8] need CUDA")
    else:
        cuda_tests()
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED:", *FAILURES, sep="\n  ")
        return 1
    return 0


def cuda_tests():
    from estimator.masked_observation import MaskedObservationModel, prepare
    from estimator.occlusion_likelihood import OcclusionLikelihood
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import IEKFConfig, iekf_frame_step
    from tests.test_phase1_occlusion_likelihood import blob, cat
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

    cams = make_cameras(n_cam=3, H=96, W=96, radius=2.2)
    A0 = blob(n=40, seed=0, spread=0.3)
    Bh = blob(n=12, seed=4, centre=(0.0, 0.45, 0.0), spread=0.1)
    nA = A0.mu.shape[0]
    canon = cat(A0, Bh)
    shift = torch.tensor([0.012, 0.0, 0.0], device="cuda")
    truthA = dataclasses.replace(A0, mu=A0.mu + shift)
    la, lb = build_linearization(truthA, cams), build_linearization(Bh, cams)
    obs = []
    for k, (va, vb) in enumerate(zip(la.views, lb.views)):
        M = va.alpha > 0.5
        foot = vb.alpha > 1e-7
        occ_depth = float(vb.depth[vb.alpha > 0.5].min()) - 0.3
        d = torch.where(M, va.depth, torch.full_like(va.depth, 4.0))
        d = torch.where(foot & ~M, torch.full_like(d, occ_depth), d)
        rgb = va.rgb + (1 - va.alpha)[None] * 0.3
        rgb[0] = torch.where(foot, torch.full_like(rgb[0], 0.9), rgb[0])
        obs.append(prepare(rgb, d, d > 0, M, f"c{k}", 1))
    model = MaskedObservationModel("masked-rgb-expected-quad", (0.5, 0.5, 0.5), "uniform01", 0.1)
    base = IEKFConfig(max_iters=8, pcg_max_iters=30, camera_names=("c0", "c1", "c2"), use_supernodes=True,
                      supernode_target_size=64, depth_sigma_model="constant", depth_sigma_a=0.02)

    def P0(n):
        P = torch.zeros((n, 12, 12), device="cuda")
        P[:, range(12), range(12)] = torch.tensor([1e-2] * 6 + [1e-1] * 6, device="cuda")
        return P

    def run(mode):
        if mode == "legacy":
            return iekf_frame_step(canon, P0(canon.mu.shape[0]), obs, cams, 1 / 30, base, observation_model=model)
        cfg = dataclasses.replace(base, group_update="observable")
        return iekf_frame_step(canon, P0(canon.mu.shape[0]), obs, cams, 1 / 30, cfg, observation_model=model,
                               occlusion=OcclusionLikelihood("certified"))

    print("\n[7] visible part moves 12 mm, a static part is hidden behind a front occluder (one supernode)")
    res = {m: run(m) for m in ("legacy", "new")}
    errA = lambda s: float((s.mu[:nA] - truthA.mu).norm(dim=1).mean())          # noqa: E731
    dB = lambda s: float((s.mu[nA:] - Bh.mu).norm(dim=1).max())                 # noqa: E731
    for m, (x, Pp, dg) in res.items():
        print(f"      {m:6s}: visible part error {errA(canon) * 1e3:.2f} -> {errA(x) * 1e3:.2f} mm, "
              f"hidden part max displacement {dB(x) * 1e3:.3f} mm")
    x_new, P_new, d_new = res["new"]
    x_leg = res["legacy"][0]
    check("[7] legacy moves the hidden part (drag + off-mask push)", dB(x_leg) > 1e-3, f"{dB(x_leg) * 1e3:.2f} mm")
    check("[7] new: the hidden part stays exactly at the prediction", dB(x_new) < 1e-7, f"{dB(x_new):.1e} m")
    check("[7] new: the visible part is still re-acquired (error at least halved)", errA(x_new) < 0.5 * errA(canon),
          f"{errA(canon) * 1e3:.2f} -> {errA(x_new) * 1e3:.2f} mm")
    ev = [r for r in d_new.rows if r.get("event") in ("initial", "accepted")]
    check("[7] every accepted step lowered the new full objective",
          len(ev) > 1 and all(ev[i + 1]["energy"] < ev[i]["energy"] for i in range(len(ev) - 1)),
          " > ".join(f"{r['energy']:.2f}" for r in ev))
    check("[7] the objective contains the negative-ray term", all("negative_ray_energy" in r for r in ev))
    Pprior_pose = torch.diag(torch.tensor([1e-2] * 6, device="cuda")).double()
    # hidden members: F == 0 -> block posterior == predicted covariance (process noise added once)
    from estimator.covariance import make_process_noise
    Qn = make_process_noise(canon.mu.shape[0], 1 / 30, base.process_noise_pos, base.process_noise_rot,
                            base.process_noise_vel, base.process_noise_omega, device="cuda", dtype=torch.float32)
    del Pprior_pose, Qn
    ex = d_new.observability_export["per_gaussian"]
    check("[7] export: hidden members have zero whitened information, visible ones do not",
          float(ex["lam_max"][nA:].max()) == 0.0 and int((ex["lam_max"][:nA] > 0).sum()) >= int(0.8 * nA),
          f"hidden lam_max {float(ex['lam_max'][nA:].max()):.1e}, visible informed {int((ex['lam_max'][:nA] > 0).sum())}/{nA}")
    check("[7] export: hidden members' occluded weight > 0, front weight ~0",
          float(ex["occluded_weight"][nA:].sum()) > 1.0 and float(ex["front_weight"][nA:].max()) < 1e-6,
          f"occluded {float(ex['occluded_weight'][nA:].sum()):.1f}, front {float(ex['front_weight'][nA:].max()):.1e}")
    # posterior of hidden members equals the prior the step used: recompute with the same predict
    from estimator.process import error_state_transition
    from estimator.covariance import covariance_predict, make_process_noise as mpn
    Fm = error_state_transition(canon, 1 / 30, mode="so3", tau=0.0)
    Qp = mpn(canon.mu.shape[0], 1 / 30, base.process_noise_pos, base.process_noise_rot, base.process_noise_vel,
             base.process_noise_omega, device="cuda", dtype=torch.float32, tau=0.0)
    Pprior = covariance_predict(P0(canon.mu.shape[0]), Fm, Qp)
    relP = float((P_new[nA:].double() - Pprior[nA:].double()).abs().max() / Pprior[nA:].double().abs().max())
    check("[7] hidden members keep the predicted covariance (no fictitious contraction)", relP < 1e-5,
          f"max rel diff {relP:.1e}")

    from estimator.supernodes import GroupedObservation, build_supernodes
    part = build_supernodes(canon.mu, target_size=16)
    gro = GroupedObservation(linearization=None, partition=part)
    u = torch.randn(part.num_supernodes, 6, device="cuda")
    yv = torch.randn(canon.mu.shape[0], 6, device="cuda")
    l_, r_ = float((gro.expand(u) * yv).sum()), float((u * gro.reduce(yv)).sum())
    check("[7] the real GroupedObservation adjoint holds (rel < 1e-5)", abs(l_ - r_) <= 1e-5 * max(abs(l_), 1.0),
          f"{l_:.6e} vs {r_:.6e}")

    print("\n[8] deterministic mode: identical frame solves are bitwise equal")
    from estimator.deterministic_ops import enable
    info = enable()
    cfg = dataclasses.replace(base, group_update="observable")
    occ = OcclusionLikelihood("certified", deterministic=True)
    outs = [iekf_frame_step(canon, P0(canon.mu.shape[0]), obs, cams, 1 / 30, cfg, observation_model=model,
                            occlusion=occ) for _ in range(2)]
    check("[8] mu, R and P bitwise equal across two runs",
          torch.equal(outs[0][0].mu, outs[1][0].mu) and torch.equal(outs[0][0].R, outs[1][0].R)
          and torch.equal(outs[0][1], outs[1][1]), f"deterministic ops: {str(info)[:80]}")


if __name__ == "__main__":
    raise SystemExit(main())
