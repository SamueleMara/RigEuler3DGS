"""`exact_cached_structured` is the SAME system as `legacy_reference`.

CODEX_EXACT_FUSED_IEKS_ACCELERATION_SPARSE_DATA_LAYOUT_NO_APPROXIMATIONS.md
sections 3-6 and 8.

Fixture: two sources (4 + 4 Gaussians) OVERLAPPING in every view, so the image
operator has real cross-Gaussian and cross-source terms; a rotating body, so
the SO(3) process residuals and Jacobians are nonzero; 4 frames, 2 cameras;
the initial group anchored at frame 0 and the final group at frame 3.

Every comparison is against a MEASURED floor: the CUDA image path is float32
with atomics, so two legacy builds at the same state already differ. The
floor is measured from three legacy-vs-legacy rebuilds and each check allows
at most 4x it. The float64 process stencil is held to ~1e-12.

    python tests/test_exact_cached_structured_backend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CHECKS = 0
FAILURES: list[str] = []
ROWS = []


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    ROWS.append((label, bool(cond), detail))
    if not cond:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")
    else:
        print(f"  ok    {label}  {detail}")


def rel(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).norm()) / max(float(a.norm()), 1e-300)


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from bidirectional_iekf import accelerated as acc
    from bidirectional_iekf.full_window_free import (block_jacobi_preconditioner,
                                                     pcg)
    from bidirectional_iekf.ieks import (IEKSConfig, Trajectory, UnifiedFreeIEKS,
                                         _eliminated)
    from bidirectional_iekf.observations import FreeRGBDObservation
    from estimator.manifold import so3_exp
    from estimator.renderer_bridge import build_linearization
    from estimator.state import GaussianMotionState
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

    torch.manual_seed(0)
    N, T, dt = 8, 4, 0.1
    A_rows, B_rows = torch.arange(0, 4), torch.arange(4, 8)
    cams = make_cameras(n_cam=2, H=64, W=64, radius=2.2)
    g = torch.Generator().manual_seed(5)
    colors = torch.rand(N, 3, generator=g) * 0.7 + 0.2
    mu0 = 0.18 * (torch.rand(N, 3, generator=g) - 0.5)      # all overlapping
    w = torch.tensor([0.0, 0.0, 0.6])

    def make(mu, R, v, om):
        return GaussianMotionState(
            mu=mu.float().cuda(), R=R.float().cuda(), v=v.float().cuda(),
            omega=om.float().cuda(), colors=colors.cuda(),
            opacity=torch.full((N, 1), 0.7).cuda(),
            scales=torch.tensor([[.10, .07, .05]]).expand(N, 3).contiguous().cuda())

    def truth(t):
        Rb = so3_exp((w * dt * t).unsqueeze(0).double())[0].float()
        mu = mu0 @ Rb.T
        return make(mu, Rb.expand(N, 3, 3).contiguous(), torch.cross(
            w.expand(N, 3), mu, dim=1), w.expand(N, 3).contiguous())
    states = [truth(t) for t in range(T)]
    obs_data = [[(r.clone(), d.clone(), torch.ones_like(d, dtype=torch.bool))
                 for r, d in build_linearization(states[t], cams).predicted()]
                for t in range(T)]
    Q = torch.diag(torch.tensor([1e4] * 3 + [1e4] * 3 + [1e2] * 3 + [1e2] * 3,
                                dtype=torch.float64))
    Q[0:3, 6:9] = Q[6:9, 0:3] = 0.3e2 * torch.eye(3, dtype=torch.float64)  # pose-rate cross
    Q = Q.expand(N, 12, 12).contiguous()
    # per-Gaussian rotation and rate errors, so every SO(3) block differs
    # from Gaussian to Gaussian and the residual rotation is nonzero
    warm = Trajectory(states=[make(
        states[t].mu.cpu() + 0.008 * torch.randn(N, 3, generator=g),
        so3_exp(0.05 * torch.randn(N, 3, generator=g).double()).float() @ states[t].R.cpu(),
        torch.zeros(N, 3),
        states[t].omega.cpu() * (0.5 + torch.rand(N, 1, generator=g)))
        for t in range(T)], frame_ids=list(range(T)), dt=dt)
    groups = ((0, A_rows), (T - 1, B_rows))

    def solver(backend, **kw):
        return UnifiedFreeIEKS(
            FreeRGBDObservation(cams, obs_data, frame_ids=list(range(T))), Q,
            config=IEKSConfig(source_pose_policy="fixed_conditional",
                              anchor_groups=groups, linear_backend=backend,
                              solve_on_cpu=True, verbose=False, **kw))

    lam = 3e-3
    print("[1] the measured floor: legacy against legacy")
    refs = [solver("legacy_reference").build_system(warm, lam)[0] for _ in range(4)]
    u = torch.randn(T, N, 12, generator=g, dtype=torch.float64)
    v = torch.randn(T, N, 12, generator=g, dtype=torch.float64)
    floor = {"matvec": max(rel(refs[0].information_matvec(u), r.information_matvec(u))
                           for r in refs[1:]),
             "rhs": max(rel(refs[0].rhs(), r.rhs()) for r in refs[1:]),
             "curv": max(rel(refs[0].curvature_scale(), r.curvature_scale())
                         for r in refs[1:])}
    print("       " + ", ".join(f"{k} {v_:.2e}" for k, v_ in floor.items()))
    ref = refs[0]

    def tol(k):
        return max(4 * floor[k], 1e-12)

    print("\n[2] the frozen host cache round-trips exactly")
    ob = FreeRGBDObservation(cams, obs_data, frame_ids=list(range(T)))
    cache = ob.linearize_frame(warm.states[1], 1)
    tensors: list = []
    acc._pack(cache, tensors, {})
    packed = acc.pack_frame(cache)
    dev = torch.empty(packed.nbytes, dtype=torch.uint8, device="cuda")
    dev.copy_(packed.flat)
    back = acc.unpack_frame(packed, dev)
    t2: list = []
    acc._pack(back, t2, {})
    same = all(torch.equal(a.cpu(), b.cpu()) and a.dtype == b.dtype and a.shape == b.shape
               for a, b in zip(tensors, t2)) and len(tensors) == len(t2)
    check("every tensor bitwise identical, dtype and shape preserved", same,
          f"{len(tensors)} tensors, {packed.nbytes / 2**20:.2f} MiB for 2 cameras")
    v0, v1 = back.linearization.views
    check("shared tensors stay shared (one means3D for both cameras)",
          v0.cache.means3D.data_ptr() == v1.cache.means3D.data_ptr())
    check("integer contributor order (point_list, ranges) preserved",
          torch.equal(cache.linearization.views[0].cache.point_list,
                      v0.cache.point_list)
          and torch.equal(cache.linearization.views[0].cache.ranges, v0.cache.ranges))
    check("the camera is the same object type with the same calibration",
          type(v0.camera) is type(cache.linearization.views[0].camera)
          and torch.equal(v0.camera.projmatrix, cache.linearization.views[0].camera.projmatrix))

    print("\n[3] same A, b, D")
    cs = solver("exact_cached_structured")
    op = cs.build_system(warm, lam)[0]
    d_mv = rel(ref.information_matvec(u), op.information_matvec(u))
    check("A u", d_mv <= tol("matvec"), f"{d_mv:.2e} (floor {floor['matvec']:.2e})")
    d_b = rel(ref.rhs(), op.rhs())
    check("b = -grad", d_b <= tol("rhs"), f"{d_b:.2e} (floor {floor['rhs']:.2e})")
    d_c = rel(ref.curvature_scale(), op.curvature_scale())
    check("LM metric D", d_c <= tol("curv"), f"{d_c:.2e} (floor {floor['curv']:.2e})")
    s1 = float((u * op.information_matvec(v)).sum())
    s2 = float((v * op.information_matvec(u)).sum())
    l1 = float((u * ref.information_matvec(v)).sum())
    l2 = float((v * ref.information_matvec(u)).sum())
    asym_new, asym_ref = abs(s1 - s2) / abs(s1), abs(l1 - l2) / abs(l1)
    check("symmetry u'Av = v'Au, as symmetric as the legacy operator",
          asym_new <= max(4 * asym_ref, 4 * floor["matvec"]),
          f"{asym_new:.2e} vs legacy's own {asym_ref:.2e} (float32 JVP/VJP)")
    check("positive on random directions",
          float((u * op.information_matvec(u)).sum()) > 0)
    Au = op.information_matvec(u)
    check("anchored pose coordinates are zero in A u and b",
          float(Au[0, A_rows, :6].abs().max()) == 0 and float(Au[T - 1, B_rows, :6].abs().max()) == 0
          and float(op.rhs()[0, A_rows, :6].abs().max()) == 0)
    check("and the other source's pose at those frames is NOT eliminated",
          float(Au[0, B_rows, :6].abs().max()) > 0 and float(Au[T - 1, A_rows, :6].abs().max()) > 0)

    print("\n[4] the structured process stencil is the dense one, in float64")
    from bidirectional_iekf.process_factors import process_residual_jacobians
    proc = cs._frozen["proc"]
    worst = 0.0
    for t in range(T - 1):
        Ad, Bd = process_residual_jacobians(warm.states[t], warm.states[t + 1], dt)
        Qc = Q.cuda()
        ut, ut1 = u[t].cuda(), u[t + 1].cuda()
        z = torch.einsum("nij,nj->ni", Ad, ut) + torch.einsum("nij,nj->ni", Bd, ut1)
        wv = torch.einsum("nij,nj->ni", Qc, z)
        l_ref, r_ref = (torch.einsum("nji,nj->ni", Ad, wv), torch.einsum("nji,nj->ni", Bd, wv))
        l, r = proc.pair(t, ut, ut1)
        worst = max(worst, rel(l_ref, l), rel(r_ref, r))
    check("pair stencil equals the dense validated Jacobians", worst < 1e-12, f"{worst:.2e}")
    check("the sparsity pattern was READ OFF the matrices",
          len(proc.patterns["A"]) < 16 and proc.patterns["A_varying"],
          f"A blocks {proc.patterns['A']}, varying {proc.patterns['A_varying']}, "
          f"B blocks {proc.patterns['B']}, Q uniform {proc.patterns['Q_uniform']}")
    rot_res = float(proc.A[0].blocks[(1, 1)].abs().max()) if (1, 1) in proc.A[0].blocks else 0
    check("the rotation residual is genuinely nonzero (not an identity-residual case)",
          rot_res > 0 and proc.patterns["B_varying"], f"B varying {proc.patterns['B_varying']}")
    Qbad = Q.clone()
    Qbad[:, 0:3, 6:9] = Qbad[:, 6:9, 0:3] = 0
    from bidirectional_iekf.accelerated import StructuredProcess
    bad = StructuredProcess(warm, Qbad, device=torch.device("cuda"))
    lb, _ = bad.pair(0, u[0].cuda(), u[1].cuda())
    l0, _ = proc.pair(0, u[0].cuda(), u[1].cuda())
    check("negative control: dropping the pose-rate cross block of Q is caught",
          rel(l0, lb) > 1e-3, f"{rel(l0, lb):.2e} difference")

    print("\n[5] the cache is frozen per state, not per PCG iteration or lambda")
    calls = {"n": 0}
    orig = cs.obs.linearize_frame

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    cs.obs.linearize_frame = counted
    op2 = cs.build_system(warm, lam)[0]
    check("same state, same lambda: no rebuild", calls["n"] == 0, f"{calls['n']} builds")
    op3 = cs.build_system(warm, 10 * lam)[0]
    check("same state, new lambda: no rebuild of the image caches", calls["n"] == 0)
    check("but the operator does change with lambda",
          rel(op2.information_matvec(u), op3.information_matvec(u)) > 1e-6)
    before = calls["n"]
    pcg(op2, op2.rhs(), preconditioner=_eliminated(op2, op2.make_preconditioner("diagonal", lam)),
        max_iters=15, tol=1e-30)
    check("15 PCG iterations rebuild nothing", calls["n"] == before, f"{calls['n'] - before}")
    check("and stage each frame once per application",
          op2.store.stage_calls >= 15 * T)
    new_state = warm.retract(0.001 * torch.ones(T, N, 12, dtype=torch.float64))
    cs.build_system(new_state, lam)
    check("a NEW accepted state rebuilds every frame", calls["n"] - before == T,
          f"{calls['n'] - before} builds for {T} frames")
    cs.obs.linearize_frame = orig

    print("\n[6] negative controls: a stale or corrupted cache is not the operator")
    stale = cs.build_system(new_state, lam)[0]
    d_stale = rel(ref.information_matvec(u), stale.information_matvec(u))
    check("a cache from another state gives a different A u", d_stale > 100 * tol("matvec"),
          f"{d_stale:.2e}")
    op_ok = solver("exact_cached_structured").build_system(warm, lam)[0]
    pk = op_ok.store.frames[1]
    probe = ob.linearize_frame(warm.states[1], 1)
    probe.residuals = [None] * len(probe.residuals)
    tl: list = []
    acc._pack(probe, tl, {})
    pl = probe.linearization.views[0].cache.point_list
    k = next(i for i, x in enumerate(tl) if x.data_ptr() == pl.data_ptr())
    o, nb, dt_, sh = pk.layout[k]
    view = pk.flat[o:o + nb].view(dt_)
    perm = torch.randperm(view.numel(), generator=g)
    view.copy_(view[perm].clone())                # scramble camera 0's point_list
    d_bad = rel(ref.information_matvec(u), op_ok.information_matvec(u))
    check("a scrambled contributor order is detected", d_bad > 100 * tol("matvec"),
          f"{d_bad:.2e}")

    print("\n[7] synchronous and double-buffered staging agree")
    db = solver("exact_cached_structured", transfer_policy="double_buffered",
                pinned_budget_mib=64).build_system(warm, lam)[0]
    d_db = rel(op.information_matvec(u), db.information_matvec(u))
    check("double-buffered A u", d_db <= tol("matvec"), f"{d_db:.2e}")

    print("\n[8] preconditioners: M changes, A does not; same solution")
    leg_diag = _eliminated(ref, block_jacobi_preconditioner(ref, damping=lam, diagonal_only=True))
    new_diag = _eliminated(op, op.make_preconditioner("diagonal", lam))
    r_ = op._freeze(u)
    check("diagonal M^-1 matches the legacy diagonal", rel(leg_diag(r_), new_diag(r_)) <= tol("curv"),
          f"{rel(leg_diag(r_), new_diag(r_)):.2e}")
    leg_blk = _eliminated(ref, block_jacobi_preconditioner(ref, damping=lam, diagonal_only=False))
    cs_b = solver("exact_cached_structured", preconditioner="block12")
    opb = cs_b.build_system(warm, lam)[0]
    new_blk = _eliminated(opb, opb.make_preconditioner("block12", lam))
    mid = slice(1, T - 1)                          # frames with no anchored rows
    d_blk = rel(leg_blk(r_)[mid], new_blk(r_)[mid])
    d_anch = rel(leg_blk(r_)[0, A_rows], new_blk(r_)[0, A_rows])
    check("block12 M^-1 matches legacy block-Jacobi wherever nothing is anchored",
          d_blk <= 1e-6, f"{d_blk:.2e} (legacy adds a 1e-9 ridge)")
    check("on anchored blocks it restricts to the FREE coordinates before inverting",
          d_anch > 1e-6, f"{d_anch:.2e}: legacy inverts the 12x12 block including the "
          "fixed pose and zeroes it afterwards, which is a Schur complement, not the "
          "restriction; both are valid preconditioners, the restriction is the "
          "free-subspace one the spec asks for")
    temporal = _eliminated(opb, opb.make_preconditioner("temporal_block_tridiagonal", lam))

    # dense oracle: M must equal A with every CROSS-Gaussian block removed
    DIM = T * N * 12
    Adense = torch.zeros(DIM, DIM, dtype=torch.float64)
    Minv = torch.zeros(DIM, DIM, dtype=torch.float64)
    e = torch.zeros(T, N, 12, dtype=torch.float64)
    for k in range(DIM):
        e.zero_(); e.view(-1)[k] = 1.0
        Adense[:, k] = opb.information_matvec(e).reshape(-1)
        Minv[:, k] = temporal(e).reshape(-1)
    free = (opb._freeze(torch.ones(T, N, 12, dtype=torch.float64)).reshape(-1) > 0)
    gid = torch.arange(N).view(1, N, 1).expand(T, N, 12).reshape(-1)
    same_g = gid.unsqueeze(0) == gid.unsqueeze(1)
    Mtarget = (Adense * same_g)[free][:, free]
    Mf = torch.linalg.inv(Minv[free][:, free])
    d_M = rel(Mtarget, Mf)
    check("temporal M = A restricted to same-Gaussian blocks (dense oracle)",
          d_M <= 1e-5, f"{d_M:.2e}")
    ev = torch.linalg.eigvalsh(0.5 * (Minv[free][:, free] + Minv[free][:, free].T))
    check("temporal M is SPD on the free subspace", float(ev.min()) > 0,
          f"min eigenvalue of M^-1 {float(ev.min()):.3e}")
    check("M^-1 never re-introduces an eliminated coordinate",
          float(Minv[~free].abs().max()) == 0.0)
    check("A is untouched by the preconditioner choice",
          rel(op.information_matvec(u), opb.information_matvec(u)) <= tol("matvec"))

    b_ = op.rhs()
    sols = {}
    for name, pre in (("legacy diagonal", (ref, leg_diag)), ("diagonal", (op, new_diag)),
                      ("block12", (opb, new_blk)), ("temporal", (opb, temporal))):
        x, info = pcg(pre[0], b_ if pre[0] is not ref else ref.rhs(), preconditioner=pre[1],
                      max_iters=400, tol=1e-10, true_residual=True)
        sols[name] = (x, info)
        print(f"       {name:<16} {info['status']:<10} {info['iterations']:>4} iterations, "
              f"true residual {info['true_residual']:.2e}")
    x0 = sols["legacy diagonal"][0]
    for name in ("diagonal", "block12", "temporal"):
        check(f"{name}: same solution as the legacy solve",
              rel(x0, sols[name][0]) <= 1e-6 and sols[name][1]["status"] == "CONVERGED",
              f"{rel(x0, sols[name][0]):.2e}")
    check("the temporal preconditioner needs no more iterations than diagonal",
          sols["temporal"][1]["iterations"] <= sols["diagonal"][1]["iterations"],
          f"{sols['temporal'][1]['iterations']} vs {sols['diagonal'][1]['iterations']}")

    print("\n[9] A*delta reuse equals a fresh application")
    x, info = sols["temporal"]
    check("the Ax PCG returns is A applied to its own x",
          rel(info["Ax"], opb.information_matvec(x)) <= tol("matvec"))

    print("\n[10] whole nonlinear runs agree")
    runs = {}
    for name, kw in (("legacy", dict(linear_backend="legacy_reference")),
                     ("legacy2", dict(linear_backend="legacy_reference")),
                     ("cached diag", dict(linear_backend="exact_cached_structured",
                                          preconditioner="diagonal")),
                     ("cached temporal", dict(linear_backend="exact_cached_structured",
                                              preconditioner="temporal_block_tridiagonal"))):
        s_ = UnifiedFreeIEKS(FreeRGBDObservation(cams, obs_data, frame_ids=list(range(T))), Q,
                             config=IEKSConfig(source_pose_policy="fixed_conditional",
                                               anchor_groups=groups, solve_on_cpu=True,
                                               max_outer=60, damping=lam, pcg_max_iters=400,
                                               pcg_tol=1e-10, pcg_true_residual=True,
                                               verbose=False, **kw))
        runs[name] = s_.run(warm)
        print(f"       {name:<16} {runs[name]['status']:<10} E {runs[name]['energy']['E_total']:.6e}")

    def worst_mu(a, b):
        return max(float((a["trajectory"].states[t].mu - b["trajectory"].states[t].mu).abs().max())
                   for t in range(T))
    ctrl = worst_mu(runs["legacy"], runs["legacy2"])
    e_ctrl = abs(runs["legacy"]["energy"]["E_total"] - runs["legacy2"]["energy"]["E_total"])
    check("the reference itself converges", runs["legacy"]["status"] == "CONVERGED",
          runs["legacy"]["status"])
    for name in ("cached diag", "cached temporal"):
        check(f"{name}: converges", runs[name]["status"] == "CONVERGED", runs[name]["status"])
        d = worst_mu(runs["legacy"], runs[name])
        check(f"{name}: same converged states as legacy", d <= max(10 * ctrl, 1e-5),
              f"{d:.2e} (legacy-vs-legacy {ctrl:.2e})")
        e0, e1 = runs["legacy"]["energy"]["E_total"], runs[name]["energy"]["E_total"]
        check(f"{name}: same converged objective",
              abs(e0 - e1) <= max(10 * e_ctrl, 1e-6 * max(abs(e0), 1.0)),
              f"{e0:.6e} vs {e1:.6e} (legacy-vs-legacy {e_ctrl:.1e})")
        st = runs[name]["trajectory"].states
        check(f"{name}: anchors bitwise unchanged",
              torch.equal(st[0].mu[A_rows.cuda()], warm.states[0].mu[A_rows.cuda()])
              and torch.equal(st[T - 1].mu[B_rows.cuda()], warm.states[T - 1].mu[B_rows.cuda()]))

    print()
    import json
    log = REPO / "docs" / "FUSED_PHASE1_EXACT_ACCELERATION_EQUIVALENCE_tests.json"
    log.write_text(json.dumps({"floor": floor, "rows": [
        {"check": r[0], "pass": r[1], "detail": r[2]} for r in ROWS]}, indent=1))
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
