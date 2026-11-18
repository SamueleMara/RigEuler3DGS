"""The calibrated contamination negative-ray likelihood L_p = -log((1 - q_p) + q_p E[T_front])
on the REAL CUDA renderer (estimator/negative_ray_reliability.py, occlusion_likelihood.py).

    python tests/test_phase1_negative_ray_reliability.py

[0] calibration estimator: the range quantile is exact for n = 2; an injected contamination
    rate is recovered; consistent pixels near the mask get q << 1, inconsistent ones q = 1
[1] per-pixel q: JVP / energy FD on frozen-branch pixels, score == -J^T (q/E), adjoint,
    symmetric PSD normal, Fisher == diagonal blocks of the normal operator
[2] dense small-scene reference: negative-ray J^T diag((q/E)^2) J and -J^T (q/E) from dense
    JVP columns; the grouped RHS B^T b / G^T b and the reduced normal operator B^T A B of the
    production frame step (observable and legacy group maps) from dense columns
[3] static object, imperfect (eroded) mask contour, touching depth on the false-negative ring:
    certified q = 1 drives depth motion; the calibrated likelihood does not
[4] object behind a measured occluder: zero free-space pressure (calibrated)
[5] a Gaussian genuinely in front of reliable background depth keeps its correction
[6] mixed-depth silhouette pixels are UNKNOWN (q = 0), never hard background certainty
[7] visible in two cameras, hidden in the third: the hidden camera exerts no force, the
    motion is still re-acquired from the others
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CHECKS, FAILURES = 0, []
SIG = 0.004


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def cfg(names=("c0", "c1", "c2"), **kw):
    from estimator.sparse_iekf import IEKFConfig
    base = dict(max_iters=8, pcg_max_iters=40, camera_names=tuple(names), depth_sigma_model="constant",
                depth_sigma_a=SIG, use_supernodes=True, supernode_target_size=16)
    base.update(kw)
    return IEKFConfig(**base)


def model():
    from estimator.masked_observation import MaskedObservationModel
    return MaskedObservationModel("masked-rgb-expected-quad", (0.5, 0.5, 0.5), "uniform01", 0.1)


def erode(M, k):
    return ~(torch.nn.functional.max_pool2d((~M).float()[None, None], 2 * k + 1, 1, k)[0, 0] > 0)


def P0(n):
    P = torch.zeros((n, 12, 12), device="cuda")
    P[:, range(12), range(12)] = torch.tensor([1e-4] * 3 + [1e-3] * 3 + [1.0] * 3 + [0.1] * 3, device="cuda")
    return P


def calibration_for(frames_per_cam, c, names):
    """frames_per_cam[cam] = list of PreparedRGBD; calibrated with the run's own sigma map."""
    from estimator import negative_ray_reliability as nrr
    from estimator.occlusion_likelihood import depth_sigma_map
    out = {}
    for name in names:
        fr = []
        for ob in frames_per_cam[name]:
            raw = ob.depth_valid_raw
            fr.append((ob.object_coverage > 0.5, ob.depth_observed, raw, depth_sigma_map(c, name, ob.depth_observed, raw)))
        out[name] = nrr.calibrate(name, fr)
    return out


def make_obs(views, masks, depths, noise_seed, names):
    from estimator.masked_observation import prepare
    g = torch.Generator(device="cuda").manual_seed(noise_seed)
    out = []
    for k, (v, M, d) in enumerate(zip(views, masks, depths)):
        dn = d + SIG * torch.randn(d.shape, device="cuda", generator=g)
        out.append(prepare(v.rgb + (1 - v.alpha)[None] * 0.3, dn, dn > 0, M, names[k], noise_seed))
    return out


def run_seq(state, frames, cams, c, occ, group="observable"):
    from estimator.sparse_iekf import iekf_frame_step
    c = dataclasses.replace(c, group_update=group)
    x, P = state, P0(state.mu.shape[0])
    for ob in frames:
        x, P, _ = iekf_frame_step(x, P, ob, cams, 1 / 30, c, observation_model=model(), occlusion=occ)
    return x


def axes_of(lin):
    return torch.stack([(lambda a: a / a.norm())(v.camera.viewmatrix.double()[:3, 2]) for v in lin.views])


def rand_q_model(base):
    from estimator.occlusion_likelihood import OcclusionLikelihood

    class _R(OcclusionLikelihood):
        def pixel_model(self, ob, camera, sigma):
            st, q = super().pixel_model(ob, camera, sigma)
            g = torch.Generator(device=q.device).manual_seed(sum(map(ord, camera)))
            return st, q * torch.rand(q.shape, generator=g, device=q.device)
    return _R(base)


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from estimator import negative_ray_operators as no
    from estimator import negative_ray_reliability as nrr
    from estimator.occlusion_likelihood import OcclusionLikelihood, evaluate
    from estimator.renderer_bridge import build_linearization
    from estimator.state import boxplus
    from tests.test_phase1_occlusion_likelihood import blob, cat
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    torch.manual_seed(0)
    names = ("c0", "c1", "c2")

    # ------------------------------------------------------------------ [0]
    print("\n[0] calibration estimator")
    z = nrr._norm_ppf(1 - nrr.LEVEL / 2)
    check("[0] range quantile n=2 == sqrt(2) z_(1-a/2) (exact law of |X1 - X2|)",
          abs(nrr.range_quantile(2) - math.sqrt(2) * z) < 1e-3, f"{nrr.range_quantile(2):.4f} vs {math.sqrt(2) * z:.4f}")
    H = W = 160
    yy, xx = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij")
    r = ((yy - 80.0) ** 2 + (xx - 80.0) ** 2).sqrt()
    M = r <= 25
    g = torch.Generator(device="cuda").manual_seed(0)
    rho = 0.6
    frames = []
    for t in range(6):
        D = torch.full((H, W), 3.0, device="cuda")              # far wall
        D[M] = 1.0
        ring = (r > 25) & (r <= 29)
        hit = ring & (torch.rand((H, W), generator=g, device="cuda") < rho)
        D[hit] = 1.0                                            # mask false negatives: object depth
        D = D + 0.002 * torch.randn((H, W), generator=g, device="cuda")
        frames.append((M, D, torch.ones_like(M), torch.full((H, W), 0.002, device="cuda", dtype=torch.float64)))
    cal = nrr.calibrate("synthetic", frames)
    cmid = float(cal.contamination[3:5].mean())
    check("[0] injected contamination rate recovered on the ring (|c - 0.6| < 0.1)", abs(cmid - rho) < 0.1,
          f"c(3..4) = {cmid:.3f}, r_inf = {cal.r_inf:.4f}, d_far = {cal.d_far}")
    check("[0] no contamination far from the mask", float(cal.contamination[12:cal.d_far].max(initial=0.0)) < 0.02)
    f = nrr.features(*frames[0][:3], frames[0][3])
    q = cal.q_map(f)
    ringc = f.consistent & (f.distance > 2) & (f.distance <= 4)
    check("[0] consistent ring pixels get q < 0.05, inconsistent candidates q == 1",
          float(q[ringc].max()) < 0.05 and bool((q[f.candidate & ~f.consistent] == 1).all()),
          f"max q on consistent ring {float(q[ringc].max()):.3g}")

    # ------------------------------------------------------------------ scenes
    cams = make_cameras(n_cam=3, H=96, W=96, radius=2.2)
    c = cfg()
    obj = blob(n=60, seed=0, spread=0.3)
    N_obj = obj.mu.shape[0]
    lin0 = build_linearization(obj, cams)
    axes = axes_of(lin0)

    # ------------------------------------------------------------------ [1]
    print("\n[1] per-pixel q: derivatives at frozen contributor order")
    stray = blob(n=6, seed=3, centre=(0.33, 0.05, 0.0), spread=0.08)
    scene = cat(obj, stray)
    N = scene.mu.shape[0]
    masks = [v.alpha > 0.5 for v in lin0.views]
    depths = [torch.where(m, v.depth, torch.full_like(v.depth, 4.0)) for m, v in zip(masks, lin0.views)]
    obs = make_obs(lin0.views, masks, depths, 1, names)
    occ = rand_q_model("certified")
    ev, lin = evaluate(occ, build_linearization(scene, cams), obs, c), None
    from estimator import first_return_operators as fo

    def composited(state):
        L_ = build_linearization(state, cams)
        out = []
        for k, vv in enumerate(L_.views):
            st_ = ev.cams[k].status.reshape(-1).cpu()
            fa = fo.camera_args(vv.cache, torch.zeros_like(vv.depth), torch.zeros_like(ev.cams[k].status),
                                torch.ones_like(vv.depth), torch.ones_like(vv.depth), 0.0, 0.01, 100.0)
            dmp = fo.fr_dump(fa)
            off, gid, al = dmp["offsets"].cpu(), dmp["gid"].cpu(), dmp["alpha"].cpu()
            m = {}
            for p in torch.nonzero(st_ == no.ST_NONOBJECT).reshape(-1).tolist():
                T, ids = torch.tensor(1.0), []
                for j in range(int(off[p]), int(off[p + 1])):
                    tt = T * (1 - al[j])
                    if float(tt) < 1e-4:
                        break
                    ids.append(int(gid[j])); T = tt
                m[p] = tuple(ids)
            out.append(m)
        return out
    gg = torch.Generator(device="cuda").manual_seed(1)
    v = torch.randn((N, 6), device="cuda", generator=gg)
    v[:, 3:] *= 0.5
    eps = 1e-4
    d12 = torch.zeros((N, 12), device="cuda"); d12[:, :6] = eps * v
    xp, xm = boxplus(scene, d12), boxplus(scene, -d12)
    Ep = evaluate(occ, build_linearization(xp, cams), obs, c)
    Em = evaluate(occ, build_linearization(xm, cams), obs, c)
    cp_, cm_ = composited(xp), composited(xm)
    num = den = e_fd = e_pred = 0.0
    nfroz = 0
    qs = []
    for k, (cp, cm, cam) in enumerate(zip(Ep.cams, Em.cams, ev.cams)):
        qmap = cam.args[20].double()
        qs.append(qmap[cam.status == no.ST_NONOBJECT])
        Hh, Ww = cam.A.shape
        frozen = torch.zeros(Hh * Ww, dtype=torch.bool)
        for p, ids in cp_[k].items():
            frozen[p] = ids == cm_[k][p]
            nfroz += int(frozen[p])
        frozen = frozen.reshape(Hh, Ww).cuda()
        fd = (cp.A - cm.A) / (2 * eps)
        jv = no.nr_jvp(cam.args, cam.A, v).double()
        num += float((fd - jv)[frozen].square().sum()); den += float(fd[frozen].square().sum())
        Lp = -torch.log((1 - qmap * cp.A).clamp_min(1e-12)); Lm = -torch.log((1 - qmap * cm.A).clamp_min(1e-12))
        e_fd += float(((Lp - Lm) / (2 * eps))[frozen].sum())
        e_pred += float((cam.c1.double() * jv)[frozen].sum())
    qs = torch.cat(qs)
    check("[1] the q map is genuinely per-pixel (spread over (0, 1))", float(qs.std()) > 0.2 and float(qs.min()) < 0.1,
          f"q mean {float(qs.mean()):.2f} sd {float(qs.std()):.2f}")
    check("[1] JVP vs FD of A on frozen-branch pixels (rel L2 < 1e-2)", math.sqrt(num / max(den, 1e-30)) < 1e-2,
          f"{math.sqrt(num / max(den, 1e-30)):.2e} over {nfroz} pixels")
    check("[1] energy FD of -log(1 - qA) vs (q/E) JVP on frozen-branch pixels (rel < 1e-2)",
          abs(e_fd - e_pred) / max(abs(e_pred), 1e-12) < 1e-2, f"FD {e_fd:.5f} vs {e_pred:.5f}")
    score = ev.score_pose()
    pred = -float((score * v).sum())
    sumd = sum(float((cm.c1.double() * no.nr_jvp(cm.args, cm.A, v).double()).sum()) for cm in ev.cams)
    check("[1] score == -J^T (q/E) (rel < 1e-4)", abs(pred - sumd) / max(abs(pred), 1e-12) < 1e-4, f"{pred:.5f} vs {sumd:.5f}")
    lhs = rhs = 0.0
    for cam in ev.cams:
        w = torch.randn(cam.A.shape, device="cuda", generator=gg) * (cam.status == no.ST_NONOBJECT)
        lhs += float((no.nr_jvp(cam.args, cam.A, v).double() * w).sum())
        rhs += float((no.nr_vjp(cam.args, cam.A, w.float()).double() * v.double()).sum())
    check("[1] adjoint <Jv,w> == <v,J^T w> (rel < 1e-4)", abs(lhs - rhs) / max(abs(lhs), 1e-30) < 1e-4)
    u = torch.randn((N, 6), device="cuda", generator=gg)
    uNv, vNu, vNv = (float((u * ev.normal_pose(v)).sum()), float((v * ev.normal_pose(u)).sum()),
                     float((v * ev.normal_pose(v)).sum()))
    check("[1] normal operator symmetric PSD", abs(uNv - vNu) / max(abs(uNv), 1e-30) < 1e-4 and vNv >= 0)
    F = ev.fisher_pose()
    act = torch.nonzero(F.diagonal(dim1=1, dim2=2).sum(-1) > 0).reshape(-1)[:4].tolist()
    worst = 0.0
    for i in act:
        for b in range(6):
            e = torch.zeros((N, 6), device="cuda"); e[i, b] = 1.0
            worst = max(worst, float((ev.normal_pose(e)[i] - F[i, :, b]).abs().max()) / max(float(F[i].abs().max()), 1e-30))
    check("[1] Fisher blocks == diagonal blocks of the normal operator (rel < 1e-3)", len(act) > 0 and worst < 1e-3,
          f"worst {worst:.1e}")

    # ------------------------------------------------------------------ [2]
    print("\n[2] dense small-scene reference: negative-ray operators, grouped RHS and reduced normal operator")
    cams_s = make_cameras(n_cam=2, H=40, W=40, radius=2.2)
    small = blob(n=10, seed=5, spread=0.25)
    strays = blob(n=4, seed=6, centre=(0.25, 0.0, 0.0), spread=0.05)
    sc = cat(small, strays)
    Ns = sc.mu.shape[0]
    ls = build_linearization(small, cams_s)
    ms = [vv.alpha > 0.5 for vv in ls.views]
    obs_s = make_obs(ls.views, ms, [torch.where(m, vv.depth, torch.full_like(vv.depth, 4.0)) for m, vv in zip(ms, ls.views)],
                     2, ("c0", "c1"))
    cs = cfg(("c0", "c1"), supernode_target_size=4, max_iters=1)
    occ_s = rand_q_model("certified")
    res = {}

    def observer(*, iteration, x, x_prior, Lambda_prior, lin, meas_normal, b, grouped, lm, solve_b=None,
                 observable_map=None, expand_step=None, **kw):
        if iteration != 0 or "done" in res:
            return
        from estimator.observable_group import operators
        from estimator.sparse_iekf import apply_prior
        POSE_DIM = 6
        nr = lin.negative_ray_evaluation
        cols = []
        for j in range(6 * Ns):
            e = torch.zeros((Ns, 6), device="cuda"); e.view(-1)[j] = 1.0
            cols.append(torch.cat([no.nr_jvp(cm.args, cm.A, e).double().reshape(-1) for cm in nr.cams]))
        J = torch.stack(cols, 1)
        c1 = torch.cat([cm.c1.double().reshape(-1) for cm in nr.cams])
        c2 = torch.cat([cm.c2.double().reshape(-1) for cm in nr.cams])
        Nd = J.T @ (c2[:, None] * J)
        vv = torch.randn((Ns, 6), device="cuda", dtype=torch.float64)
        res["nr_normal"] = float((nr.normal_pose(vv.float()).double().reshape(-1) - Nd @ vv.reshape(-1)).norm()
                                 / (Nd @ vv.reshape(-1)).norm())
        res["nr_score"] = float((nr.score_pose().double().reshape(-1) + J.T @ c1).norm() / (J.T @ c1).norm())

        def A_mv(y):
            meas = torch.zeros_like(y)
            meas[:, :POSE_DIM] = meas_normal(y[:, :POSE_DIM])
            return apply_prior(Lambda_prior, y) + meas + lm * y
        red_dim = solve_b.numel()
        Bc = []
        for j in range(red_dim):
            e = torch.zeros_like(solve_b); e.view(-1)[j] = 1.0
            Bc.append(expand_step(e).double().reshape(-1))
        Bd = torch.stack(Bc, 1)
        res["grouped_rhs"] = float((solve_b.double().reshape(-1) - Bd.T @ b.double().reshape(-1)).norm()
                                   / (Bd.T @ b.double().reshape(-1)).norm())
        Ac = []
        for j in range(12 * Ns):
            e = torch.zeros((Ns, 12), device="cuda"); e.view(-1)[j] = 1.0
            Ac.append(A_mv(e).double().reshape(-1))
        Ad = torch.stack(Ac, 1)
        Ar = Bd.T @ Ad @ Bd
        uu = torch.randn_like(solve_b)
        if observable_map is not None:
            Bop, BTop = operators(observable_map, grouped)
        else:
            Bop = expand_step

            def BTop(y):
                return torch.cat([grouped.reduce(y[:, :6].float().contiguous()),
                                  grouped.reduce(y[:, 6:].float().contiguous())], 1)
        got = BTop(A_mv(Bop(uu))).double().reshape(-1)
        res["reduced_normal"] = float((got - Ar @ uu.double().reshape(-1)).norm() / (Ar @ uu.double().reshape(-1)).norm())
        res["reduced_symmetric"] = float((Ar - Ar.T).abs().max() / Ar.abs().max())
        res["done"] = True
    from estimator.sparse_iekf import iekf_frame_step
    for grp in ("observable", "legacy"):
        res.clear()
        iekf_frame_step(sc, P0(Ns), obs_s, cams_s, 1 / 30, dataclasses.replace(cs, group_update=grp),
                        observation_model=model(), occlusion=occ_s, iteration_observer=observer)
        if grp == "observable":
            check("[2] negative-ray normal == dense J^T diag((q/E)^2) J (rel < 1e-4)", res["nr_normal"] < 1e-4,
                  f"{res['nr_normal']:.1e}")
            check("[2] negative-ray score == dense -J^T (q/E) (rel < 1e-4)", res["nr_score"] < 1e-4, f"{res['nr_score']:.1e}")
        check(f"[2] {grp}: grouped RHS == dense B^T b (rel < 1e-4)", res["grouped_rhs"] < 1e-4, f"{res['grouped_rhs']:.1e}")
        check(f"[2] {grp}: reduced normal operator == dense B^T A B (rel < 1e-4), symmetric",
              res["reduced_normal"] < 1e-4 and res["reduced_symmetric"] < 1e-5,
              f"{res['reduced_normal']:.1e}, asym {res['reduced_symmetric']:.1e}")

    # ------------------------------------------------------------------ [3]
    print("\n[3] static object, eroded mask contour with object depth on the false-negative ring")
    # The eroded ring (3 px) is object surface labelled M = 0 (mask false negatives, depth = the
    # object's own). Removing it from M also removes positive evidence, which moves the static
    # object by itself; the arm with NO negative evidence (q = 0) measures that floor, and the
    # perfect-mask arms the floor without contamination. The quantity under test is the depth
    # motion the negative-ray term ADDS on top of q = 0.
    depths3 = [torch.where(v.alpha > 0.5, v.depth, torch.full_like(v.depth, 4.0)) for v in lin0.views]

    def depth_motion(x):
        dmu = (x.mu - obj.mu).double()
        return 1e3 * float((dmu @ axes.T).abs().mean())          # mean |along each camera axis|, mm
    moved = {}
    for contour, masks3 in (("perfect", [v.alpha > 0.5 for v in lin0.views]),
                            ("eroded", [erode(v.alpha > 0.5, 3) for v in lin0.views])):
        frames3 = [make_obs(lin0.views, masks3, depths3, 100 + t, names) for t in range(8)]
        cal3 = calibration_for({nm: [fr[k] for fr in frames3] for k, nm in enumerate(names)}, c, names)
        for arm, o in (("none", OcclusionLikelihood("none")), ("certified", OcclusionLikelihood("certified")),
                       ("calibrated", OcclusionLikelihood("calibrated", calibration=cal3))):
            moved[(contour, arm)] = depth_motion(run_seq(obj, frames3, cams, c, o))
        print(f"      {contour:7s}: depth motion (mm) q=0 {moved[(contour, 'none')]:.3f}, certified "
              f"{moved[(contour, 'certified')]:.3f}, calibrated {moved[(contour, 'calibrated')]:.3f}")
    add_k = moved[("eroded", "certified")] - moved[("eroded", "none")]
    add_c = moved[("eroded", "calibrated")] - moved[("eroded", "none")]
    check("[3] certified q = 1 adds depth motion on the imperfect contour (> 0.3 mm over q = 0)", add_k > 0.3,
          f"+{add_k:.3f} mm")
    check("[3] calibrated adds no artificial depth motion (< 10% of certified's increment, < 0.1 mm)",
          add_c < 0.1 * add_k and add_c < 0.1, f"{add_c:+.3f} mm vs certified +{add_k:.3f} mm")
    check("[3] perfect contour: calibrated keeps the certified evidence (depth motion within 5%)",
          abs(moved[("perfect", "calibrated")] - moved[("perfect", "certified")])
          <= 0.05 * moved[("perfect", "certified")], f"{moved[('perfect', 'calibrated')]:.3f} vs "
          f"{moved[('perfect', 'certified')]:.3f} mm")
    print("      residual (not the negative ray): the eroded mask removes positive evidence on the ring; "
          f"q = 0 moves {moved[('eroded', 'none')]:.2f} mm vs {moved[('perfect', 'none')]:.2f} mm with the perfect mask")

    # ------------------------------------------------------------------ [4]
    print("\n[4] object behind a measured occluder (calibrated)")
    A0 = blob(n=40, seed=0, spread=0.3)
    Bh = blob(n=12, seed=4, centre=(0.0, 0.45, 0.0), spread=0.1)
    nA = A0.mu.shape[0]
    canon = cat(A0, Bh)
    la, lb = build_linearization(A0, cams), build_linearization(Bh, cams)
    m4, d4 = [], []
    for va, vb in zip(la.views, lb.views):
        M = va.alpha > 0.5
        foot = vb.alpha > 1e-7
        occ_depth = float(vb.depth[vb.alpha > 0.5].min()) - 0.3
        d = torch.where(M, va.depth, torch.full_like(va.depth, 4.0))
        d = torch.where(foot & ~M, torch.full_like(d, occ_depth), d)
        m4.append(M); d4.append(d)
    fr4 = [make_obs(la.views, m4, d4, 200 + t, names) for t in range(4)]
    cal4 = calibration_for({nm: [fr[k] for fr in fr4] for k, nm in enumerate(names)}, c, names)
    e4 = evaluate(OcclusionLikelihood("calibrated", calibration=cal4), build_linearization(canon, cams), fr4[0], c)
    s4 = e4.score_pose()
    check("[4] hidden Gaussians: exactly zero negative-ray score", float(s4[nA:].abs().max()) == 0.0,
          f"max |score| hidden {float(s4[nA:].abs().max()):.1e}, visible {float(s4[:nA].abs().max()):.1e}")
    F4 = e4.fisher_pose()
    check("[4] hidden Gaussians: zero negative-ray Fisher", float(F4[nA:].abs().max()) == 0.0)

    # ------------------------------------------------------------------ [5]
    print("\n[5] Gaussian in front of reliable background depth (calibrated)")
    # an isolated Gaussian group far from the object, on free-space rays only: the negative ray
    # (and the prior) are the only terms acting on it
    iso = blob(n=4, seed=7, centre=(0.0, 0.42, 0.0), spread=0.04)
    scene5 = cat(obj, iso)
    N5 = scene5.mu.shape[0]
    fr5 = [make_obs(lin0.views, masks, depths, 300 + t, names) for t in range(4)]
    cal5 = calibration_for({nm: [fr[k] for fr in fr5] for k, nm in enumerate(names)}, c, names)
    occ5 = OcclusionLikelihood("calibrated", calibration=cal5)
    li5 = build_linearization(iso, cams)
    e5 = evaluate(occ5, build_linearization(scene5, cams), fr5[0], c)
    on_m = sum(int(((vv.alpha > 1e-3) & (fr5[0][k].object_coverage > 0.5)).sum()) for k, vv in enumerate(li5.views))
    qfoot = torch.cat([cm.args[20][(vv.alpha > 0.05) & (cm.status == no.ST_NONOBJECT)] for cm, vv in zip(e5.cams, li5.views)])
    check("[5] the isolated Gaussians touch no M = 1 pixel and have q == 1 on their free-space pixels",
          on_m == 0 and qfoot.numel() > 0 and float(qfoot.min()) > 0.99, f"{qfoot.numel()} pixels, min q {float(qfoot.min()):.3f}")
    from estimator.sparse_iekf import iekf_frame_step as step
    c5 = dataclasses.replace(c, group_update="observable", use_supernodes=False)

    def iso_energy(x):
        st = dataclasses.replace(scene5, mu=torch.cat([scene5.mu[:N_obj], x.mu[N_obj:]]),
                                 R=torch.cat([scene5.R[:N_obj], x.R[N_obj:]]))
        return evaluate(occ5, build_linearization(st, cams), fr5[0], c).energy
    x5, _, _ = step(scene5, P0(N5), fr5[0], cams, 1 / 30, c5, observation_model=model(), occlusion=occ5)
    x0_, _, _ = step(scene5, P0(N5), fr5[0], cams, 1 / 30, c5, observation_model=model(),
                     occlusion=OcclusionLikelihood("none"))
    mv5 = 1e3 * float((x5.mu[N_obj:] - scene5.mu[N_obj:]).norm(dim=1).mean())
    mv0 = 1e3 * float((x0_.mu[N_obj:] - scene5.mu[N_obj:]).norm(dim=1).mean())
    from estimator.state import boxminus
    dpose = boxminus(x5, scene5)[N_obj:, :6].double()
    sc5 = e5.score_pose()[N_obj:].double()
    cosd = float((sc5 * dpose).sum() / (sc5.norm() * dpose.norm()).clamp_min(1e-30))
    check("[5] calibrated: their correction is a descent direction of the negative-ray energy (<score, step> > 0)",
          mv5 > 0.1 and cosd > 0, f"moved {mv5:.2f} mm, cos {cosd:.3f}; their energy {e5.energy:.3f} -> "
          f"{iso_energy(x5):.3f} (flat: a far wall gives Phi = 1, only the footprint term remains)")
    check("[5] with q = 0 there is no evidence and no correction (they stay at the prediction)", mv0 < 1e-3,
          f"moved {mv0:.4f} mm")

    # ------------------------------------------------------------------ [6]
    print("\n[6] mixed-depth silhouette pixels")
    m6, d6 = [], []
    from scipy.ndimage import distance_transform_edt
    for vv in lin0.views:
        M = vv.alpha > 0.5
        dist, (iy, ix) = distance_transform_edt(~M.cpu().numpy(), return_indices=True)
        dist = torch.from_numpy(dist).cuda()
        Dm = vv.depth[torch.from_numpy(iy).cuda(), torch.from_numpy(ix).cuda()]
        d = torch.where(M, vv.depth, torch.full_like(vv.depth, 4.0))
        ring = (~M) & (dist <= 2.5)
        d = torch.where(ring, 0.5 * (Dm + 4.0), d)               # flying pixels between object and wall
        m6.append(M); d6.append(d)
    fr6 = [make_obs(lin0.views, m6, d6, 400 + t, names) for t in range(4)]
    cal6 = calibration_for({nm: [fr[k] for fr in fr6] for k, nm in enumerate(names)}, c, names)
    e6c = evaluate(OcclusionLikelihood("calibrated", calibration=cal6), build_linearization(obj, cams), fr6[0], c)
    e6k = evaluate(OcclusionLikelihood("certified"), build_linearization(obj, cams), fr6[0], c)
    mixq_c, mixq_k = [], []
    for k, vv in enumerate(lin0.views):
        M = vv.alpha > 0.5
        dist = torch.from_numpy(distance_transform_edt(~M.cpu().numpy())).cuda()
        ring = (~M) & (dist > 1.5) & (dist <= 2.5)              # outside the 3x3 mask edge
        mixq_c.append(e6c.cams[k].args[20][ring]); mixq_k.append(e6k.cams[k].args[20][ring])
    mixq_c, mixq_k = torch.cat(mixq_c), torch.cat(mixq_k)
    check("[6] certified gives the mixed pixels hard certainty (q = 1) -- the defect", float(mixq_k.min()) == 1.0)
    check("[6] calibrated: every mixed pixel is UNKNOWN (q = 0)", mixq_c.numel() > 0 and float(mixq_c.max()) == 0.0,
          f"{mixq_c.numel()} pixels, max q {float(mixq_c.max()):.3f}")

    # ------------------------------------------------------------------ [7]
    print("\n[7] visible in two cameras, hidden behind an occluder in the third")
    shift = torch.tensor([0.008, 0.0, 0.004], device="cuda")
    truth7 = dataclasses.replace(obj, mu=obj.mu + shift)
    lt = build_linearization(truth7, cams)
    m7, d7 = [], []
    for k, vv in enumerate(lt.views):
        M = vv.alpha > 0.5
        d = torch.where(M, vv.depth, torch.full_like(vv.depth, 4.0))
        if k == 2:
            foot = vv.alpha > 1e-7
            d = torch.where(foot, torch.full_like(d, float(vv.depth[M].min()) - 0.3), d)
            M = torch.zeros_like(M)
        m7.append(M); d7.append(d)
    fr7 = [make_obs(lt.views, m7, d7, 500 + t, names) for t in range(4)]
    cal7 = calibration_for({nm: [fr[k] for fr in fr7] for k, nm in enumerate(names)}, c, names)
    check("[7] a camera that never sees the object calibrates to far field (no contamination)",
          float(cal7["c2"].contamination.max(initial=0.0)) == 0.0)
    occ7 = OcclusionLikelihood("calibrated", calibration=cal7)
    e7 = evaluate(occ7, build_linearization(obj, cams), fr7[0], c)
    by = dict(e7.score_pose_by_camera())
    check("[7] the hidden camera exerts exactly zero negative-ray force on the object", float(by["c2"].abs().max()) == 0.0,
          f"c2 {float(by['c2'].abs().max()):.1e}")
    x7, _, _ = step(obj, P0(N_obj), fr7[0], cams, 1 / 30, dataclasses.replace(c, group_update="observable"),
                    observation_model=model(), occlusion=occ7)
    err0 = float((obj.mu - truth7.mu).norm(dim=1).mean())
    err1 = float((x7.mu - truth7.mu).norm(dim=1).mean())
    check("[7] the motion is re-acquired from the visible cameras (error at least halved)", err1 < 0.5 * err0,
          f"{1e3 * err0:.2f} -> {1e3 * err1:.2f} mm")

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed" + ("" if not FAILURES else f"; FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
