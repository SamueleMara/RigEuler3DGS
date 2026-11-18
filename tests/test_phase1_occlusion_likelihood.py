"""Gate I.C: the occlusion-consistent negative-ray likelihood on the REAL CUDA renderer.

RigEuler3DGS_PHASE1_COHERENCE_CODEX_IMPLEMENTATION_SPEC_2026-11-16.md section I.C.

    python tests/test_phase1_occlusion_likelihood.py

[1] A_p of the kernel == a CPU reference over the dumped contributor lists (T_STOP applied)
[2] dA/d alpha and dA/d z of the formula == central finite differences (frozen branches, CPU)
[3] end-to-end: nonlinear E(x +- eps v) vs the score; JVP vs FD of A; adjoint identity;
    normal operator symmetric PSD; Fisher blocks == diagonal blocks of the normal operator
[4] an object BEHIND a sharp front occluder (M = 0): zero term, zero score, zero JVP;
    the effect decreases monotonically as the separation grows
[5] the same Gaussian moved IN FRONT of measured non-object depth raises the NLL
[6] invalid depth on M = 0: no term, no fallback to the full-ray alpha; RGB and alpha
    precisions are exactly zero off M = 1
[7] the negative-ray energy depends on geometry / opacity, not on colour
[8] 3 cameras: hidden by an occluder in one camera, seen in the others -> nonzero Fisher
[9] deterministic reductions are bitwise repeatable
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


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def blob(n=40, seed=0, centre=(0.0, 0.0, 0.0), colour=None, spread=0.3, opacity=0.85, scale=0.035):
    from estimator.state import GaussianMotionState
    g = torch.Generator().manual_seed(seed)
    mu = (torch.rand(n, 3, generator=g) - 0.5) * spread + torch.tensor(centre)
    col = torch.rand(n, 3, generator=g) * 0.6 + 0.2 if colour is None else torch.tensor(colour).expand(n, 3)
    z = torch.zeros(n, 3)
    return GaussianMotionState(mu=mu.cuda(), R=torch.eye(3).repeat(n, 1, 1).cuda(), v=z.cuda(), omega=z.cuda(),
                               colors=col.contiguous().float().cuda(), opacity=torch.full((n, 1), opacity).cuda(),
                               scales=torch.full((n, 3), scale).cuda())


def cat(*states):
    return dataclasses.replace(states[0], **{f: torch.cat([getattr(s, f) for s in states]).contiguous()
                                             for f in ("mu", "R", "v", "omega", "colors", "opacity", "scales")})


def prepared(rgb, depth, valid, mask, cam):
    from estimator.masked_observation import prepare
    return prepare(rgb, depth, valid, mask.bool(), cam, 0)


def model():
    from estimator.masked_observation import MaskedObservationModel
    return MaskedObservationModel("masked-rgb-expected-quad", (0.5, 0.5, 0.5), "uniform01", 0.1)


def cfg():
    from estimator.sparse_iekf import IEKFConfig
    return IEKFConfig(max_iters=8, pcg_max_iters=30, camera_names=("c0", "c1", "c2"),
                      depth_sigma_model="constant", depth_sigma_a=0.02)


def evaluate(state, cams, obs, occ, c):
    from estimator.occlusion_likelihood import evaluate as ev
    from estimator.renderer_bridge import build_linearization
    lin = build_linearization(state, cams, background=torch.full((3,), 0.5, device="cuda"))
    return ev(occ, lin, obs, c), lin


def scene_free_space(state, cams, wall=4.0, carve=None):
    """Mask = silhouette of `state` (optionally minus `carve`'s silhouette); depth = the
    object's own depth on M = 1 and a wall far behind elsewhere (free space)."""
    from estimator.renderer_bridge import build_linearization
    lin = build_linearization(state, cams)
    lc = build_linearization(carve, cams) if carve is not None else None
    out = []
    for k, v in enumerate(lin.views):
        M = v.alpha > 0.5
        if lc is not None:
            M = M & ~(lc.views[k].alpha > 0.5)
        d = torch.where(M, v.depth, torch.full_like(v.depth, wall))
        out.append(prepared(v.rgb + (1 - v.alpha)[None] * 0.3, d, d > 0, M, f"c{k}"))
    return out


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from estimator import negative_ray_operators as no
    from estimator.occlusion_likelihood import OcclusionLikelihood, positive_camera_terms, ray_status
    from estimator.renderer_bridge import build_linearization
    from estimator.state import boxplus
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    torch.manual_seed(0)
    cams = make_cameras(n_cam=3, H=96, W=96, radius=2.2)
    occ = OcclusionLikelihood("certified")
    c = cfg()
    obj = blob()
    stray = blob(n=6, seed=3, centre=(0.33, 0.05, 0.0), spread=0.08)
    scene = cat(obj, stray)
    obs = scene_free_space(obj, cams)               # the stray Gaussians sit on free-space rays
    N = scene.mu.shape[0]

    print(f"\n[0] extension {no.source_sha()[:12]} built from {no.SOURCE.name}")
    ev, lin = evaluate(scene, cams, obs, occ, c)
    counts = ev.status_counts()
    check("[0] statuses present in every camera", all(x["KNOWN_NONOBJECT_DEPTH"] > 0 and x["OBJECT_SUPPORTED"] > 0
                                                       and x["UNKNOWN"] > 0 for x in counts), str(counts[0]))
    check("[0] stray Gaussians on free-space rays carry a positive energy", ev.energy > 0.1, f"E = {ev.energy:.4f}")
    check("[0] no log-floor use", all(x["log_floor_hits"] == 0 for x in counts))

    print("\n[1] kernel A_p == CPU reference over the dumped contributor lists")
    from estimator import first_return_operators as fo
    worst = 0.0
    for k, (v, cam) in enumerate(zip(lin.views, ev.cams)):
        D, sig, st = cam.args[17], cam.args[18], cam.args[19]
        H, W = D.shape
        fa = fo.camera_args(v.cache, torch.zeros_like(D), torch.zeros_like(st), torch.ones_like(D),
                            torch.ones_like(D), 0.0, 0.01, 100.0)
        dump = fo.fr_dump(fa)
        off, al, zs = dump["offsets"].cpu(), dump["alpha"].cpu().double(), dump["z"].cpu().double()
        Dc, sc, stc = D.cpu().double().reshape(-1), sig.cpu().double().reshape(-1), st.cpu().reshape(-1)
        Aref = torch.zeros(H * W, dtype=torch.float64)
        for p in torch.nonzero(stc == no.ST_NONOBJECT).reshape(-1).tolist():
            Tf, Td, A = 1.0, 1.0, 0.0
            for j in range(int(off[p]), int(off[p + 1])):
                a = float(al[j])
                tt = torch.tensor(Tf, dtype=torch.float32) * (1 - torch.tensor(a, dtype=torch.float32))
                if float(tt) < 1e-4:
                    break
                s = 0.5 * math.erfc(-((float(Dc[p]) - float(zs[j])) / float(sc[p])) / math.sqrt(2))
                A += Td * a * s
                Td *= 1 - a
                Tf = float(tt)
            Aref[p] = A
        worst = max(worst, float((Aref - cam.A.cpu().reshape(-1)).abs().max()))
    check("[1] max |A_kernel - A_reference| < 1e-6", worst < 1e-6, f"{worst:.2e}")

    print("\n[2] dA/d alpha, dA/d z of the formula vs finite differences (frozen branches)")
    a = torch.tensor([0.3, 0.5, 0.2, 0.7], dtype=torch.float64)
    z = torch.tensor([1.9, 2.0, 2.05, 2.3], dtype=torch.float64)
    D0, s0 = 2.02, 0.03

    def Af(a, z):
        T, A = 1.0, 0.0
        for k in range(len(a)):
            A = A + T * a[k] * 0.5 * torch.erfc(-((D0 - z[k]) / s0) / math.sqrt(2))
            T = T * (1 - a[k])
        return A
    worst = 0.0
    for k in range(4):
        e = torch.zeros(4, dtype=torch.float64); e[k] = 1e-6
        fd_a = (Af(a + e, z) - Af(a - e, z)) / 2e-6
        fd_z = (Af(a, z + e) - Af(a, z - e)) / 2e-6
        T = float(torch.prod(1 - a[:k]))
        s = 0.5 * math.erfc(-((D0 - float(z[k])) / s0) / math.sqrt(2))
        w = [float(torch.prod(1 - a[:m])) * float(a[m]) * 0.5 * math.erfc(-((D0 - float(z[m])) / s0) / math.sqrt(2))
             for m in range(4)]
        pa = T * s - sum(w[k + 1:]) / (1 - float(a[k]))
        pz = -T * float(a[k]) * math.exp(-0.5 * ((D0 - float(z[k])) / s0) ** 2) / math.sqrt(2 * math.pi) / s0
        worst = max(worst, abs(pa - float(fd_a)), abs(pz - float(fd_z)))
    check("[2] analytic partials == FD within 1e-7", worst < 1e-7, f"{worst:.2e}")

    print("\n[3] end-to-end derivatives of the negative-ray term")
    g = torch.Generator(device="cuda").manual_seed(1)
    v = torch.randn((N, 6), device="cuda", generator=g)
    v[:, 3:] *= 0.5

    def composited(state):
        """per camera: {pixel: tuple(gids composited before T_STOP)} on KNOWN_NONOBJECT pixels"""
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
    from estimator import first_return_operators as fo
    eps = 1e-4
    d12 = torch.zeros((N, 12), device="cuda"); d12[:, :6] = eps * v
    xp, xm = boxplus(scene, d12), boxplus(scene, -d12)
    Ep, _ = evaluate(xp, cams, obs, occ, c)
    Em, _ = evaluate(xm, cams, obs, occ, c)
    cp_, cm_ = composited(xp), composited(xm)
    num = den = 0.0
    e_fd = e_pred = 0.0
    nfroz = nchg = 0
    for k, (cp, cm, cam) in enumerate(zip(Ep.cams, Em.cams, ev.cams)):
        H_, W_ = cam.A.shape
        frozen = torch.zeros(H_ * W_, dtype=torch.bool)
        for p, ids in cp_[k].items():
            same = ids == cm_[k][p]
            frozen[p] = same
            nfroz += int(same); nchg += int(not same)
        frozen = frozen.reshape(H_, W_).cuda()
        fd = (cp.A - cm.A) / (2 * eps)
        jv = no.nr_jvp(cam.args, cam.A, v).double()
        num += float((fd - jv)[frozen].square().sum()); den += float(fd[frozen].square().sum())
        Lp = -torch.log((1 - cp.A).clamp_min(1e-12)); Lm = -torch.log((1 - cm.A).clamp_min(1e-12))
        e_fd += float(((Lp - Lm) / (2 * eps))[frozen].sum())
        e_pred += float((cam.c1.double() * jv)[frozen].sum())
    check("[3] JVP vs FD of A_p on frozen-branch pixels (relative L2 < 1e-2)",
          math.sqrt(num / max(den, 1e-30)) < 1e-2,
          f"{math.sqrt(num / max(den, 1e-30)):.2e} over {nfroz} pixels ({nchg} pixels changed branch, excluded)")
    check("[3] nonlinear energy FD vs (pi/E) JVP on frozen-branch pixels (rel < 1e-2)",
          abs(e_fd - e_pred) / max(abs(e_pred), 1e-12) < 1e-2, f"FD {e_fd:.5f} vs {e_pred:.5f}")
    score = ev.score_pose()
    pred = -float((score * v).sum())
    full_fd = (Ep.energy - Em.energy) / (2 * eps)
    check("[3] score == the summed per-pixel derivative (-<score,v> vs sum (pi/E) Jv, all pixels, rel < 1e-4)",
          abs(pred - sum(float((cm.c1.double() * no.nr_jvp(cm.args, cm.A, v).double()).sum()) for cm in ev.cams))
          / max(abs(pred), 1e-12) < 1e-4, f"-<score,v> {pred:.5f}; whole-image FD incl. branch changes {full_fd:.5f}")
    lhs = rhs = 0.0
    for cam in ev.cams:
        w = torch.randn(cam.A.shape, device="cuda", generator=g) * (cam.status == no.ST_NONOBJECT)
        lhs += float((no.nr_jvp(cam.args, cam.A, v).double() * w).sum())
        rhs += float((no.nr_vjp(cam.args, cam.A, w.float()).double() * v.double()).sum())
    check("[3] adjoint <Jv,w> == <v,J^T w> (rel < 1e-4)", abs(lhs - rhs) / max(abs(lhs), 1e-30) < 1e-4,
          f"{lhs:.6e} vs {rhs:.6e}")
    u = torch.randn((N, 6), device="cuda", generator=g)
    uNv, vNu, vNv = float((u * ev.normal_pose(v)).sum()), float((v * ev.normal_pose(u)).sum()), \
        float((v * ev.normal_pose(v)).sum())
    check("[3] normal operator symmetric and PSD", abs(uNv - vNu) / max(abs(uNv), 1e-30) < 1e-4 and vNv >= 0,
          f"<u,Nv> {uNv:.5e} <v,Nu> {vNu:.5e} <v,Nv> {vNv:.5e}")
    F = ev.fisher_pose()
    act = torch.nonzero(F.diagonal(dim1=1, dim2=2).sum(-1) > 0).reshape(-1)[:4].tolist()
    worst = 0.0
    for i in act:
        for b in range(6):
            e = torch.zeros((N, 6), device="cuda"); e[i, b] = 1.0
            col = ev.normal_pose(e)[i]
            worst = max(worst, float((col - F[i, :, b]).abs().max()) / max(float(F[i].abs().max()), 1e-30))
    check("[3] Fisher blocks == diagonal blocks of the normal operator (rel < 1e-3)",
          len(act) > 0 and worst < 1e-3, f"{len(act)} Gaussians, worst {worst:.2e}")

    print("\n[4] object behind a sharp front occluder: no evidence")
    part = blob(n=20, seed=5, centre=(0.0, 0.0, 0.0), spread=0.2)
    sep_A = []
    for sep in (0.30, 0.10, 0.06, 0.04, 0.02, 0.0):
        lp = build_linearization(part, cams)
        obs_o = []
        for k, vv in enumerate(lp.views):
            M = torch.zeros_like(vv.alpha, dtype=torch.bool)               # fully occluded: mask empty
            d = torch.where(vv.alpha > 1e-7, torch.full_like(vv.depth, float(vv.depth[vv.alpha > 0.5].min()) - sep),
                            torch.full_like(vv.depth, 4.0))
            obs_o.append(prepared(torch.full((3,) + vv.alpha.shape, 0.2, device="cuda"), d, d > 0, M, f"c{k}"))
        e_o, _ = evaluate(part, cams, obs_o, occ, c)
        occl = [cam.A[(cam.status == no.ST_NONOBJECT) & (lp.views[k].alpha > 0.5)] for k, cam in enumerate(e_o.cams)]
        sep_A.append((sep, float(torch.cat(occl).mean()), e_o))
    far = sep_A[0][2]
    sc_far = far.score_pose()
    behind = [far.cams[k].A[(far.cams[k].status == no.ST_NONOBJECT) & (build_linearization(part, cams).views[k].alpha > 0.5)]
              for k in range(3)]
    check("[4] occluder 0.30 m in front (15 sigma): A_p < 1e-12 on every occluded ray",
          float(torch.cat(behind).max()) < 1e-12, f"max A {float(torch.cat(behind).max()):.1e}")
    check("[4] ... and the score of the fully covered part is ~0",
          float(sc_far.abs().max()) < 1e-9, f"max |score| {float(sc_far.abs().max()):.2e} (A_max {float(far.cams[0].A.max()):.1e})")
    mono = all(sep_A[i][1] <= sep_A[i + 1][1] + 1e-12 for i in range(len(sep_A) - 1))
    check("[4] mean occluded-ray A_p grows continuously as the separation shrinks",
          mono, ", ".join(f"{s:.2f}m:{a_:.3g}" for s, a_, _ in sep_A))
    legacy = [t for t in __import__("estimator.masked_observation", fromlist=["x"]).phase1_camera_terms(
        build_linearization(part, cams, background=torch.full((3,), 0.5, device="cuda")).views, obs_o, c, model())]
    leg_rhs = build_linearization(part, cams, background=torch.full((3,), 0.5, device="cuda")).apply_HT_alpha(
        [t.kappa * t.r_a for t in legacy])
    check("[4] (contrast) the legacy alpha term pushes the same hidden part", float(leg_rhs.abs().max()) > 1.0,
          f"legacy max |H_A^T kappa r_A| {float(leg_rhs.abs().max()):.3g}")

    print("\n[5] a Gaussian moved in front of measured non-object depth raises the NLL")
    one = blob(n=1, seed=9, centre=(0.0, 0.0, 0.0), spread=0.0)
    base = scene_free_space(obj, cams)
    e_in, _ = evaluate(cat(obj, dataclasses.replace(one, mu=one.mu + torch.tensor([0.0, 0.0, 0.0], device="cuda"))),
                       cams, base, occ, c)
    e_out, _ = evaluate(cat(obj, dataclasses.replace(one, mu=one.mu + torch.tensor([0.42, 0.0, 0.0], device="cuda"))),
                        cams, base, occ, c)
    check("[5] inside the silhouette -> outside onto free-space rays: E increases",
          e_out.energy > e_in.energy + 0.05, f"{e_in.energy:.4f} -> {e_out.energy:.4f}")

    print("\n[6] invalid depth on M = 0: no term, no full-ray fallback")
    lin6 = build_linearization(scene, cams, background=torch.full((3,), 0.5, device="cuda"))
    inval = [prepared(o.rgb_foreground_premult, torch.where(o.object_coverage > 0.5, o.depth_observed,
                                                            torch.zeros_like(o.depth_observed)),
                      o.object_coverage > 0.5, o.object_coverage > 0.5, o.camera) for o in obs]
    e6, _ = evaluate(scene, cams, inval, occ, c)
    check("[6] every M = 0 pixel UNKNOWN, energy and score exactly 0",
          all(x["KNOWN_NONOBJECT_DEPTH"] == 0 for x in e6.status_counts()) and e6.energy == 0.0
          and float(e6.score_pose().abs().max()) == 0.0, str(e6.status_counts()[0]))
    pos = positive_camera_terms(lin6.views, obs, c, model())
    offm = [(o.object_coverage <= 0.5) for o in obs]
    check("[6] RGB precision and kappa exactly 0 off M = 1",
          all(float(t.p_rgb[:, m].abs().max()) == 0.0 and float(t.kappa[m].abs().max()) == 0.0
              for t, m in zip(pos, offm)))
    check("[6] status map: OBJECT_SUPPORTED exactly on M = 1",
          all(bool(((ray_status(o) == no.ST_SUPPORTED) == (o.object_coverage > 0.5)).all()) for o in obs))

    print("\n[7] colour independence of the negative-ray evidence")
    energies = {}
    for name, col in (("black", (0.0, 0.0, 0.0)), ("bright", (0.95, 0.95, 0.95)), ("grey-bg", (0.5, 0.5, 0.5))):
        s7 = cat(obj, blob(n=6, seed=3, centre=(0.33, 0.05, 0.0), spread=0.08, colour=col))
        energies[name] = evaluate(s7, cams, obs, occ, c)[0].energy
    check("[7] identical negative energy for black / bright / background-grey Gaussians",
          len({round(x, 9) for x in energies.values()}) == 1, str({k: round(x, 6) for k, x in energies.items()}))
    lowop = evaluate(cat(obj, blob(n=6, seed=3, centre=(0.33, 0.05, 0.0), spread=0.08, opacity=0.1)),
                     cams, obs, occ, c)[0].energy
    check("[7] lower opacity -> lower negative energy", lowop < energies["black"],
          f"opacity 0.85: {energies['black']:.4f}, 0.1: {lowop:.4f}")

    print("\n[8] 3 cameras: occluded in camera 0, visible in 1 and 2")
    l8 = build_linearization(obj, cams)
    obs8 = []
    for k, vv in enumerate(l8.views):
        M = vv.alpha > 0.5
        d = torch.where(M, vv.depth, torch.full_like(vv.depth, 4.0))
        if k == 0:
            d = torch.where(M, vv.depth - 0.3, d)
            M = torch.zeros_like(M)
        obs8.append(prepared(vv.rgb + (1 - vv.alpha)[None] * 0.3, d, d > 0, M, f"c{k}"))
    lin8 = build_linearization(obj, cams, background=torch.full((3,), 0.5, device="cuda"))
    from estimator.sparse_iekf import measurement_operators
    terms8 = positive_camera_terms(lin8.views, obs8, c, model())
    _, _, fisher8 = measurement_operators(lin8, [t.as_prec4() for t in terms8], [(t.kappa, t.r_a) for t in terms8])
    F8 = fisher8()
    tr8 = F8.diagonal(dim1=1, dim2=2).sum(-1)
    only0 = cat(obj)                         # same Gaussians; camera 0 alone has no positive pixels
    seen = int((tr8 > 0).sum())
    check("[8] the camera that sees them still informs most Gaussians", seen >= int(0.8 * obj.mu.shape[0]),
          f"{seen}/{obj.mu.shape[0]} with nonzero Fisher")
    check("[8] camera 0 carries no positive terms (its rays are occluded)",
          float(terms8[0].p_rgb.abs().max()) == 0.0 and float(terms8[0].kappa.abs().max()) == 0.0)
    del only0

    print("\n[9] deterministic reductions are bitwise repeatable")
    cam = ev.cams[0]
    w = cam.c1
    r1 = no.nr_vjp(cam.args, cam.A, w, True); r2 = no.nr_vjp(cam.args, cam.A, w, True)
    f1 = no.nr_fisher(cam.args, cam.A, cam.c2, True); f2 = no.nr_fisher(cam.args, cam.A, cam.c2, True)
    rf = no.nr_vjp(cam.args, cam.A, w, False)
    check("[9] deterministic VJP and Fisher bitwise equal across calls", torch.equal(r1, r2) and torch.equal(f1, f2))
    check("[9] deterministic VJP == float-atomic VJP within 1e-5 relative",
          float((r1 - rf).abs().max()) <= 1e-5 * max(float(rf.abs().max()), 1e-30),
          f"{float((r1 - rf).abs().max()):.2e}")

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED:", *FAILURES, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
