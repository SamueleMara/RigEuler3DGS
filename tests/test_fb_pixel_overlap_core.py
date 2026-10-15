#!/usr/bin/env python
"""Ground-truth tests of the F/B overlap diagnostic on small synthetic scenes,
through the REAL CUDA renderer and the diagnostic contributor kernel.

CODEX_PHASE1_FORWARD_BACKWARD_PIXEL_OVERLAP_COVERAGE_DIAGNOSTICS.md section 6:
T1 contribution conservation, T3 identical Gaussian in both models, T4 same
pixel / different depth, T5 partial occlusion, T6 wrong geometry despite
coverage, T7 unobservable ambiguity, T8 calibration and depth checks fail
closed, T9 sparse pairing equals CPU exhaustive pairing, T11 sensitivity
sweeps are monotone, T14 GT-invalid is never missing geometry.

    python tests/test_fb_pixel_overlap_core.py
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# scene helpers
# ---------------------------------------------------------------------------

def state(mu, colors, opacity, scales, R=None):
    from estimator.state import GaussianMotionState
    n = mu.shape[0]
    return GaussianMotionState(
        mu=mu.float().cuda().contiguous(),
        R=(torch.eye(3).expand(n, 3, 3) if R is None else R).float().cuda().contiguous(),
        v=torch.zeros(n, 3).cuda(), omega=torch.zeros(n, 3).cuda(),
        colors=colors.float().cuda().contiguous(),
        opacity=opacity.reshape(n, 1).float().cuda().contiguous(),
        scales=scales.float().cuda().contiguous())


def plate(cx, cy, z, w, h, n, colour, opacity=0.95):
    """A flat grid of splats facing +z (camera 0 looks down -z from z = 2)."""
    xs = torch.linspace(cx - w / 2, cx + w / 2, n)
    ys = torch.linspace(cy - h / 2, cy + h / 2, n)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    mu = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((n * n,), z)], 1)
    sp = max(w, h) / (n - 1)
    sc = torch.tensor([[0.5 * sp, 0.5 * sp, 0.002]]).expand(n * n, 3)
    col = torch.tensor(colour).expand(n * n, 3)
    return mu, col, torch.full((n * n,), opacity), sc


def cat(*parts):
    return tuple(torch.cat([p[k] for p in parts]) for k in range(4))


def make_state(*parts):
    mu, col, op, sc = cat(*parts)
    return state(mu, col, op, sc)


def render(st, cam):
    from estimator.renderer_bridge import build_linearization
    from observation_diagnostics.renderer_access import pixel_contributions
    v = build_linearization(st, [cam]).views[0]
    return v, pixel_contributions(v.cache)


def gt_of(st, cam, alpha_min=0.5):
    v, _ = render(st, cam)
    valid = v.alpha >= alpha_min
    return {"rgb": v.rgb.clone(), "depth": torch.where(valid, v.depth, torch.zeros_like(v.depth)),
            "valid": valid}


def run_scene(F_states, B_states, GT, cams, prune_tau=0.0):
    """The pipeline's measurement and labelling steps, through its functions."""
    from observation_diagnostics import aggregate as ag
    from observation_diagnostics import gaussian_evidence as ge
    from observation_diagnostics import pair_evidence as pv
    from observation_diagnostics import pixel_evidence as pe
    th = pe.Thresholds()
    NF, NB = F_states[0].mu.shape[0], B_states[0].mu.shape[0]
    gp = pv.GlobalPairs(NF, NB, shards=3)
    tot = {"F": None, "B": None}
    per_frame = {"F": [], "B": []}
    counts = []
    om = {"omitted_O_surf_F": np.zeros(NF), "omitted_O_surf_B": np.zeros(NB)}
    for t, (sF, sB) in enumerate(zip(F_states, B_states)):
        accF, accB = ge.new_frame_accumulator(NF, "cuda"), ge.new_frame_accumulator(NB, "cuda")
        views = []
        for ci, cam in enumerate(cams):
            vF, cF = render(sF, cam)
            vB, cB = render(sB, cam)
            gt = GT[t][ci]
            Fd = {"rgb": vF.rgb, "depth": vF.depth, "alpha": vF.alpha}
            Bd = {"rgb": vB.rgb, "depth": vB.depth, "alpha": vB.alpha}
            f = pe.classify(Fd, Bd, gt, th)
            counts.append(pe.view_metrics(f, Fd, Bd, th) | {"fields": f})
            ge.accumulate(accF, cF, vF.cache.depth, f, gt["depth"], "F", ci, th.contrib_depth_tol_m)
            ge.accumulate(accB, cB, vB.cache.depth, f, gt["depth"], "B", ci, th.contrib_depth_tol_m)
            vp = pv.view_pairs(cF, cB, vF.cache.depth, vB.cache.depth, sF.colors, sB.colors,
                               sF.mu, sB.mu, gt["depth"], gt["valid"], NF, NB, th,
                               prune_tau=prune_tau)
            for s in ("F", "B"):
                om[f"omitted_O_surf_{s}"] += vp[f"omitted_O_surf_{s}"]
            views.append((ci, vp))
        gp.add(t, pv.frame_table(views))
        for s, acc in (("F", accF), ("B", accB)):
            a = ge.to_numpy(acc)
            per_frame[s].append(a)
            tot[s] = ({k: v.astype(np.float64) for k, v in a.items()} if tot[s] is None else
                      {k: tot[s][k] + v for k, v in a.items()})
    tab = gp.table()
    res = {"tot": tot, "per_frame": per_frame, "counts": counts, "table": tab, "th": th}
    if not tab:
        return res
    k = tab["keys"]
    iF, jB = k // NB, k % NB
    rho_surf = tab["O_surf_1"] / np.sqrt(np.maximum(tot["F"]["q2_kappa_1"][iF] * tot["B"]["q2_kappa_1"][jB], 1e-30))
    rho_pix = tab["O"] / np.sqrt(np.maximum(tot["F"]["q2_all"][iF] * tot["B"]["q2_all"][jB], 1e-30))
    inc = np.divide(tab["O_incompat"], tab["O"], out=np.zeros(k.size), where=tab["O"] > 0)
    bpF = ag.best_partners(iF, jB, rho_surf, rho_pix, inc, tab["frames_surf"], NF)
    bpB = ag.best_partners(jB, iF, rho_surf, rho_pix, inc, tab["frames_surf"], NB)
    ag.certify(bpF, iF, jB, tab["O_surf_1"], om["omitted_O_surf_F"], NF)
    ag.certify(bpB, jB, iF, tab["O_surf_1"], om["omitted_O_surf_B"], NB)
    LT = ag.LABEL_THRESHOLDS
    res["labF"], _ = ag.label(tot["F"]["mass_total"], tot["F"]["mass_supported"], bpF, LT)
    res["labB"], _ = ag.label(tot["B"]["mass_total"], tot["B"]["mass_supported"], bpB, LT)
    res |= {"bpF": bpF, "bpB": bpB, "rho_surf": rho_surf, "rho_pix": rho_pix, "iF": iF, "jB": jB}
    return res


def cams_pair():
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    ring = make_cameras(n_cam=16, H=48, W=64, radius=2.0)
    return [ring[0], ring[1]]          # 0 and 22.5 degrees


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def t1_conservation():
    from observation_diagnostics.renderer_access import reference_composite
    print("\n[T1] contribution conservation and an independent compositor")
    g = torch.Generator().manual_seed(1)
    n = 160
    mu = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([0.9, 0.7, 0.6])
    from estimator.manifold import so3_exp
    R = so3_exp(torch.randn(n, 3, generator=g).double()).float()          # anisotropic, rotated
    sc = 0.01 + 0.08 * torch.rand(n, 3, generator=g)
    op = 0.3 + 0.69 * torch.rand(n, generator=g)
    # an opaque stack at the centre forces early termination
    stack = plate(0.0, 0.0, 0.35, 0.2, 0.2, 6, (0.9, 0.9, 0.9), opacity=0.99)
    s2 = plate(0.0, 0.0, 0.30, 0.2, 0.2, 6, (0.2, 0.9, 0.9), opacity=0.99)
    s3 = plate(0.0, 0.0, 0.25, 0.2, 0.2, 6, (0.9, 0.2, 0.9), opacity=0.99)
    allmu, allc, allo, alls = cat((mu, torch.rand(n, 3, generator=g), op, sc), stack, s2, s3)
    Rall = torch.cat([R, torch.eye(3).expand(3 * 36, 3, 3)])
    st = state(allmu, allc, allo, alls, R=Rall)
    cam = cams_pair()[0]
    v, c = render(st, cam)
    HW = c.H * c.W
    pix = c.pixel
    A = torch.zeros(HW, dtype=torch.float64, device="cuda").index_add_(0, pix, c.q.double())
    Z = torch.zeros_like(A).index_add_(0, pix, c.q.double() * v.cache.depth[c.gid.long()].double())
    C = torch.zeros(3, HW, dtype=torch.float64, device="cuda")
    for ch in range(3):
        C[ch].index_add_(0, pix, c.q.double() * st.colors[c.gid.long(), ch].double())
    dA = float((A - v.alpha.reshape(-1).double()).abs().max())
    check("sum_i q = A at every pixel", dA < 2e-6, f"max {dA:.1e}")
    cov = v.alpha.reshape(-1) > 0.05
    dD = float((Z / (A + v.cache.eps_depth) - v.depth.reshape(-1).double())[cov].abs().max())
    check("sum_i q z_i / (A + eps) = rendered depth", dD < 1e-5, f"max {dD:.1e} m")
    dC = float((C - v.rgb.reshape(3, -1).double()).abs().max())
    check("sum_i q c_i + T bg = rendered RGB (bg = 0)", dC < 2e-6, f"max {dC:.1e}")
    check("final T bitwise equal to the renderer's", torch.equal(c.final_T, v.transmittance.reshape(-1)))
    check("early termination exercised", int(c.terminated.sum()) > 0, f"{int(c.terminated.sum())} pixels")
    check("background pixels have no contributors and T = 1",
          bool((c.final_T[c.counts == 0] == 1).all()) and int((c.counts == 0).sum()) > 0)
    ref = reference_composite(v.cache, c.W, c.H)
    same_sets, worst = True, 0.0
    off = c.offsets.cpu().numpy()
    gid, al, T = c.gid.cpu().numpy(), c.alpha.cpu().numpy(), c.T.cpu().numpy()
    near = 0
    for p, recs in ref.items():
        a, b = off[p], off[p + 1]
        mine = list(zip(gid[a:b].tolist(), al[a:b].tolist(), T[a:b].tolist()))
        if [r[0] for r in recs] != [m[0] for m in mine]:
            same_sets = False
            near += 1
            continue
        for (g1, a1, t1), (g2, a2, t2) in zip(recs, mine):
            worst = max(worst, abs(a1 - a2), abs(t1 - t2))
    # float64 exp vs __expf can flip a contributor sitting exactly at 1/255 or T_STOP
    check("independent float64 compositor: same contributor lists",
          near <= max(2, len(ref) // 500), f"{near} of {len(ref)} pixels differ (boundary flips allowed: "
          f"{max(2, len(ref) // 500)})")
    check("independent compositor: alpha and T agree", worst < 1e-5, f"max {worst:.1e}")


def t3_identical():
    print("\n[T3] the identical physical Gaussian in both models")
    cams = cams_pair()
    P = plate(0.0, 0.0, 0.0, 0.6, 0.45, 9, (0.8, 0.5, 0.2))
    st = make_state(P)
    T = 3
    GT = [[gt_of(st, c) for c in cams] for _ in range(T)]
    r = run_scene([st] * T, [st] * T, GT, cams)
    lab, bp = r["labF"], r["bpF"]
    ev = lab != "UNRESOLVED_LOW_EVIDENCE"
    ev &= lab != "NEVER_VISIBLE"
    self_best = bp["best"][ev] == np.flatnonzero(ev)
    check("every evidenced F Gaussian's best partner is its own copy",
          bool(self_best.all()), f"{int(self_best.sum())}/{int(ev.sum())}")
    check("their rho_surf is close to 1", float(bp["rho_surf"][ev].min()) > 0.9,
          f"min {bp['rho_surf'][ev].min():.3f}")
    frac = float((lab[ev] == "LIKELY_COMMON_SURFACE").mean())
    check("labelled LIKELY_COMMON_SURFACE", frac >= 0.9, f"{100 * frac:.1f}%")
    U = r["tot"]["B"]["witness"].sum() + r["tot"]["F"]["witness"].sum()
    check("no exclusive witness: nothing argues for keeping two copies", U == 0.0, f"U {U}")
    c0 = r["counts"][0]
    check("pixels: same-surface candidate dominates both-cover",
          c0["BOTH_SAME_OBSERVED_SURFACE_CANDIDATE"] >= 0.95 * c0["BOTH_COVER__on_gt_valid"])


def t4_layers():
    print("\n[T4] same pixel, distinct depth surfaces")
    cams = cams_pair()
    front = make_state(plate(0.0, 0.0, 0.0, 0.6, 0.45, 9, (0.8, 0.5, 0.2)))
    back = make_state(plate(0.0, 0.0, -0.3, 0.6, 0.45, 9, (0.8, 0.5, 0.2)))
    T = 3
    GT = [[gt_of(front, c) for c in cams] for _ in range(T)]
    r = run_scene([front] * T, [back] * T, GT, cams)
    tab = r["table"]
    check("high same-pixel overlap", float(tab["O"].sum()) > 10.0, f"O {tab['O'].sum():.1f}")
    check("zero same-surface overlap", float(tab["O_surf_1"].sum()) == 0.0)
    check("the overlap is attributed to different depth", float(tab["O_incompat"].sum()) >= 0.99 * float(tab["O"].sum()))
    bad = {"LIKELY_COMMON_SURFACE", "WEAK_COMMON_SURFACE", "AMBIGUOUS"}
    check("no Gaussian of either source claims a common surface",
          not (set(r["labF"]) & bad) and not (set(r["labB"]) & bad))
    ev = r["tot"]["F"]["mass_supported"] >= 1.0
    frac = float((r["labF"][ev] == "OVERLAPPING_DIFFERENT_DEPTH").mean())
    check("F Gaussians (which explain GT) are OVERLAPPING_DIFFERENT_DEPTH", frac >= 0.9,
          f"{100 * frac:.1f}% of {int(ev.sum())}")
    c0 = r["counts"][0]
    check("pixel map: BOTH_COVER_DIFFERENT_DEPTH, never same-surface",
          c0["BOTH_SAME_OBSERVED_SURFACE_CANDIDATE"] == 0 and c0["BOTH_COVER_DIFFERENT_DEPTH"] > 0)


def t5_occlusion(tmp: Path):
    print("\n[T5] an initially occluded surface revealed later")
    from observation_diagnostics import aggregate as ag
    from PIL import Image
    cams = cams_pair()
    H = plate(0.05, 0.0, 0.0, 0.3, 0.3, 7, (0.1, 0.8, 0.2))
    nH = H[0].shape[0]
    T = 6
    F_states, B_states, true_states = [], [], []
    for t in range(T):
        ox = 0.05 if t < 3 else 1.6                    # the occluder leaves at frame 3
        O = plate(ox, 0.0, 0.3, 0.45, 0.45, 9, (0.7, 0.2, 0.2), opacity=0.99)
        F_states.append(make_state(O))                 # F never modelled H
        B_states.append(make_state(H, O))              # B has H (rows 0..nH-1) and O
        true_states.append(make_state(H, O))
    GT = [[gt_of(s, c) for c in cams] for s in true_states]
    r = run_scene(F_states, B_states, GT, cams)
    wit = np.stack([pf["witness"] for pf in r["per_frame"]["B"]])[:, :nH].sum(1)
    check("no witness while occluded (frames 0..2)", float(wit[:3].sum()) == 0.0, f"{wit[:3]}")
    check("witnessed after the reveal (frames 3..5)", bool((wit[3:] > 1.0).all()), f"{np.round(wit[3:], 1)}")
    Ufull = r["tot"]["B"]["witness"][:nH]
    lab = r["labB"][:nH]
    ok = (Ufull >= 1.0)
    check("most of H passes the unique-candidate gate U >= 1", ok.mean() >= 0.6,
          f"{int(ok.sum())}/{nH}")
    check("H is not discarded for being absent at frame 0: no common-surface label",
          not set(lab[ok]) & {"LIKELY_COMMON_SURFACE", "AMBIGUOUS"}, f"{sorted(set(lab[ok]))}")
    # occlusion test from GT files laid out like the dataset
    names = ["cam_000", "cam_001"]
    for t in range(T):
        for ci, nm in enumerate(names):
            d = tmp / "rgb" / nm
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray((GT[t][ci]["rgb"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
                            .astype(np.uint8)).save(d / f"frame_{t:06d}.png")
            dd = tmp / "depth_npy" / nm
            dd.mkdir(parents=True, exist_ok=True)
            np.save(dd / f"frame_{t:06d}.npy", GT[t][ci]["depth"].cpu().numpy().astype(np.float32))

    class Src:
        frames = list(range(T))
        mu = np.stack([s.mu.cpu().numpy() for s in B_states])

    ctx = {"frames": list(range(T)), "cams": names, "all_cams": names, "specs": cams, "data": tmp}
    occ = ag.occlusion_status(ctx, np.arange(nH), Src, 3, 0.04)
    check("occlusion test: OCCLUSION_HYPOTHESIS from the GT depth in front",
          occ["status"] == "OCCLUSION_HYPOTHESIS", f"{occ['status']} {occ.get('votes_over_early_frame_cameras')}")


def t6_wrong_geometry():
    print("\n[T6] coverage with the wrong depth does not explain")
    cams = cams_pair()
    good = make_state(plate(0.0, 0.0, 0.0, 0.6, 0.45, 9, (0.8, 0.5, 0.2)))
    wrong = make_state(plate(0.0, 0.0, 0.2, 0.6, 0.45, 9, (0.8, 0.5, 0.2)))
    T = 3
    GT = [[gt_of(good, c) for c in cams] for _ in range(T)]
    r = run_scene([wrong] * T, [good] * T, GT, cams)
    c0 = r["counts"][0]
    f = c0["fields"]
    covF_valid = int((f["covF"] & f["valid"]).sum())
    check("F covers the observed pixels", covF_valid > 0.5 * int(f["valid"].sum()))
    check("but explains none of them", int(f["SdF"].sum()) == 0)
    check("those pixels are B_SUPPORTED_F_UNEXPLAINED", c0["B_SUPPORTED_F_UNEXPLAINED"] > 0.8 * covF_valid)
    check("and recorded as F in front of the observation where B is right",
          c0["F_IN_FRONT_OF_OBS_B_CORRECT"] > 0.8 * covF_valid)
    check("pixel overlap without same-surface support yields no match",
          float(r["table"]["O_surf_1"].sum()) == 0.0
          and not set(r["labB"]) & {"LIKELY_COMMON_SURFACE", "WEAK_COMMON_SURFACE", "AMBIGUOUS"})
    check("F's mass in front of the observation is counted", r["tot"]["F"]["mass_front"].sum() > 1.0)


def t7_unobservable():
    print("\n[T7] candidates only behind an opaque foreground")
    cams = cams_pair()
    front = plate(0.0, 0.0, 0.2, 0.9, 0.7, 13, (0.9, 0.9, 0.9), opacity=0.995)
    hid = plate(0.0, 0.0, -0.1, 0.3, 0.3, 5, (0.1, 0.1, 0.9))
    nf = front[0].shape[0]
    st = make_state(front, hid)
    T = 3
    GT = [[gt_of(st, c) for c in cams] for _ in range(T)]
    r = run_scene([st] * T, [st] * T, GT, cams)
    lf, lb = r["labF"][nf:], r["labB"][nf:]
    allowed = {"NEVER_VISIBLE", "UNRESOLVED_LOW_EVIDENCE"}
    check("hidden Gaussians are unresolved, not absent or bad",
          set(lf) <= allowed and set(lb) <= allowed, f"F {sorted(set(lf))} B {sorted(set(lb))}")
    check("and carry no witness", float(r["tot"]["F"]["witness"][nf:].sum() +
                                        r["tot"]["B"]["witness"][nf:].sum()) == 0.0)


def t9_pairs_exhaustive():
    print("\n[T9] sparse streaming pairs equal CPU exhaustive pairs")
    from observation_diagnostics import gaussian_evidence as ge
    from observation_diagnostics import pair_evidence as pv
    from observation_diagnostics import pixel_evidence as pe
    from observation_diagnostics.renderer_access import reference_composite
    g = torch.Generator().manual_seed(7)
    cam = cams_pair()[0]

    def rnd(n, z0):
        mu = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([0.6, 0.45, 0.1]) + torch.tensor([0, 0, z0])
        return state(mu, torch.rand(n, 3, generator=g), 0.4 + 0.5 * torch.rand(n, generator=g),
                     0.02 + 0.05 * torch.rand(n, 3, generator=g))
    sF, sB = rnd(40, 0.0), rnd(45, 0.01)
    vF, cF = render(sF, cam)
    vB, cB = render(sB, cam)
    gt = gt_of(sF, cam, 0.3)
    th = pe.Thresholds()
    NF, NB = 40, 45
    full = pv.view_pairs(cF, cB, vF.cache.depth, vB.cache.depth, sF.colors, sB.colors, sF.mu, sB.mu,
                         gt["depth"], gt["valid"], NF, NB, th, prune_tau=0.0)
    tiny = pv.view_pairs(cF, cB, vF.cache.depth, vB.cache.depth, sF.colors, sB.colors, sF.mu, sB.mu,
                         gt["depth"], gt["valid"], NF, NB, th, prune_tau=0.0, block_budget=50)
    rF = reference_composite(vF.cache, cF.W, cF.H)
    rB = reference_composite(vB.cache, cB.W, cB.H)
    zF, zB = vF.cache.depth.cpu().numpy().astype(np.float64), vB.cache.depth.cpu().numpy().astype(np.float64)
    D = gt["depth"].reshape(-1).cpu().numpy().astype(np.float64)
    V = gt["valid"].reshape(-1).cpu().numpy()
    ref = {}
    tols = ge.PAIR_TOLS_M
    for p in rF:
        for (i, a1, t1) in rF[p]:
            for (j, a2, t2) in rB[p]:
                w = a1 * t1 * a2 * t2
                d = ref.setdefault(i * NB + j, np.zeros(6))
                d[0] += w
                d[5] += 1
                for k, t in enumerate(tols):
                    if V[p] and abs(zF[i] - zB[j]) <= t and abs(zF[i] - D[p]) <= t and abs(zB[j] - D[p]) <= t:
                        d[1 + k] += w
                if abs(zF[i] - zB[j]) > tols[1]:
                    d[4] += w
    keys = np.array(sorted(ref))
    exp = np.stack([ref[k] for k in keys])
    check("same set of (i, j) pairs as the exhaustive CPU enumeration",
          np.array_equal(keys, full["keys"]), f"{keys.size} vs {full['keys'].size}")
    if np.array_equal(keys, full["keys"]):
        got = np.stack([full["cols"][c] for c in ("O", "O_surf_0", "O_surf_1", "O_surf_2", "O_incompat")], 1)
        err = np.abs(got - exp[:, :5]).max() / max(np.abs(exp[:, :5]).max(), 1e-12)
        check("O, O_surf (3 tolerances), O_incompat equal the CPU sums", err < 5e-5, f"rel {err:.1e}")
        check("shared-pixel counts equal the CPU count", np.array_equal(full["cols"]["npix"], exp[:, 5]))
        ii, jj = full["keys"] // NB, full["keys"] % NB
        d3 = np.linalg.norm(sF.mu.cpu().numpy()[ii] - sB.mu.cpu().numpy()[jj], axis=1)
        rg = np.sqrt(((sF.colors.cpu().numpy()[ii] - sB.colors.cpu().numpy()[jj]) ** 2).mean(1))
        check("centre-distance and colour columns are O times the pair's value",
              np.allclose(full["cols"]["dist3d"], exp[:, 0] * d3, rtol=1e-4, atol=1e-9)
              and np.allclose(full["cols"]["rgb_dist"], exp[:, 0] * rg, rtol=1e-4, atol=1e-9))
    check("blocking the sparse products (tiny blocks) does not change a single sum",
          np.array_equal(full["keys"], tiny["keys"]) and
          all(np.allclose(full["cols"][c], tiny["cols"][c], rtol=1e-6, atol=1e-9) for c in pv.SUM_COLS))
    tau = float(np.median(full["cols"]["O"]))
    pr = pv.view_pairs(cF, cB, vF.cache.depth, vB.cache.depth, sF.colors, sB.colors, sF.mu, sB.mu,
                       gt["depth"], gt["valid"], NF, NB, th, prune_tau=tau)
    kept = np.zeros(NF)
    np.add.at(kept, pr["keys"] // NB, pr["cols"]["O_surf_1"])
    allm = np.zeros(NF)
    np.add.at(allm, full["keys"] // NB, full["cols"]["O_surf_1"])
    # stored columns are float32 (sums and the pruning decision are float64)
    check("pruning: stored + omitted = exact, per F Gaussian",
          np.allclose(kept + pr["omitted_O_surf_F"], allm, rtol=1e-6, atol=1e-9),
          f"max rel {np.max(np.abs(kept + pr['omitted_O_surf_F'] - allm) / np.maximum(allm, 1e-12)):.1e}")
    check("pruning: stored pairs keep their exact per-view sums",
          all(np.allclose(pr["cols"][c], full["cols"][c][np.isin(full["keys"], pr["keys"])]) for c in pv.SUM_COLS))
    # global merge: three frames of the same view = three times the sums, frames = 3
    gp = pv.GlobalPairs(NF, NB, shards=4)
    ft = pv.frame_table([(0, full)])
    for t in range(3):
        gp.add(t, ft)
    tab = gp.table()
    check("sequence merge: sums add, frames count, camera bits OR",
          np.allclose(tab["O"], 3 * ft["O"], rtol=1e-6) and bool((tab["frames"] == 3).all())
          and bool((tab["cams"] == 1).all()))


def t8_fail_closed(tmp: Path):
    print("\n[T8] calibration and depth problems fail closed")
    from types import SimpleNamespace as NS
    from observation_diagnostics.sources import check_gauge, load_gt
    eye = torch.eye(4)

    def cam(name, w=64, h=48, fx=0.9, shift=0.0):
        v = eye.clone()
        v[3, 0] = shift
        return NS(image_name=name, image_width=w, image_height=h, FoVx=fx, FoVy=0.7,
                  world_view_transform=v, full_proj_transform=v)

    def raises(fn):
        try:
            fn()
            return False
        except SystemExit:
            return True
    A = NS(cams_all=[cam("cam_000"), cam("cam_001")])
    check("identical calibration accepted", not raises(lambda: check_gauge(A, A)))
    check("different extrinsics refused",
          raises(lambda: check_gauge(A, NS(cams_all=[cam("cam_000"), cam("cam_001", shift=0.01)]))))
    check("different intrinsics refused",
          raises(lambda: check_gauge(A, NS(cams_all=[cam("cam_000"), cam("cam_001", fx=0.8)]))))
    check("different resolution refused",
          raises(lambda: check_gauge(A, NS(cams_all=[cam("cam_000"), cam("cam_001", w=32)]))))
    check("different camera names refused",
          raises(lambda: check_gauge(A, NS(cams_all=[cam("cam_000"), cam("cam_002")]))))
    from PIL import Image
    (tmp / "rgb" / "cam_000").mkdir(parents=True, exist_ok=True)
    (tmp / "depth_npy" / "cam_000").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(tmp / "rgb" / "cam_000" / "frame_000000.png")
    np.save(tmp / "depth_npy" / "cam_000" / "frame_000000.npy", np.ones((48, 64), np.float32))
    check("valid GT loads", not raises(lambda: load_gt(tmp, "cam_000", 0, 48, 64, device="cpu")))
    check("resolution mismatch refused", raises(lambda: load_gt(tmp, "cam_000", 0, 40, 64, device="cpu")))
    check("missing frame refused", raises(lambda: load_gt(tmp, "cam_000", 1, 48, 64, device="cpu")))
    np.save(tmp / "depth_npy" / "cam_000" / "frame_000000.npy", -np.ones((48, 64), np.float32))
    check("negative depth refused", raises(lambda: load_gt(tmp, "cam_000", 0, 48, 64, device="cpu")))
    _r, d, valid, _ = (None, None, None, None)
    np.save(tmp / "depth_npy" / "cam_000" / "frame_000000.npy",
            np.where(np.arange(64)[None, :] < 32, 1.5, 0.0).astype(np.float32) * np.ones((48, 1), np.float32))
    _r, d, valid, _ = load_gt(tmp, "cam_000", 0, 48, 64, device="cpu")
    check("metric depth read verbatim, 0 = no return = invalid",
          float(d[0, 0]) == 1.5 and not bool(valid[0, 40]) and bool(valid[0, 0]))


def t11_sweeps():
    print("\n[T11] sensitivity sweeps behave monotonically")
    from observation_diagnostics import pixel_evidence as pe
    cams = cams_pair()
    good = make_state(plate(0.0, 0.0, 0.0, 0.6, 0.45, 9, (0.8, 0.5, 0.2)))
    off = make_state(plate(0.0, 0.0, 0.03, 0.6, 0.45, 9, (0.7, 0.5, 0.3)))
    gt = gt_of(good, cams[0])
    vF, _ = render(off, cams[0])
    vB, _ = render(good, cams[0])
    rows = pe.sweep_rows({"rgb": vF.rgb, "depth": vF.depth, "alpha": vF.alpha},
                         {"rgb": vB.rgb, "depth": vB.depth, "alpha": vB.alpha}, gt)
    mono_d, mono_a = True, True
    for a in pe.SWEEP_ALPHA:
        r = [x for x in rows if x["alpha_cover"] == a and "BOTH_VALID_SUPPORT" in x]
        r.sort(key=lambda x: x["depth_tol_m"])
        sup = [x["BOTH_VALID_SUPPORT"] + x["F_SUPPORTED_B_UNEXPLAINED"] for x in r]
        mono_d &= all(b >= a_ for a_, b in zip(sup, sup[1:]))
    for d in pe.SWEEP_DEPTH_M:
        r = [x for x in rows if x["depth_tol_m"] == d and "BOTH_COVER" in x]
        r.sort(key=lambda x: x["alpha_cover"])
        cov = [x["BOTH_COVER"] + x["F_ONLY_COVER"] for x in r]
        mono_a &= all(b <= a_ for a_, b in zip(cov, cov[1:]))
    check("F's explained pixels never shrink as the depth tolerance grows", mono_d)
    check("F's covered pixels never grow as the alpha threshold rises", mono_a)
    d1 = [x for x in rows if x.get("alpha_cover") == 0.5 and x.get("depth_tol_m") == 0.02
          and "BOTH_VALID_SUPPORT" in x][0]
    d6 = [x for x in rows if x.get("alpha_cover") == 0.5 and x.get("depth_tol_m") == 0.06
          and "BOTH_VALID_SUPPORT" in x][0]
    check("a 3 cm offset is unexplained at 2 cm and explained at 6 cm (the sweep is informative)",
          d1["B_SUPPORTED_F_UNEXPLAINED"] > d6["B_SUPPORTED_F_UNEXPLAINED"])


def t14_invalid():
    print("\n[T14] GT-invalid pixels are never missing geometry")
    from observation_diagnostics import gaussian_evidence as ge
    from observation_diagnostics import pixel_evidence as pe
    cams = cams_pair()
    P = plate(0.0, 0.0, 0.0, 0.8, 0.6, 11, (0.8, 0.5, 0.2))
    st = make_state(P)
    left = plate(-0.2, 0.0, 0.0, 0.4, 0.6, 6, (0.8, 0.5, 0.2))
    sF = make_state(left)
    gt = gt_of(st, cams[0])
    W = gt["depth"].shape[1]
    inval = torch.zeros_like(gt["valid"])
    inval[:, W // 2:] = True
    gt["valid"] = gt["valid"] & ~inval
    gt["depth"] = torch.where(gt["valid"], gt["depth"], torch.zeros_like(gt["depth"]))
    vF, cF = render(sF, cams[0])
    vB, cB = render(st, cams[0])
    th = pe.Thresholds()
    f = pe.classify({"rgb": vF.rgb, "depth": vF.depth, "alpha": vF.alpha},
                    {"rgb": vB.rgb, "depth": vB.depth, "alpha": vB.alpha}, gt, th)
    right = inval
    check("every GT-invalid pixel is GT_INVALID in the support map",
          bool((f["support"][right] == 4).all()))
    check("none is B_SUPPORTED_F_UNEXPLAINED or NEITHER_VALID_SUPPORT",
          int(((f["support"] == 2) | (f["support"] == 0))[right].sum()) == 0)
    check("no witness pixel on GT-invalid pixels", int(f["W_Bonly"][right].sum()) == 0)
    acc = ge.new_frame_accumulator(st.mu.shape[0], "cuda")
    ge.accumulate(acc, cB, vB.cache.depth, f, gt["depth"], "B", 0, th.contrib_depth_tol_m)
    from observation_diagnostics.aggregate import project
    u, _v, _z = project(cams[0], st.mu.cpu().numpy())
    radius = vB.cache.radii.cpu().numpy()
    only_invalid = (u - radius) > W // 2          # whole footprint in the GT-invalid half
    w = acc["witness"].cpu().numpy()
    check("Gaussians seen only where GT is invalid carry no witness",
          only_invalid.sum() > 0 and float(w[only_invalid].sum()) == 0.0,
          f"{int(only_invalid.sum())} such Gaussians, witness {w[only_invalid].sum():.3g}")
    m = pe.view_metrics(f, {"alpha": vF.alpha, "depth": vF.depth, "rgb": vF.rgb},
                        {"alpha": vB.alpha, "depth": vB.depth, "rgb": vB.rgb}, th)
    check("B's coverage there is reported as GT_INVALID_BUT_B_COVERS", m["GT_INVALID_BUT_B_COVERS"] > 0)


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    with tempfile.TemporaryDirectory() as td:
        t1_conservation()
        t3_identical()
        t4_layers()
        t5_occlusion(Path(td) / "t5")
        t6_wrong_geometry()
        t7_unobservable()
        t8_fail_closed(Path(td) / "t8")
        t9_pairs_exhaustive()
        t11_sweeps()
        t14_invalid()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
