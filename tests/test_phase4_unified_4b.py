"""Unified spec 2026-11-11, section 5.8 (4B) and the 4B side of section 4.

  refuse        4B refuses without a 4A complete.json and on a tampered 4A artifact
  mixed edge    the 19 / 80 pre-unified failures (frozen fixture): every source pixel is traced, with the ground-truth
                scene's per-surface visible weights, to a MIXTURE of surfaces whose expected depth lies > 5 mm from every
                contributing surface; the new taxonomy classifies them as non-birth codes (MIXED_EDGE_AMBIGUOUS from the
                measured depth alone), none of them reaches a candidate, and single-surface pixels are rarely flagged
  no births     4B adds nothing, leaves the 4A artifact, the original Gaussians and the joint byte-identical
  thin layers   ray-cast two-face panels (opposite-facing and same-facing; 3 / 5 / 8 / 12 / 20 mm; 0 and 35 deg
                obliquity; 1 and 3 mm noise; a per-camera bias): opposite faces never share a patch; same-facing layers
                are reported per case and never merged once the separation exceeds the declared resolvability
                2 k sigma sqrt(Var + 1) of the per-view two-sided line test
  never seen    a panel hidden behind the back wall in every view yields no 4B source sample
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import recovery  # noqa: E402,F401
from recovery import phase4_artifacts as AR  # noqa: E402
from recovery import phase4a_exact as X  # noqa: E402
from recovery import phase4b_surface as B  # noqa: E402
from recovery.canonical_fusion import backproject  # noqa: E402
from recovery.exact_objective import ExactLossConfig  # noqa: E402
from recovery.gaussians import GaussianTensors  # noqa: E402
from recovery.phase4_stages import stage_4a, stage_4b  # noqa: E402
from recovery.views import camera_from_K  # noqa: E402
from phase4_ctx import QUIET, ctx_from_scene, surface_layers  # noqa: E402
from synthetic_articulated import make_scene, reobserve  # noqa: E402

CFG4B = B.Surface4B(voxel=0.015, role_block_px=32, role_time_block=2, sigma_depth_assumed=0.001)
FIX = Path(__file__).resolve().parent / "fixtures" / "phase4_pre_unified_drawer_births.npz"
_RUN = {}


def _drawer():
    if "drawer" not in _RUN:
        fx = np.load(FIX)
        sp = json.loads(str(fx["scene"]))
        s = make_scene(sp["jtype"], sp["direction"], drop=(sp["drop"],), withhold_frac=sp["withhold_frac"], seed=sp["seed"],
                       W=sp["W"], H=sp["H"])
        ctx = ctx_from_scene(s)
        tmp = tempfile.mkdtemp(prefix="p4b_")
        ra = stage_4a(ctx, tmp, X.Policy4A(sensitivity=False), ExactLossConfig(), log=QUIET)
        assert ra["complete"], ra.get("reasons")
        h_a = AR.dir_hash(ra["dir"])
        rb = stage_4b(ctx, tmp, ra["dir"], CFG4B, log=QUIET)
        _RUN["drawer"] = (s, ctx, tmp, ra, rb, h_a, fx)
    return _RUN["drawer"]


def test_4b_refuses_without_valid_4a():
    s, ctx, tmp, ra, rb, h_a, fx = _drawer()
    with tempfile.TemporaryDirectory() as t2:
        empty = Path(t2) / AR.STAGE_A
        empty.mkdir()
        for bad in (empty,):
            try:
                stage_4b(ctx, t2, bad, CFG4B, log=QUIET)
                raise AssertionError("4B ran without a complete 4A")
            except SystemExit:
                pass
        import shutil
        copy = Path(t2) / "copy" / AR.STAGE_A
        shutil.copytree(ra["dir"], copy)
        z = dict(np.load(copy / "membership.npz"))
        z["render_label"] = 1 - z["render_label"]
        np.savez_compressed(copy / "membership.npz", **z)
        try:
            stage_4b(ctx, Path(t2) / "copy", copy, CFG4B, log=QUIET)
            raise AssertionError("4B ran on a tampered 4A artifact")
        except SystemExit as e:
            msg = str(e)
    print(f"  refuse: no complete.json -> refused; relabelled membership.npz -> refused ({msg[:60]}...)")


def test_4b_zero_births_and_no_mutation():
    s, ctx, tmp, ra, rb, h_a, fx = _drawer()
    assert rb["complete"]
    assert len(ctx.scene.g) == len(s.model)
    assert AR.dir_hash(ra["dir"]) == h_a, "4B modified the 4A artifact"
    cb = AR.require_complete(rb["dir"], AR.STAGE_B)
    assert cb["births"] == 0 and cb["phase4a_artifact_sha256"] == ra["payload"]["artifact_sha256"]
    for f in ("scan_manifest.json", "residual_taxonomy.npz", "moving_frontier.npz", "source_observations.npz",
              "layer_graph.json", "candidates.npz", "candidate_provenance.jsonl", "ambiguous_measurements.json",
              "unexplained_coverage_report.json", "complete.json"):
        assert (rb["dir"] / f).is_file(), f
    assert AR.canonical_hashes(ctx.scene.g)["combined"] == ra["payload"]["canonical_sha256"]
    print(f"  no births: {cb['candidates']} candidate patches, 0 births; 4A artifact, canonical originals and joint "
          f"unchanged; sigma_model {cb['sigma_model_m']} m ({cb['sensor']['sigma_model_source']})")


def test_4b_mixed_edge_failures_traced():
    s, ctx, tmp, ra, rb, h_a, fx = _drawer()
    rows = fx["rows"]
    W = int(fx["W"])
    codes = np.load(rb["dir"] / "residual_taxonomy.npz")["codes"]
    inv = {v: k for k, v in B.CODES.items()}
    static_rows = rows[rows[:, 1] == 0]
    assert int(fx["births_static"]) == 19 and len(static_rows) > 0
    phys_bad, new_codes = [], []
    fp_n = fp_d = 0
    for vi in np.unique(static_rows[:, 2]):
        v = s.views[int(vi)]
        names, Wt, Dd = surface_layers(s, v.cam, v.frame)
        D = v.get().depth.cuda()
        for p in static_rows[static_rows[:, 2] == vi][:, 3]:
            y, x = divmod(int(p), W)
            w = Wt[:, y, x].cpu().numpy()
            dd = Dd[:, y, x].cpu().numpy()
            sig = w > 0.05
            dev = float(np.min(np.abs(float(D[y, x]) - dd[sig])))
            phys_bad.append((sig.sum() > 1) and dev > 0.005)
            new_codes.append(inv[int(codes[int(vi)].reshape(-1)[int(p)])])
    phys_bad = np.array(phys_bad)
    birth = np.isin(new_codes, ["COVERAGE_GAP", "FRONT_SURFACE"])
    mixed = np.array(new_codes) == "MIXED_EDGE_AMBIGUOUS"
    # every source sample that reached a candidate: never one of the traced pixels
    S = np.load(rb["dir"] / "source_observations.npz")
    cand = torch.load(rb["dir"] / "candidates.pt", weights_only=False)
    used = set()
    for c in cand:
        used |= set(zip(S["view"][c["sources"]].tolist(), S["pix"][c["sources"]].tolist()))
    reached = sum((int(v), int(p)) in used for _, _, v, p in static_rows)
    # false flags on single-surface pixels (GT weight > 0.99 of one surface), all views
    for vi, v in enumerate(s.views):
        names, Wt, Dd = surface_layers(s, v.cam, v.frame)
        pure = (Wt.max(0).values > 0.99).cpu().numpy() & v.get().depth_valid.numpy()
        fp_d += int(pure.sum())
        fp_n += int((pure & (codes[vi] == B.CODES["MIXED_EDGE_AMBIGUOUS"])).sum())
    print(f"  mixed edge: {len(static_rows)} source pixels of the 19 old STATIC births; physically invalid (>= 2 surfaces, "
          f"> 5 mm from every contributing surface): {phys_bad.mean():.3f}; new codes "
          f"{dict(zip(*np.unique(new_codes, return_counts=True)))}; reach a candidate: {reached}; single-surface pixels "
          f"flagged MIXED: {fp_n}/{fp_d} = {fp_n / max(fp_d, 1):.3f}")
    assert phys_bad.mean() >= 0.95, "the frozen failures were expected to be physical mixtures"
    assert mixed.mean() >= 0.9 and (~birth).mean() >= 0.95, (mixed.mean(), (~birth).mean())
    assert reached == 0, f"{reached} traced mixed pixels still reach a candidate"
    assert fp_n / max(fp_d, 1) < 0.05


# ------------------------------------------------------------------ thin layers (ray-cast)
def _cam(eye, target, W=320, H=240, name="c"):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    zc = target - eye; zc /= np.linalg.norm(zc)
    up = np.array([0.0, -1.0, 0.0])
    xc = np.cross(up, zc); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    c2w = np.eye(4); c2w[:3, :3] = np.stack([xc, yc, zc], 1); c2w[:3, 3] = eye
    f = W / (2 * math.tan(math.radians(20)))
    return camera_from_K(name, [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], c2w, W, H)


def _raycast(cam, planes, sigma, bias, rng):
    """planes: (origin, t1, t2, extent1 (a,b), extent2 (a,b), layer, outward normal). Nearest FRONT-side hit per pixel
    (a face is seen only from the side its outward normal points to, as for a solid panel)."""
    K, c2w = cam.K, cam.c2w
    H, W = cam.H, cam.W
    vv, uu = np.mgrid[0:H, 0:W]
    d_cam = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0], (vv + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    R, C = c2w[:3, :3], c2w[:3, 3]
    dw = d_cam @ R.T
    best = np.full((H, W), np.inf)
    lay = np.full((H, W), -1)
    for o, t1, t2, e1, e2, L, n_out in planes:
        n = np.cross(t1, t2)
        den = dw @ n
        tt = ((o - C) @ n) / np.where(np.abs(den) > 1e-12, den, np.nan)
        P = C + tt[..., None] * dw
        a, b = (P - o) @ t1, (P - o) @ t2
        hit = (tt > 0) & (a >= e1[0]) & (a <= e1[1]) & (b >= e2[0]) & (b <= e2[1]) & (tt < best) & ((dw @ n_out) < 0)
        best[hit] = tt[hit]
        lay[hit] = L
    D = np.where(np.isfinite(best), best, 0.0)               # axial depth: d_cam has z = 1
    ok = D > 0
    D = np.where(ok, D + rng.normal(0, sigma, D.shape) + bias, 0.0).astype(np.float32)
    return D, ok, lay


def _layer_case(sep, opposite, angle, sigma, bias, rng):
    a = math.radians(angle)
    nrm = np.array([0.0, math.sin(a), -math.cos(a)])          # faces the front cameras (they sit at smaller z)
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.cross(nrm, t1)
    o = np.array([0.0, 0.0, 2.0])
    if opposite:
        planes = [(o, t1, t2, (-0.10, 0.10), (-0.10, 0.10), 0, nrm),
                  (o - sep * nrm, t1, t2, (-0.10, 0.10), (-0.10, 0.10), 1, -nrm)]
        eyes = [o + 1.0 * nrm + np.array([dx, 0.0, 0.0]) for dx in (-0.3, 0.0, 0.3)] + \
               [o - (1.0 + sep) * nrm + np.array([dx, 0.0, 0.0]) for dx in (-0.3, 0.0, 0.3)]
    else:
        planes = [(o, t1, t2, (-0.12, 0.04), (-0.10, 0.10), 0, nrm), (o - sep * nrm, t1, t2, (-0.02, 0.14), (-0.10, 0.10), 1, nrm)]
        eyes = [o + 1.0 * nrm + np.array([dx, dy, 0.0]) for dx, dy in ((-0.3, 0.0), (0.0, 0.0), (0.3, 0.0), (0.15, 0.2))]
    S = {k: [] for k in ("view", "cam", "frame", "row", "pix", "D", "sigma", "rgb", "yW", "nW", "xM", "nM", "camS", "camM",
                         "group")}
    layer, store = [], {}
    for vi, e in enumerate(eyes):
        cam = _cam(e, o, name=f"c{vi}")
        D, ok, lay = _raycast(cam, planes, sigma, bias * (1 if vi % 2 else -1), rng)
        sel = ok.copy()
        sel[:1] = sel[-1:] = False
        sel[:, :1] = sel[:, -1:] = False
        yw, nw, pc = backproject(cam, D, sel)
        good = np.isfinite(nw).all(1)
        idx = np.flatnonzero(sel)[good]
        n = len(idx)
        C = cam.c2w[:3, 3]
        for k, v in (("view", np.full(n, vi)), ("cam", np.full(n, vi)), ("frame", np.zeros(n, int)), ("row", np.zeros(n, int)),
                     ("pix", idx), ("D", D.reshape(-1)[idx]), ("sigma", np.full(n, sigma)), ("rgb", np.full((n, 3), 0.5)),
                     ("yW", yw[good]), ("nW", nw[good]), ("xM", yw[good]), ("nM", nw[good]),
                     ("camS", np.repeat(C[None], n, 0)), ("camM", np.repeat(C[None], n, 0)), ("group", np.full(n, vi))):
            S[k].append(v)
        layer.append(lay.reshape(-1)[idx])
        store[vi] = (D, ok, np.full(D.shape, sigma), cam.H, cam.W)
    S = {k: np.concatenate(v) for k, v in S.items()}
    return S, np.concatenate(layer), store


def test_4b_thin_layers():
    rng = np.random.default_rng(0)
    rows = []
    for opposite in (True, False):
        for sep in (0.003, 0.005, 0.008, 0.012, 0.020):
            for angle, sigma, bias in ((0, 0.001, 0.0), (35, 0.001, 0.0), (0, 0.003, 0.0), (35, 0.003, 0.002)):
                cfg = B.Surface4B(voxel=0.01, sigma_depth_assumed=sigma, min_samples=6)
                S, layer, store = _layer_case(sep, opposite, angle, sigma, bias, rng)
                frag = B.fragments(S, lambda v: store[v][0], lambda v: store[v][1], lambda v: store[v][2],
                                   lambda v: store[v][3], lambda v: store[v][4], cfg)
                cells, lab, co, corner = B.patches_from(S, frag, np.arange(len(S["view"])), "M", cfg)
                mixed_samples = 0
                for l in np.unique(lab):
                    mem = np.concatenate([cells[i]["members"] for i in np.nonzero(lab == l)[0]])
                    fr = np.bincount(layer[mem], minlength=2) / len(mem)
                    if fr.min() >= 0.05:
                        mixed_samples += len(mem)
                rows.append((opposite, sep, angle, sigma, bias, mixed_samples / len(layer)))
    from recovery.phase4b_surface import _line_weights
    var = _line_weights(5, 1.0)[1]
    bound = lambda sg: 2 * CFG4B.k_sigma * math.sqrt(var + 1) * sg     # step clearing both one-sided line tests
    print("  thin layers: fraction of samples in a patch holding both layers (>= 5 % each)")
    for o, sep, ang, sg, b, m in rows:
        tag = "opposite" if o else "same-facing"
        print(f"    {tag:11s} sep {sep * 1000:4.0f} mm angle {ang:2d} noise {sg * 1000:.0f} mm bias {b * 1000:.0f} mm: {m:.2f}"
              + ("" if o else f"   (declared resolvable above {bound(sg) * 1000:.1f} mm)"))
    for o, sep, ang, sg, b, m in rows:
        if o:
            assert m == 0, f"opposite faces {sep} m apart share a patch"
        elif sep > bound(sg) + 2 * b:
            assert m == 0, f"same-facing layers {sep} m apart (> declared resolvability) merged ({m:.2f})"


def test_4b_never_observed_panel_gives_no_sources():
    s = make_scene("revolute", "forward", withhold_frac=0.0, seed=7, W=256, H=192)
    g = s.full
    n = 36
    gx, gy = np.meshgrid(np.linspace(-0.15, 0.15, 6), np.linspace(-0.1, 0.1, 6))
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32, device="cuda")
    hidden = GaussianTensors(t(np.stack([gx.ravel(), gy.ravel(), np.full(n, 2.36)], 1)), t(np.tile([1.0, 0, 0, 0], (n, 1))),
                             torch.log(t(np.tile([0.02, 0.02, 0.002], (n, 1)))), torch.logit(t(np.full((n, 1), 0.9))),
                             t(np.full((n, 1, 3), 1.2)), torch.zeros(n, g._features_rest.shape[1], 3, device="cuda"),
                             g.max_sh_degree, {"material": np.zeros(n, np.int8),
                                               "gaussian_uid": np.arange(len(g), len(g) + n, dtype=np.int64),
                                               "parent_uid": np.full(n, -1, np.int64),
                                               "original_id": np.arange(len(g), len(g) + n, dtype=np.int64),
                                               "provenance": np.ones(n, np.int8)})
    full = GaussianTensors.concat(g, hidden)
    full.active_sh_degree = 0
    s.views = reobserve(s, full, np.concatenate([np.isin(s.surface_full, ["door_front", "door_back"]), np.zeros(n, bool)]))
    ctx = ctx_from_scene(s)
    with tempfile.TemporaryDirectory() as tmp:
        ra = stage_4a(ctx, tmp, X.Policy4A(sensitivity=False), ExactLossConfig(), log=QUIET)
        rb = stage_4b(ctx, tmp, ra["dir"], CFG4B, log=QUIET)
        S = rb["prop"]["samples"]
        near = int((np.abs(S["yW"][:, 2] - 2.36) < 0.02).sum()) if len(S["yW"]) else 0
    assert near == 0, f"{near} source samples on the never-observed panel"
    print(f"  never seen: {len(S['view'])} source samples in total, 0 within 2 cm of the hidden panel")


if __name__ == "__main__":
    failed = []
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                failed.append(name)
    raise SystemExit(1 if failed else 0)
