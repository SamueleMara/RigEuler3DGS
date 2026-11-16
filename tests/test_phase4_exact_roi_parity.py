"""Gate B (directive 2026-11-11, 4.3): the exact ROI scorer equals full-image rendering and scoring.

For every case, every view and every tested group B (S <-> M flip, and OMIT):
  * the evaluator's fast render (PosedPair) is BITWISE the production render (render_articulated)
  * every pixel outside the conservative footprint (kernel radii + projected mean, padded one tile, both
    poses) is BITWISE identical between the two full hypothesis renders (RGB, alpha, Z)
  * SSIM evaluated on a crop is BITWISE the full-image SSIM map at the same pixels (fixed-order sums)
  * per channel, the ROI delta equals the sum of the full-image per-pixel difference maps to float64
    rounding (|err| <= 1e-15 + 1e-12 |delta|), and the difference of full-image view losses to
    TOL_TOTAL x (|L_alt| + |L_base|) (subtracting two O(1) float64 sums)
  * views the evaluator skips (off-screen in both poses / identity transform) have a full-image delta of
    exactly 0
  * the Phase-4 SSIM equals utils.loss_utils.ssim (float32 conv reference) to 2e-6
Cases: revolute forward and prismatic backward procedural objects; late-visible geometry (drawer floor),
occluded geometry (door back behind the front), large projected Gaussians, near-alpha-threshold opacity,
Gaussians at the image border and outside the frustum, exact depth ties in one tile, a group crossing
the depth order of static surfaces, background pixels, invalid / missing observed depth, and a joint move of
several groups. Optional: one real red_fridge view (skipped when the data is not mounted).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import recovery  # noqa: E402,F401
from recovery import footprint as FP  # noqa: E402
from recovery import phase4a_exact as X  # noqa: E402
from recovery.exact_objective import CHANNELS, SSIM_R, ExactLossConfig, ViewTargets  # noqa: E402
from recovery.gaussians import GaussianTensors  # noqa: E402
from recovery.material_completion import Phase4Scene  # noqa: E402
from recovery.rgbd_loss import LossConfig  # noqa: E402
from recovery.view_cache import PreparedViews, ViewKey  # noqa: E402
from synthetic_articulated import make_scene  # noqa: E402

TOL_TOTAL = 1e-12
DEV = "cuda"


def _prepared(views):
    keys = [ViewKey(v.cam.name, int(v.frame), v.cam.W, v.cam.H) for v in views]
    return PreparedViews.from_memory(keys, [v.get() for v in views], log=lambda *a, **k: None)


def _stress(s, seed=0):
    """Append stress Gaussians to the model: large, faint (near 1/255 alpha), at / beyond the border,
    exact depth ties, and a thin two-sided layer just in front of the body."""
    g = s.model
    rng = np.random.default_rng(seed)
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32, device=DEV)
    xyz = [[0.0, 0.0, 2.15], [0.05, 0.02, 1.98], [0.051, 0.02, 1.98], [0.051, 0.02, 1.98],     # large, ties
           [0.6, -0.2, 2.1], [2.5, 0.0, 2.0], [0.0, 0.0, -1.0],                                  # border / outside
           [-0.1, 0.05, 1.99], [-0.1, 0.05, 1.991], [0.1, -0.1, 2.0]]                            # two-sided, faint
    n = len(xyz)
    rot = np.tile([1.0, 0, 0, 0], (n, 1))
    scl = [[0.08, 0.08, 0.08], [0.01, 0.01, 0.002], [0.01, 0.01, 0.002], [0.01, 0.01, 0.002],
           [0.05, 0.05, 0.01], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05],
           [0.03, 0.03, 0.0005], [0.03, 0.03, 0.0005], [0.02, 0.02, 0.02]]
    op = [0.6, 0.9, 0.9, 0.5, 0.8, 0.8, 0.8, 0.9, 0.9, 1.2 / 255]
    dc = (rng.uniform(0.1, 0.9, (n, 3)) - 0.5) / 0.28209479177387814
    K = g._features_rest.shape[1]
    lin = {"material": np.zeros(n, np.int8), "gaussian_uid": np.arange(len(g), len(g) + n, dtype=np.int64),
           "parent_uid": np.full(n, -1, np.int64), "original_id": np.arange(len(g), len(g) + n, dtype=np.int64),
           "provenance": np.ones(n, np.int8)}
    add = GaussianTensors(t(xyz), t(rot), torch.log(t(scl)), torch.logit(t(op)[:, None]), t(dc[:, None, :]),
                          torch.zeros(n, K, 3, device=DEV), g.max_sh_degree, lin)
    m = GaussianTensors.concat(g, add)
    m.active_sh_degree = 0
    return m, np.arange(len(g), len(g) + n)


def _invalidate_depth(views):
    """A rectangle of missing observed depth inside the mask, in every view (no depth observation there)."""
    out = []
    for vr in views:
        v = vr.get()
        H, W = v.depth.shape
        dv = v.depth_valid.clone()
        dv[H // 3:H // 2, W // 3:W // 2] = False
        d = v.depth.clone()
        d[~dv & v.mask] = 0.0
        from recovery.views import View
        vv = View(v.camera, v.frame, v.rgb, d, v.mask, dv)
        out.append(type(vr)(vr.cam, vr.frame, (lambda vv=vv: vv)))
    return out


def _check(scene, pv, labels, items, omit=True, tag=""):
    ev = X.ExactEvaluator(scene, pv, ExactLossConfig(), log=lambda *a, **k: None)
    rS, rM, ident = ev.footprints()
    gS = np.stack([X.union_rects(ev.group_rects(it)[0]) for it in items])
    gM = np.stack([X.union_rects(ev.group_rects(it)[1]) for it in items])
    r = ev.sweep(items, labels, (gS, gM), omit=omit)
    worst = {"roi_vs_maps": 0.0, "roi_vs_totals_rel": 0.0, "outside_changed": 0, "omit": 0.0, "skipped_nonzero": 0.0}
    for vi, vr in enumerate(scene.views):
        tgt = ViewTargets(pv.gpu_view(vi, DEV), ev.cfg)
        base = scene.render(vr, labels)
        pp = X.PosedPair(scene, vr, scene.transport)
        fb = pp.render(labels)
        for k in ("color", "alpha", "zdepth"):
            assert torch.equal(base[k], fb[k]), f"{tag}: fast render != production render ({k})"
        bm = tgt.full_maps(base)
        # SSIM on a crop is bitwise the full map
        H, W = tgt.H, tgt.W
        box = (H // 4, H // 4 + 17, W // 5, W // 5 + 23)
        assert torch.equal(tgt.ssim_map(base, box), bm["rgb_ssim"][box[0]:box[1], box[2]:box[3]]), f"{tag}: SSIM crop"
        ci = ev.cam_of_view[vi]
        for k, it in enumerate(items):
            alt = labels.clone()
            for g in it:
                ix = torch.as_tensor(g.members, device=DEV)
                alt[ix] = not bool(labels[ix[0]])
            a = scene.render(vr, alt)
            am = tgt.full_maps(a)
            full_maps = torch.stack([(am[c] - bm[c]).sum() for c in CHANNELS]).cpu().numpy()
            full_tot = np.array([float(am[c].sum() - bm[c].sum()) for c in CHANNELS])
            scale = float(sum(am[c].sum() + bm[c].sum() for c in CHANNELS))
            if r["skip"][k, vi] != X.SKIP_NONE:
                worst["skipped_nonzero"] = max(worst["skipped_nonzero"], float(np.abs(full_maps).max()))
                continue
            # every changed pixel lies inside the padded footprint union (both poses)
            inside = torch.zeros(H, W, dtype=torch.bool, device=DEV)
            for R in (gS[k, ci], gM[k, vi]):
                if FP.rect_nonempty(R):
                    inside[R[1] * 16:R[3] * 16, R[0] * 16:R[2] * 16] = True
            diff = (a["color"] != base["color"]).any(0) | (a["alpha"][0] != base["alpha"][0]) | \
                   (a["zdepth"][0] != base["zdepth"][0])
            worst["outside_changed"] += int((diff & ~inside).sum())
            d = r["d_alt"][k, vi]
            e = np.abs(d - full_maps)
            assert (e <= 1e-15 + 1e-12 * np.abs(full_maps)).all(), f"{tag}: ROI vs full maps {e}"
            worst["roi_vs_maps"] = max(worst["roi_vs_maps"], float(e.max()))
            worst["roi_vs_totals_rel"] = max(worst["roi_vs_totals_rel"], float(np.abs(d - full_tot).max()) / scale)
            if omit and len(it) == 1:
                keep = torch.ones_like(labels)
                keep[torch.as_tensor(it[0].members, device=DEV)] = False
                om = tgt.full_maps(scene.render(vr, labels, keep=keep))
                ref = float(sum((om[c] - bm[c]).sum() for c in CHANNELS))
                worst["omit"] = max(worst["omit"], abs(float(r["omit"][k, vi]) - ref))
    assert worst["outside_changed"] == 0, f"{tag}: {worst['outside_changed']} changed pixels outside the footprint"
    assert worst["roi_vs_totals_rel"] <= TOL_TOTAL, f"{tag}: {worst}"
    assert worst["omit"] <= 1e-14, f"{tag}: OMIT {worst}"
    assert worst["skipped_nonzero"] == 0.0, f"{tag}: a skipped view changes the loss {worst}"
    print(f"  {tag}: {len(items)} items x {len(scene.views)} views; max |ROI - full maps| {worst['roi_vs_maps']:.2e}, "
          f"max rel vs totals {worst['roi_vs_totals_rel']:.2e}, OMIT {worst['omit']:.1e}, "
          f"skipped {int((r['skip'] != 0).sum())}")
    return r


def _groups_from(s, model, names, extra=(), rng=None, n_random=6):
    rng = rng or np.random.default_rng(0)
    out = []
    for nm in names:
        idx = np.nonzero(s.surface_model == nm)[0]
        if len(idx):
            out.append(X.Group(len(out), idx[:24]))
    for e in extra:
        out.append(X.Group(len(out), np.asarray(e)))
    N0 = len(s.surface_model)
    for _ in range(n_random):
        out.append(X.Group(len(out), np.sort(rng.choice(N0, rng.integers(1, 12), replace=False))))
    return out


def _labels_uniform(model, core, groups):
    lab = torch.zeros(len(model), dtype=torch.bool, device=DEV)
    lab[torch.as_tensor(core, device=DEV)] = True
    for g in groups:                                   # a group renders with one label (that of its first row)
        ix = torch.as_tensor(g.members, device=DEV)
        lab[ix] = bool(lab[ix[0]])
    return lab


def _case(jtype, direction, names, tag, invalid_depth=False, stress=True):
    s = make_scene(jtype, direction, withhold_frac=0.4, seed=3, W=112, H=84)
    model, extra = _stress(s) if stress else (s.model, np.zeros(0, int))
    views = _invalidate_depth(s.views) if invalid_depth else s.views
    scene = Phase4Scene(model, s.joint, views, LossConfig(), sh_degree=0)
    pv = _prepared(views)
    groups = _groups_from(s, model, names, extra=[extra[:4], extra[4:7], extra[7:9], extra[9:]] if stress else ())
    core = np.setdiff1d(s.core, np.concatenate([g.members for g in groups]))
    labels = _labels_uniform(model, core, groups)
    items = [[g] for g in groups] + [groups[:3]]       # single flips + one joint move
    _check(scene, pv, labels, items, tag=tag)


def test_revolute_forward_occluded_back_and_stress():
    _case("revolute", "forward", ["door_back", "door_front", "left", "back"], "R/forward")


def test_prismatic_backward_late_visible_floor_and_invalid_depth():
    _case("prismatic", "backward", ["drawer_floor", "drawer_left", "bottom", "top"], "P/backward + invalid depth",
          invalid_depth=True)


def test_ssim_matches_reference_implementation():
    from utils.loss_utils import ssim
    s = make_scene("revolute", "forward", withhold_frac=0.0, seed=5, W=96, H=72)
    scene = Phase4Scene(s.model, s.joint, s.views, LossConfig(), sh_degree=0)
    vr = s.views[5]
    tgt = ViewTargets(vr.get().to(DEV), ExactLossConfig())
    out = scene.render(vr, torch.zeros(len(s.model), dtype=torch.bool, device=DEV))
    mine = 1 - float(tgt.ssim_map(out).sum()) / tgt.w_ssim / tgt.HW
    ref = float(ssim((out["color"] * tgt.Mf)[None], tgt.MI[None]))
    assert abs(mine - ref) < 2e-6, (mine, ref)
    print(f"  SSIM: phase-4 float64 {mine:.8f} vs utils.loss_utils {ref:.8f}")


def test_real_red_fridge_view():
    RES = Path("/media/samuelemara/Extreme SSD/Samuele/results_11_10_26")
    REAL = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real")
    n = "final_red_fridge_episode_000000"
    R = RES / "ours_v6_two_cams" / n / "unanimous" / "pipeline" / n
    if not (R / "bidirectional" / "forward_joint").is_dir() or not (REAL / n).is_dir():
        print("  SKIP: real red_fridge data not mounted")
        return
    import argparse
    from recovery.io_setup import add_common_cli, setup_phase4
    ap = argparse.ArgumentParser()
    add_common_cli(ap)
    a = ap.parse_args(["--source", str(REAL / n), "--direction", "forward", "--model", str(R / "static_bidi/initial"),
                       "--phase2-dir", str(R / "bidirectional/forward_rigid"), "--phase3-dir",
                       str(R / "bidirectional/forward_joint"), "--joint-type", "revolute", "--cameras",
                       "cam_001,cam_002", "--output", "/tmp/unused"])
    a.view_subset = 2
    up, cams, views, pv, _ = setup_phase4(a, log=lambda *x, **k: None)
    g = GaussianTensors.load_ply(up.ply).to(DEV)
    g.active_sh_degree = 0
    scene = Phase4Scene(g, up.joint, views, LossConfig(), sh_degree=0)
    rng = np.random.default_rng(0)
    cand = np.setdiff1d(np.arange(len(g)), up.members)
    xyz = g._xyz.detach().cpu().numpy()
    near = cand[np.argsort(np.linalg.norm(xyz[cand] - xyz[up.members].mean(0), axis=1))]
    groups = [X.Group(0, np.sort(near[:64])), X.Group(1, np.sort(rng.choice(cand, 40, replace=False))),
              X.Group(2, np.sort(near[500:700]))]
    labels = _labels_uniform(g, up.members, groups)
    _check(scene, pv, labels, [[x] for x in groups], tag="real red_fridge (2 frames x 2 cameras)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
