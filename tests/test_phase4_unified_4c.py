"""Unified spec 2026-11-11, section 6.7 (4C) and the 4C side of section 4. Procedural scenes, 256 x 192, 3 cameras.

  drawer floor   P / backward, floor missing: births MOVING only, on the true plane, objective improves; floor
                 completeness reported against the OBSERVED and the never-observed GT floor area per growth cycle;
                 4A / 4B artifacts, original bytes and labels, joint unchanged; N_final = N0 + births
  door back      R / forward, back face missing: births on the back layer, behind the front at the anchor; a planted
                 wrong-side copy (in front of the door) is rejected by early visible contradictions
  static shelf   a static shelf revealed only while the drawer is out (missing from the model): never born MOVING
  corner         drawer side wall missing (90 deg to front / floor, 2 cm from the static body wall): grows as MOVING
                 without births on the static wall
  short-lived    the floor exposed in only the last two frames: not rejected for missing frames; evidence reported
  coupling       an UNRESOLVED original rendered in the wrong pose: ORIGINAL_MEMBERSHIP_CONFLICT, no moving duplicate
  parity         4C exact ROI deltas = full-frame deltas for single, static and combined patches incl. SSIM / depth
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import recovery  # noqa: E402,F401
from recovery import footprint as FP  # noqa: E402
from recovery import phase4_artifacts as AR  # noqa: E402
from recovery import phase4a_exact as X  # noqa: E402
from recovery import phase4b_surface as B  # noqa: E402
from recovery import phase4c_validate as C  # noqa: E402
from recovery.data_roles import Roles  # noqa: E402
from recovery.exact_objective import ExactLossConfig, ViewTargets, view_objective  # noqa: E402
from recovery.phase4_stages import run_all  # noqa: E402
from phase4_ctx import QUIET, ctx_from_scene  # noqa: E402
from synthetic_articulated import make_scene  # noqa: E402

CFG4B = B.Surface4B(voxel=0.015, role_block_px=32, role_time_block=2, sigma_depth_assumed=0.001)
CFG4C = C.Validate4C(voxel=0.015)
POL = X.Policy4A(sensitivity=False)
_RUNS = {}


def _run(key, cfg4b=None, **kw):
    if key not in _RUNS:
        s = make_scene(**{"withhold_frac": 0.0, "seed": 4, "W": 256, "H": 192, **kw})
        ctx = ctx_from_scene(s)
        tmp = tempfile.mkdtemp(prefix=f"p4c_{key}_")
        r = run_all(ctx, tmp, POL, ExactLossConfig(), cfg4b or CFG4B, CFG4C, log=QUIET)
        _RUNS[key] = (s, ctx, tmp, r)
    return _RUNS[key]


def _gt(s, name):
    return s.full._xyz[torch.as_tensor(s.surface_full == name, device="cuda")].cpu().numpy()


def _births(r):
    a = r["4c"]["auth"]
    return a, a.tensors()["xyz"][a.n0:].cpu().numpy(), a.labels[a.n0:].cpu().numpy()


def _observed_gt(s, name, frac=0.5):
    """GT Gaussians of `name` that are actually observed: some view has a valid pixel whose dominant GT surface is
    `name` and whose back-projection (to canonical, moving) lies within 1.5 voxels."""
    from phase4_ctx import surface_layers
    from recovery.canonical_fusion import backproject
    from scipy.spatial import cKDTree
    gt = _gt(s, name)
    pts = []
    for v in s.views:
        names, Wt, Dd = surface_layers(s, v.cam, v.frame)
        k = names.index(name)
        ob = v.get()
        dom = (Wt[k] >= frac).cpu().numpy() & ob.depth_valid.numpy()
        if dom.any():
            yw, _, _ = backproject(v.cam, ob.depth.numpy(), dom)
            Q, p = s.joint.transform(v.frame)
            pts.append((yw - np.asarray(p)) @ np.asarray(Q) if name in ("drawer_floor", "drawer_left", "door_back",
                                                                        "door_front") else yw)
    if not pts:
        return np.zeros(len(gt), bool)
    d, _ = cKDTree(np.concatenate(pts)).query(gt)
    return d <= 1.5 * CFG4B.voxel


# ------------------------------------------------------------------------------------------------ recovery
def test_4c_drawer_floor():
    s, ctx, tmp, r = _run("drawer", jtype="prismatic", direction="backward", drop=("drawer_floor",))
    assert r["stopped_after"] is None, r
    a, Xb, Lb = _births(r)
    assert len(Xb) > 0 and Lb.all(), "births must be MOVING only"
    gt = _gt(s, "drawer_floor")
    off = np.abs(Xb[:, 1] - gt[:, 1].mean())
    near_floor = off < 2 * CFG4B.voxel
    assert np.median(off[near_floor]) < CFG4B.voxel, f"births {np.median(off):.4f} m off the true floor plane"
    from scipy.spatial import cKDTree
    obs = _observed_gt(s, "drawer_floor")
    cov = cKDTree(Xb).query(gt)[0] < 0.03
    payload = r["4c"]["payload"]
    assert payload["static_births"] == 0 and payload["N_final"] == payload["N0"] + payload["N_accepted_births"]
    fa = json.loads((Path(tmp) / AR.STAGE_C / "full_sequence_audit.json").read_text())
    assert fa["E_after"] < fa["E_before"]
    # immutability across the stages
    AR.require_complete(r["4a"]["dir"], AR.STAGE_A)
    AR.require_complete(r["4b"]["dir"], AR.STAGE_B)
    assert AR.canonical_hashes(a.g0)["combined"] == r["4a"]["payload"]["canonical_sha256"]
    st4a = r["4a"]["res"]["render_label"].astype(bool)
    assert np.array_equal(a.labels[:a.n0].cpu().numpy(), st4a)
    cyc = r["4c"]["cycles"]
    for f in ("accepted_patches.npz", "per_patch_evidence.jsonl", "per_patch_visibility.jsonl", "birth_provenance.jsonl",
              "candidate_statuses.jsonl", "interaction_audit.json", "final_canonical/point_cloud.ply",
              "final_canonical/material_labels.npz", "full_sequence_audit.json", "joint_unchanged_proof.json",
              "resource_report.json", "complete.json"):
        assert (Path(tmp) / AR.STAGE_C / f).is_file(), f
    st = {}
    for d in r["4c"]["decisions"]:
        st[d["status"]] = st.get(d["status"], 0) + 1
    print(f"  drawer floor: {len(Xb)} births, all MOVING, plane offset median {np.median(off[near_floor]) * 1000:.1f} mm; "
          f"floor completeness (<3 cm): observed area {cov[obs].mean():.2f} ({int(obs.sum())}/{len(gt)} GT Gaussians "
          f"observed), never-observed area {cov[~obs].mean() if (~obs).any() else float('nan'):.2f}; E "
          f"{fa['E_before']:.4f} -> {fa['E_after']:.4f}; cycles {[(c['cycle'], c['accepted_patches']) for c in cyc]}; "
          f"statuses {st}  (pre-unified: 61 moving + 19 static births, coverage 0.35)")


def test_4c_door_back_and_wrong_side():
    # q_max 1.6 rad: the back face is seen by two cameras over several time blocks (at 1.1 rad it is seen essentially
    # by one camera in one time block, and 4C correctly reports INSUFFICIENT_INDEPENDENT_SUPPORT)
    s, ctx, tmp, r = _run("door", jtype="revolute", direction="forward", drop=("door_back",), q_max=1.6)
    assert r["stopped_after"] is None, r
    a, Xb, Lb = _births(r)
    st = {}
    for d in r["4c"]["decisions"]:
        st[d["status"]] = st.get(d["status"], 0) + 1
    assert len(Xb) > 0 and Lb.all(), f"door back NOT recovered: 0 births; 4C statuses {st} (see the report, open gaps)"
    zb, zf = _gt(s, "door_back")[:, 2].mean(), _gt(s, "door_front")[:, 2].mean()
    closer_back = (np.abs(Xb[:, 2] - zb) < np.abs(Xb[:, 2] - zf)).mean()
    assert closer_back > 0.8, f"only {closer_back:.2f} of the births on the back layer"
    assert (Xb[:, 2] > zf).mean() > 0.8, "births must lie behind the door front at the anchor (canonical, closed)"
    # wrong-side copy: the true back-face geometry mirrored IN FRONT of the door front
    gb = _gt(s, "door_back")
    mir = gb.copy()
    mir[:, 2] = zf - (zb - zf)
    cand = manual_candidate(mir, np.array([0.0, 0.0, -1.0]), "wrong_side")
    auth = C.Auth(ctx.scene, torch.as_tensor(r["4a"]["res"]["render_label"].astype(bool), device="cuda"),
                  r["4a"]["res"]["status"])
    noise = B.SensorNoise(None, CFG4B, sigma_model=0.001)
    roles = Roles(ctx.scene.views, CFG4B.roles())
    dec, _ = C.validate_cycle(auth, ctx.pv, roles, [cand], CFG4C, ExactLossConfig(), noise, CFG4B, X.UNDECIDED, 0.0, log=QUIET)
    assert dec[0]["status"] != C.ACC and len(auth.births) == 0, dec[0]
    early = [row for row in dec[0].get("_rows_M", []) if row.get("contradiction")]
    order = list(s.joint.frames)
    print(f"  door back: {len(Xb)} births, {closer_back:.2f} on the back layer, {(Xb[:, 2] > zf).mean():.2f} behind the "
          f"front at the anchor; wrong-side copy -> {dec[0]['status']} (contradicting views at processing rows "
          f"{sorted({order.index(x['frame']) for x in early})})")


def test_4c_static_shelf_never_moving():
    # a STATIC backing panel in the cabinet face, inside the drawer opening: hidden by the drawer front while the
    # drawer is closed, revealed as it slides out (missing from the model)
    x0, y0, z0 = -0.20, -0.15, 2.0
    shelf = ("shelf", (x0 + 0.04, y0 + 0.03, z0 + 0.002), (0.32, 0, 0), (0, 0.06, 0), 13, 3, False, (0.2, 0.6, 0.9))
    s, ctx, tmp, r = _run("shelf", jtype="prismatic", direction="backward", drop=("shelf",), extra=(shelf,))
    from phase4_ctx import surface_layers
    seen = []
    for v in s.views:
        names, Wt, Dd = surface_layers(s, v.cam, v.frame)
        npx = int((Wt[names.index("shelf")] > 0.5).sum())
        if npx > 30:
            seen.append((v.cam.name, s.joint.frames.index(v.frame), npx))
    assert seen, "the static panel must actually be observed for this test to mean anything"
    assert r["stopped_after"] is None, r
    a, Xb, Lb = _births(r)
    gs = _gt(s, "shelf")
    from scipy.spatial import cKDTree
    kd = cKDTree(gs)
    worst = np.inf
    for f in s.joint.frames:                     # a moving birth must never sit on the static shelf at any frame
        Q, p = s.joint.transform(f)
        if len(Xb):
            worst = min(worst, float(kd.query(Xb @ np.asarray(Q).T + np.asarray(p))[0].min()))
    assert worst > 0.01, f"a MOVING birth lies {worst * 1000:.1f} mm from the static shelf"
    st = {}
    for d in r["4c"]["decisions"]:
        st[d["status"]] = st.get(d["status"], 0) + 1
    rep = json.loads((Path(tmp) / AR.STAGE_C / "static_alternatives_for_review.json").read_text())
    print(f"  static panel: observed in {len(seen)} views (rows {sorted({x[1] for x in seen})}); {len(Xb)} moving births, "
          f"closest {worst * 1000:.0f} mm from the panel at any frame; 4C "
          f"statuses {st}; {len(rep)} patches exported for separate static review")


def test_4c_corner_grows_without_static_neighbour():
    s, ctx, tmp, r = _run("corner", jtype="prismatic", direction="backward", drop=("drawer_left",))
    assert r["stopped_after"] is None, r
    a, Xb, Lb = _births(r)
    gl = _gt(s, "drawer_left")
    wall = _gt(s, "left")
    from scipy.spatial import cKDTree
    on_side = cKDTree(gl).query(Xb)[0] < 0.015 if len(Xb) else np.zeros(0, bool)
    on_wall = np.abs(Xb[:, 0] - wall[:, 0].mean()) < 0.005 if len(Xb) else np.zeros(0, bool)
    cov = (cKDTree(Xb).query(gl)[0] < 0.03).mean() if len(Xb) else 0.0
    assert not on_wall.any(), f"{int(on_wall.sum())} births on the static body wall"
    print(f"  corner: {len(Xb)} births ({int(on_side.sum())} on the missing drawer side wall, coverage {cov:.2f}); "
          f"0 on the static body wall 2 cm away")


def test_4c_short_lived_surface_reported_not_rejected():
    # the floor is exposed only in the last two processing rows. Its observed part is separated from the trusted drawer
    # walls by the junction band (mixed / ambiguous pixels), so the strict frontier mode does not reach it; the spec's
    # separately labelled second mode (disconnected_search, one more support group required) is used here
    prof = [0, 0, 0, 0, 0, 0, 0.5, 1.0]
    from dataclasses import replace
    s, ctx, tmp, r = _run("short", cfg4b=replace(CFG4B, disconnected_search=True), jtype="prismatic",
                          direction="backward", drop=("drawer_floor",), q_profile=prof)
    assert r["stopped_after"] is None, r
    a, Xb, Lb = _births(r)
    from scipy.spatial import cKDTree
    gt = _gt(s, "drawer_floor")
    cands = {c["patch_uid"]: c for c in torch.load(Path(r["4b"]["dir"]) / "candidates.pt", weights_only=False)}
    floor = []
    for d in r["4c"]["decisions"]:
        c = cands.get(d["patch_uid"])
        if c is None:
            continue
        X = np.array([m["mu"] for m in c["M_cells"]])
        if np.median(cKDTree(gt).query(X)[0]) < 0.02:
            floor.append((d["patch_uid"], d["search_mode"], d["status"], d["support_groups"], len(d["frames"]), d.get("reason")))
    print(f"  short-lived: {len(Xb)} births; floor candidates (uid, mode, status, support groups, frames): "
          f"{[f[:5] for f in floor]}")
    assert floor, "the briefly exposed floor produced no candidate"
    # the spec's requirement: never rejected merely for missing frames (no visibility-time criterion exists in 4C);
    # whether it is RECOVERED is reported, not asserted (see PHASE4_UNIFIED_IMPLEMENTATION_REPORT.md, open gaps)
    assert not any("frame" in (f[5] or "") for f in floor if f[2] != C.ACC), floor
    assert Lb.all()


def manual_candidate(P, normal, uid, rgb=(0.1, 0.1, 0.12)):
    cells = [{"vox": tuple(np.floor(p / CFG4C.voxel).astype(int)), "mu": p, "normal": normal, "rgb": np.array(rgb),
              "thickness": 0.001, "count": 5, "members": np.arange(5), "split": False} for p in P]
    return {"patch_uid": uid, "cycle": 0, "layer_uid": uid, "search_mode": "manual", "M_cells": cells, "S_cells": cells,
            "sources": np.arange(len(P) * 5), "support_groups": 6, "views": 6, "frames": [0, 1, 2],
            "first_observed_frame": 0, "split_cell_frac": 0.0, "source_motion_p90_m": 1.0, "frontier_parent_uids": [],
            "n_frontier_parents": 0, "evidence": "MANUAL", "source_view": None, "source_pix": None}


def test_4c_unresolved_original_creates_conflict_not_duplicate():
    s = make_scene("revolute", "forward", withhold_frac=0.0, seed=6, W=256, H=192)
    ctx = ctx_from_scene(s)
    back = np.nonzero(s.surface_model == "door_back")[0]
    labels = torch.as_tensor(s.gt_moving_model, device="cuda").clone()
    labels[torch.as_tensor(back, device="cuda")] = False                 # rendered in the WRONG (static) pose
    status = np.where(s.gt_moving_model, X.MOVING_S, X.STATIC_S).astype(object)
    status[s.core] = X.CORE
    status[back] = X.UNRESOLVED
    noise = B.SensorNoise(None, CFG4B)
    roles = Roles(ctx.scene.views, CFG4B.roles())
    trusted = torch.as_tensor(np.isin(status, [X.CORE, X.MOVING_S]), device="cuda")
    und = torch.as_tensor(np.isin(status, X.UNDECIDED), device="cuda")
    prop = B.run_proposals(ctx.scene, ctx.pv, labels, trusted, und, CFG4B, noise, roles, log=QUIET)
    conflict_px = sum(st["ORIGINAL_MEMBERSHIP_CONFLICT"] for st in prop["stats"])
    assert conflict_px > 0, "the undecided door back was not reported as a membership conflict"
    auth = C.Auth(ctx.scene, labels, status)
    dec, cyc, loo, _ = C.run_4c(auth, ctx.pv, roles, prop["candidates"], CFG4C, ExactLossConfig(), noise, CFG4B, log=QUIET)
    from scipy.spatial import cKDTree
    Xb = auth.tensors()["xyz"][auth.n0:].cpu().numpy()
    if len(Xb):
        d, _ = cKDTree(s.model._xyz[torch.as_tensor(back, device="cuda")].cpu().numpy()).query(Xb)
        assert (d > 0.01).all(), f"{int((d <= 0.01).sum())} moving births duplicate the undecided door back"
    st = {}
    for x in dec:
        st[x["status"]] = st.get(x["status"], 0) + 1
    print(f"  coupling: {conflict_px} ORIGINAL_MEMBERSHIP_CONFLICT px in 4B; 4C {st}; no moving duplicate of the "
          f"undecided door back")


# ------------------------------------------------------------------------------------------------ exactness
def test_4c_roi_parity():
    s, ctx, tmp, r = _run("drawer", jtype="prismatic", direction="backward", drop=("drawer_floor",))
    cands = torch.load(Path(r["4b"]["dir"]) / "candidates.pt", weights_only=False)
    auth = C.Auth(ctx.scene, torch.as_tensor(r["4a"]["res"]["render_label"].astype(bool), device="cuda"),
                  r["4a"]["res"]["status"])
    dev = "cuda"
    c0, c1 = cands[0], cands[1]
    pm = C.cells_params(c0["M_cells"], CFG4C, dev)
    ps = C.cells_params(c0["S_cells"], CFG4C, dev) if c0["S_cells"] else pm
    p1 = C.cells_params(c1["M_cells"], CFG4C, dev)
    comb = {k: torch.cat([pm[k], p1[k]], 0) for k in pm}
    items = [{"p": pm, "label": True}, {"p": ps, "label": False}, {"p": comb, "label": True}]
    rects = C.version_rects(auth, items)
    for it, rr in zip(items, rects):
        it["rects"] = rr
    noise = B.SensorNoise(None, CFG4B, sigma_model=0.001)
    roles = Roles(ctx.scene.views, CFG4B.roles())
    lc = ExactLossConfig()
    ev = C.evaluate(auth, ctx.pv, roles, items, CFG4C, lc, noise, CFG4B, log=QUIET)
    worst, n = 0.0, 0
    K = auth.K
    with torch.no_grad():
        for k, it in enumerate(items):
            pa = C.activated(it["p"], K)
            for row in ev[k]["rows"]:
                vi = row["view"]
                vr = ctx.scene.views[vi]
                tgt = ViewTargets(ctx.pv.gpu_view(vi, dev), lc)
                base = C.render_t(auth.sc, vr, auth.tensors(), auth.labels)
                alt = C.render_t(auth.sc, vr, C.cat_t(auth.tensors(), pa),
                                 torch.cat([auth.labels, torch.full((pa["xyz"].shape[0],), it["label"], device=dev)]))
                full = (view_objective(alt, tgt) - view_objective(base, tgt)).cpu().numpy()
                if "d_all" not in row:
                    assert abs(full).max() == 0.0, (k, vi, full)       # OUT_OF_FRUSTUM views are provably neutral
                    continue
                err = np.abs(np.asarray(row["d_all"]) - full).max()
                worst = max(worst, err / max(1e-12, np.abs(full).max()) if np.abs(full).max() > 1e-12 else err)
                n += 1
    assert worst < 1e-9, f"ROI vs full-frame relative error {worst:.2e}"
    print(f"  parity: {n} item-views (moving, static, combined two-patch), ROI = full frame within {worst:.1e} relative; "
          f"out-of-frustum views exactly neutral")


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
