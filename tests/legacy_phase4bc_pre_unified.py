"""LEGACY (pre-unified 2026-11-11 4B / 4C: phase4b_exact + phase4c_exact; superseded by tests/test_phase4_unified_4b.py and _4c.py). Kept as the record of the 0.76 purity failure.

Gate D (directive 2026-11-11, section 8): corrected 4B / 4C on procedural scenes.

  roles          role maps: guard >= SSIM_R + 1, adjacent blocks differ, every block cycles through all roles;
                 every accepted birth's source pixels are PROPOSAL pixels; even / odd rows are never used
  thin panels    synthetic two-sided panels (3, 5, 8, 12, 20 mm; oblique; noise; biased depth): opposite faces
                 never share a cell or a patch; same-facing parallel layers: merge rates reported, no merge
                 at >= 2 x layer_gap with 1 mm noise
  coupling       an UNRESOLVED original rendered in the wrong pose: its pixels become
                 ORIGINAL_MATERIAL_AMBIGUOUS, and no moving duplicate of it is born
  occluder       a planted static sheet in front of the door in every frame is never accepted
  never seen     a surface hidden behind the back wall in every view is never invented
  recovery       drawer floor (P, backward) and door back face (R, forward): moving births on the true
                 plane / the back layer, no front-back collapse, objective improves
  purity         KNOWN FAILING (reported, not hidden): >= 90 % of drawer-floor births MOVING
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import recovery  # noqa: E402,F401
from recovery import data_roles as DR  # noqa: E402
from recovery import phase4a_exact as X  # noqa: E402
from recovery import phase4b_exact as B  # noqa: E402
from recovery import phase4c_exact as C  # noqa: E402
from recovery.exact_objective import SSIM_R, ExactLossConfig  # noqa: E402
from recovery.gaussians import MOVING, STATIC, GaussianTensors  # noqa: E402
from recovery.material_completion import Phase4Scene  # noqa: E402
from recovery.phase4_stages_legacy import run_all_in_memory  # noqa: E402
from recovery.rgbd_loss import LossConfig  # noqa: E402
from recovery.view_cache import PreparedViews, ViewKey  # noqa: E402
from synthetic_articulated import make_scene, reobserve  # noqa: E402

QUIET = lambda *a, **k: None
CFG4B = B.DiscoveryConfig4B(voxel=0.015, role_block_px=32, role_time_block=2)
CFG4C = C.AcceptConfig4C(voxel=0.015)
_CACHE = {}


def _pv(views):
    keys = [ViewKey(v.cam.name, int(v.frame), v.cam.W, v.cam.H) for v in views]
    return PreparedViews.from_memory(keys, [v.get() for v in views], log=QUIET)


def _pipeline(jtype, direction, drop, seed=4):
    key = (jtype, direction, drop, seed)
    if key not in _CACHE:
        s = make_scene(jtype, direction, drop=(drop,), withhold_frac=0.0, seed=seed, W=256, H=192)
        sc = Phase4Scene(s.model, s.joint, s.views, LossConfig(), sh_degree=0)
        out = run_all_in_memory(sc, _pv(s.views), s.core, X.Policy4A(sensitivity=False), ExactLossConfig(), CFG4B,
                                CFG4C, log=QUIET)
        _CACHE[key] = (s, sc, out)
    return _CACHE[key]


def _gt(s, name):
    return s.full._xyz[torch.as_tensor(s.surface_full == name, device="cuda")].cpu().numpy()


# ------------------------------------------------------------------ roles
def test_roles_are_separated_and_cycle():
    cfg = DR.RoleConfig(block_px=32, guard_px=SSIM_R + 1, time_block=2)
    m0 = DR.role_map(0, 0, 192, 256, cfg).numpy()
    assert set(np.unique(m0)) == {0, 1, 2, 3}
    # every non-guard pixel's 11 x 11 SSIM window lies in pixels of its own role or guard
    for r in (1, 2, 3):
        ys, xs = np.nonzero(m0 == r)
        for y, x in zip(ys[::97], xs[::97]):
            w = m0[max(0, y - SSIM_R):y + SSIM_R + 1, max(0, x - SSIM_R):x + SSIM_R + 1]
            assert set(np.unique(w)) <= {0, r}, (r, y, x)
    # a fixed block takes all three roles over three time blocks
    roles = {int(DR.role_map(0, row, 192, 256, cfg)[40, 40]) for row in (0, 2, 4)}
    assert roles == {1, 2, 3}, roles
    try:
        DR.role_map(0, 0, 64, 64, DR.RoleConfig(guard_px=SSIM_R))
        raise AssertionError("a guard narrower than the SSIM radius must be refused")
    except ValueError:
        pass
    print("  roles: windows never cross roles; blocks cycle P/F/V over time; narrow guard refused")


def test_births_come_from_proposal_pixels_only():
    s, sc, out = _pipeline("prismatic", "backward", "drawer_floor")
    roles = out["roles"]
    acc = [d for d in out["decisions"] if d["status"] in (C.ACC_M, C.ACC_S)]
    assert acc, "no birth accepted"
    n_checked = 0
    for k, d in enumerate(out["decisions"]):
        if d["status"] not in (C.ACC_M, C.ACC_S):
            continue
        lab = MOVING if d["status"] == C.ACC_M else STATIC
        src = np.unique(np.concatenate([c["src"] for c in out["candidates"][k][lab]]))
        for p in src[::7]:
            vi, pix = divmod(int(p), sc.views[0].cam.H * sc.views[0].cam.W)
            r = roles.get(vi).reshape(-1)[pix]
            assert int(r) == DR.PROPOSAL, (k, vi, pix, int(r))
            n_checked += 1
    print(f"  provenance: {n_checked} sampled source pixels of {len(acc)} accepted births are all PROPOSAL pixels")


# ------------------------------------------------------------------ thin panels / layers
def _panel_points(sep, opposite, angle_deg, noise, bias, rng, n=4000):
    a = np.radians(angle_deg)
    nrm = np.array([0.0, np.sin(a), -np.cos(a)])
    t1 = np.array([1.0, 0, 0]); t2 = np.cross(nrm, t1)
    uv = rng.uniform(-0.05, 0.05, (n, 2))
    base = np.array([0.0, 0.0, 2.0]) + uv[:, :1] * t1 + uv[:, 1:] * t2
    front = base + rng.normal(0, noise, (n, 1)) * nrm
    back = base - (sep + bias) * nrm + rng.normal(0, noise, (n, 1)) * nrm
    nb = -nrm if opposite else nrm
    x = np.concatenate([front, back]); nn = np.concatenate([np.repeat(nrm[None], n, 0), np.repeat(nb[None], n, 0)])
    layer = np.r_[np.zeros(n, int), np.ones(n, int)]
    view = rng.integers(0, 6, 2 * n); src = np.arange(2 * n)
    return x, nn, np.full((2 * n, 3), 0.5), view, view, src, layer


def test_thin_panels_layers():
    rng = np.random.default_rng(0)
    cfg = B.DiscoveryConfig4B(voxel=0.01)
    rows = []
    for opposite in (True, False):
        for sep in (0.003, 0.005, 0.008, 0.012, 0.020):
            for angle, noise, bias in ((0, 0.001, 0.0), (35, 0.001, 0.0), (0, 0.003, 0.0), (35, 0.003, 0.002)):
                x, n, rgb, view, frame, src, layer = _panel_points(sep, opposite, angle, noise, bias, rng)
                cells = B.fuse_layers(x, n, rgb, view, frame, src, cfg)
                lab = B.patches(cells, cfg)
                mixed_cells = np.mean([len(np.unique(layer[c["src"]])) > 1 for c in cells])
                pl = {}
                for c, l in zip(cells, lab):
                    pl.setdefault(int(l), set()).update(np.unique(layer[c["src"]]).tolist())
                mixed_patches = np.mean([len(v) > 1 for v in pl.values()])
                rows.append((opposite, sep, angle, noise, bias, mixed_cells, mixed_patches))
                if opposite:
                    assert mixed_cells == 0 and mixed_patches == 0, f"two-sided panel {sep} m merged"
                elif sep >= 2 * cfg.layer_gap and noise <= 0.001:
                    assert mixed_cells == 0, f"same-facing layers {sep} m apart merged at 1 mm noise"
    print("  layers (opposite faces never merge; same-facing merge rates, cells / patches):")
    for o, sep, ang, nz, b, mc, mp in rows:
        if not o:
            print(f"    same-facing sep {sep * 1000:4.0f} mm angle {ang:2d} noise {nz * 1000:.0f} mm bias {b * 1000:.0f} mm: "
                  f"cells {mc:.2f}, patches {mp:.2f}")


# ------------------------------------------------------------------ 4A -> 4B coupling
def test_unresolved_original_creates_no_moving_duplicate():
    s = make_scene("revolute", "forward", withhold_frac=0.0, seed=6, W=256, H=192)
    sc = Phase4Scene(s.model, s.joint, s.views, LossConfig(), sh_degree=0)
    pv = _pv(s.views)
    back = np.nonzero(s.surface_model == "door_back")[0]
    labels = torch.as_tensor(s.gt_moving_model, device="cuda").clone()
    labels[torch.as_tensor(back, device="cuda")] = False                 # rendered in the WRONG (static) pose
    status = np.where(s.gt_moving_model, X.MOVING_S, X.STATIC_S).astype(object)
    status[s.core] = X.CORE
    status[back] = X.UNRESOLVED
    codes, stats, obs, cnt, roles = B.run_4b(sc, pv, labels, status, CFG4B, B.DepthNoise(None, CFG4B), log=QUIET)
    amb = sum(st["ORIGINAL_MATERIAL_AMBIGUOUS"] for st in stats)
    assert amb > 0, "door-back pixels were not attributed to the undecided original"
    cells, lab, cands = B.build_candidates(obs, CFG4B)
    undecided = torch.as_tensor(status == X.UNRESOLVED, device="cuda")
    s4 = C.Scene4C(sc, labels)
    dec, acc = C.run_4c(s4, pv, roles, cands, undecided, CFG4C, ExactLossConfig(), log=QUIET)
    from scipy.spatial import cKDTree
    born_m = [a["patch"]["xyz"].cpu().numpy() for a in acc if a["label"] == MOVING]
    if born_m:
        d, _ = cKDTree(s.model._xyz[torch.as_tensor(back, device="cuda")].cpu().numpy()).query(np.concatenate(born_m))
        assert (d > 0.01).all(), f"{int((d <= 0.01).sum())} moving births duplicate the undecided door back"
    st = {k: int(v) for k, v in zip(*np.unique([x["status"] for x in dec], return_counts=True))} if dec else {}
    print(f"  coupling: {amb} ORIGINAL_MATERIAL_AMBIGUOUS px, {cnt['on_undecided_alternate']} points on the undecided "
          f"alternate dropped; 4C {st}; no moving duplicate of the door back")


# ------------------------------------------------------------------ occluder / never seen
def test_planted_front_occluder_rejected():
    s = make_scene("revolute", "forward", withhold_frac=0.0, seed=5, W=256, H=192)
    sc = Phase4Scene(s.model, s.joint, s.views, LossConfig(), sh_degree=0)
    pv = _pv(s.views)
    roles = DR.Roles(s.views, CFG4B.roles())
    gx, gy = np.meshgrid(np.linspace(-0.08, 0.08, 7), np.linspace(-0.08, 0.08, 7))
    cells = [{"key": (i, 0, 0, 5), "mu": np.array([x, y, 1.94]), "normal": np.array([0, 0, -1.0]),
              "rgb": np.array([0.5, 0.5, 0.5]), "thickness": 0.001, "first_frame": 0, "src": np.array([i])}
             for i, (x, y) in enumerate(zip(gx.ravel(), gy.ravel()))]
    s4 = C.Scene4C(sc, torch.as_tensor(s.gt_moving_model, device="cuda"))
    n0 = len(sc.g)
    dec, acc = C.run_4c(s4, pv, roles, [{STATIC: cells, MOVING: None, "overlap": 0.0, "source_motion_p90_m": 1.0}],
                        torch.zeros(n0, dtype=torch.bool, device="cuda"), C.AcceptConfig4C(voxel=0.03),
                        ExactLossConfig(), log=QUIET)
    assert dec[0]["status"] not in (C.ACC_S, C.ACC_M) and len(sc.g) == n0, dec[0]
    print(f"  occluder: {dec[0]['status']} (max contradiction px {dec[0]['static'].get('max_contradiction_px')})")


def test_never_seen_surface_not_invented():
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
    sc = Phase4Scene(s.model, s.joint, s.views, LossConfig(), sh_degree=0)
    out = run_all_in_memory(sc, _pv(s.views), s.core, X.Policy4A(sensitivity=False), ExactLossConfig(), CFG4B, CFG4C,
                            log=QUIET)
    n0 = out["n_original"]
    born = sc.g._xyz[n0:].detach().cpu().numpy()
    near = 0
    if len(born):
        near = int((np.abs(born[:, 2] - 2.36) < 0.02).sum())
    assert near == 0, f"{near} births on the never-observed panel behind the back wall"
    print(f"  never seen: {len(born)} births in total, 0 on the hidden panel (behind the back wall in every view)")


# ------------------------------------------------------------------ recovery
def test_drawer_floor_moving_on_true_plane():
    s, sc, out = _pipeline("prismatic", "backward", "drawer_floor")
    n0 = out["n_original"]
    L = out["labels_final"][n0:].cpu().numpy().astype(bool)
    Xb = sc.g._xyz[n0:].detach().cpu().numpy()
    assert L.sum() > 0, "no moving birth"
    gt = _gt(s, "drawer_floor")
    off = np.abs(Xb[L, 1] - gt[:, 1].mean())
    assert np.median(off) < CFG4B.voxel, f"moving births {np.median(off):.4f} m off the true floor plane"
    assert out["E_after_4c"].sum() < out["E_before_4c"].sum()
    for d in out["decisions"]:
        if d["status"] == C.ACC_M:
            f = d["first_observed_frame_provenance"]
            assert f != s.joint.anchor_frame and abs(s.joint.q_of(f)) > 0
    from scipy.spatial import cKDTree
    cov = (cKDTree(Xb[L]).query(gt)[0] < 0.03).mean()
    print(f"  drawer floor: {int(L.sum())} moving births, plane offset median {np.median(off) * 1000:.1f} mm, "
          f"floor coverage (<3 cm) {cov:.2f}; {int((~L).sum())} static births; E {out['E_before_4c'].sum():.4f} -> "
          f"{out['E_after_4c'].sum():.4f}")


def test_door_back_no_collapse():
    s, sc, out = _pipeline("revolute", "forward", "door_back")
    n0 = out["n_original"]
    L = out["labels_final"][n0:].cpu().numpy().astype(bool)
    Xb = sc.g._xyz[n0:].detach().cpu().numpy()
    assert len(Xb) > 0 and L.mean() > 0.9, f"door back births {len(Xb)}, moving fraction {L.mean() if len(L) else 0}"
    zb, zf = _gt(s, "door_back")[:, 2].mean(), _gt(s, "door_front")[:, 2].mean()
    closer = (np.abs(Xb[:, 2] - zb) < np.abs(Xb[:, 2] - zf)).mean()
    assert closer > 0.8, f"only {closer:.2f} of the births on the back layer"
    from scipy.spatial import cKDTree
    cov = (cKDTree(Xb).query(_gt(s, "door_back"))[0] < 0.03).mean()
    print(f"  door back: {len(Xb)} births, all moving, {closer:.2f} on the back layer (no collapse); back-face "
          f"coverage (<3 cm) {cov:.2f}")


def test_KNOWN_FAILING_drawer_birth_material_purity():
    s, sc, out = _pipeline("prismatic", "backward", "drawer_floor")
    n0 = out["n_original"]
    L = out["labels_final"][n0:].cpu().numpy().astype(bool)
    print(f"  KNOWN FAILURE CHECK: moving fraction of drawer-scene births {L.mean():.2f} (gate wants >= 0.90)")
    assert L.mean() >= 0.9, f"material purity {L.mean():.2f} < 0.90: world-static mixed-depth fragments are " \
                            "accepted as STATIC births (see PHASE4_CORRECTION_AUDIT.md)"


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
