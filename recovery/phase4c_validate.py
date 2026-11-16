"""Phase 4C (unified spec 2026-11-11, section 6): validate NEW moving patches before any permanent birth.

Consumes READ-ONLY 4B candidates and the immutable 4A scene. Never relabels an original, never moves an original,
never changes the joint. Per candidate patch P (canonical cells from 4B):

    H_0 = G                 ABSENT
    H_M = G u T_t(P_M)      MOVING (the one fixed joint, every frame -- a birth exists at every frame)
    H_S = G u P_S           STATIC competitor, fused from exactly the same source samples (diagnostic only:
                            a static winner is NOT_MOVING_STATIC_ALTERNATIVE and is exported, never committed)

Fit: candidate parameters only (rotation, log-scale, opacity, DC; means stay at the measured cell centres), FIT-role
pixels, fixed full-view denominators, `refine_steps` Adam steps over the views that see the patch.

Evaluation (view-major, 6.5): one baseline render per view; every candidate version whose kernel footprint (padded
tile rect from the kernel's own radii) is non-empty in that view is added and scored EXACTLY on the footprint boxes
dilated by the SSIM radius (exact_objective.box_delta; pixels outside are bitwise unchanged), all pixels and the
VALIDATION role separately. Views where the footprint is empty are proven zero-effect (OUT_OF_FRUSTUM).
Visibility (6.2) from the renderer's own depth-ordered compositing: an auxiliary pass with indicator colours gives
the candidate's visible responsibility A_P(p) = sum_{i in P} w_i(p) (all static + moving occluders included). Where
A_P >= alpha_vis the pixel is attributed to the candidate and the FULL composite with the candidate (alpha A', depth
D' = Z'/A', i.e. the same expected-depth compositing the renderer uses everywhere) is compared with the measurement:
    VISIBLE_BACKGROUND_CONTRADICTION   M = 0 and the candidate newly makes A' >= 0.5
    VISIBLE_DEPTH_CONTRADICTION        reliable depth, D' < D - tol, and the baseline was not already in front there
                                       (the candidate would hide the measured surface)
    OBSERVED_NEARER_OCCLUDER           reliable depth and D' > D + tol      (neutral: a nearer measured surface)
    VISIBLE_MATCH                      reliable depth otherwise
    INVALID_OR_MIXED_DEPTH             M = 1 without reliable depth (invalid or MIXED_EDGE_AMBIGUOUS): neutral
'reliable' = valid and not flagged by the observation-side mixed-depth detector (phase4b_surface.mixed_depth_map).

Existing-geometry correction shadow (6.3.3): the originals / births within shadow_reach of the patch get a bounded
fit of location (+-shadow_max_shift), scale (x 1/f..f) and opacity -- not appearance -- on the same FIT pixels; its exact validation gain over EVERY view where
their footprint (before or after) is non-empty is compared with the candidate's. The shadow is discarded.
Joint sensitivity (6.3.5): the moving candidate's validation gain under the declared whole-joint perturbations
(phase4a_exact.perturbed_joints), on the views of its nominal footprint; the joint is never changed.

Decision (declared, uncalibrated policy; nothing is a probability):
    ORIGINAL_MEMBERSHIP_CONFLICT        cells inside the render support of an UNDECIDED original -> return to 4A
    REJECTED_DUPLICATE                  cells inside the support of a decided original / an accepted birth
    REJECTED_MIXED_MEASUREMENT          the patch's cells are layer mixtures (split_cell_frac >= mixed_cell_frac)
    INSUFFICIENT_INDEPENDENT_SUPPORT    < min_support_groups independent source groups or < min_val_pixels
                                        changed VALIDATION pixels (short-lived surfaces are reported, not rejected)
    REJECTED_VISIBLE_CONTRADICTION      some view still contradicts after contradiction-driven pruning
    NOT_MOVING_STATIC_ALTERNATIVE       the static competitor passes and beats moving by more than ambiguity_rel
    AMBIGUOUS_MATERIAL                  S and M within ambiguity_rel, or S-M source divergence < identifiable distance
    REJECTED_NO_IMPROVEMENT             moving validation gain (minus the complexity penalty) <= margin
    REJECTED_EXISTING_GEOMETRY_CORRECTION   the shadow correction explains >= shadow_frac of the gain
    JOINT_SENSITIVE                     a declared joint perturbation removes the gain or lets STATIC win
                                        (contradictions under perturbation are recorded, not decisive)
    ACCEPTED_MOVING                     only this creates authoritative births
    UNTESTED_RESOURCE_LIMIT             a declared resource limit stopped the stage before the candidate was tested
Growth (6.4): accepted non-interacting patches commit atomically; the scene is re-scanned with the new trusted moving
frontier (4B code, nothing mutated); repeat while genuinely new supported patches are accepted (<= max_cycles).
Final: each accepted patch leave-one-out against the FINAL scene (interaction audit) and a full all-view audit.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import footprint as FP
from .articulated_scene import posed, render_articulated
from .canonical_fusion import cells_to_gaussian_params
from .data_roles import FIT, VALIDATION
from .exact_objective import CHANNELS, SSIM_R, ExactLossConfig, ViewTargets, box_delta, box_delta_weighted
from .gaussians import MOVING, PROV_PHASE4_BIRTH, GaussianTensors

ACC = "ACCEPTED_MOVING"
CONFLICT, DUP, MIXED = "ORIGINAL_MEMBERSHIP_CONFLICT", "REJECTED_DUPLICATE", "REJECTED_MIXED_MEASUREMENT"
INSUFF, CONTRA, STATIC_ALT = "INSUFFICIENT_INDEPENDENT_SUPPORT", "REJECTED_VISIBLE_CONTRADICTION", "NOT_MOVING_STATIC_ALTERNATIVE"
AMBIG, NOIMP, SHADOW = "AMBIGUOUS_MATERIAL", "REJECTED_NO_IMPROVEMENT", "REJECTED_EXISTING_GEOMETRY_CORRECTION"
JSENS, UNTESTED = "JOINT_SENSITIVE", "UNTESTED_RESOURCE_LIMIT"
STATUSES = (ACC, AMBIG, INSUFF, MIXED, DUP, CONTRA, STATIC_ALT, JSENS, UNTESTED, CONFLICT, NOIMP, SHADOW)
VIS = ("VISIBLE_MATCH", "VISIBLE_DEPTH_CONTRADICTION", "VISIBLE_BACKGROUND_CONTRADICTION", "OBSERVED_NEARER_OCCLUDER",
       "INVALID_OR_MIXED_DEPTH", "NOT_VISIBLE_IN_MODEL", "OUT_OF_FRUSTUM")


@dataclass
class Validate4C:
    refine_steps: int = 20
    lr_xyz: float = 0.0
    tangent_factor: float = 0.45
    lr_rot: float = 1e-3
    lr_scale: float = 5e-3
    lr_opacity: float = 5e-2
    lr_dc: float = 2.5e-2
    voxel: float = 0.01
    margin_px: float = 1e-3                   # minimal validation gain, pixel-equivalent loss units
    complexity_px_per_gaussian: float = 1e-4  # declared penalty per added Gaussian
    ambiguity_rel: float = 0.25
    identifiable_voxels: float = 2.0
    min_val_pixels: int = 50
    min_support_groups: int = 2
    mixed_cell_frac: float = 0.5
    alpha_vis: float = 0.3
    contradiction_px: int = 16
    contradiction_frac: float = 0.05
    prune_rounds: int = 2
    dup_tol: float = 0.003
    dup_frac: float = 0.5
    support_k: float = 3.0
    shadow: bool = True
    shadow_reach_voxels: float = 2.0
    shadow_max: int = 256
    shadow_steps: int = 20
    shadow_lr_xyz: float = 2e-4
    shadow_max_shift_voxels: float = 0.5      # bounded correction: |dmu| <= min(this x voxel, k sqrt(sD^2 + s_model^2))
    shadow_scale_factor: float = 1.5          # bounded correction: scales within [1/f, f] x original
    shadow_frac: float = 0.8
    joint_sensitivity: bool = True
    sens_axis_deg: float = 1.0
    sens_point_m: float = 0.005
    sens_q_rel: float = 0.02
    max_cycles: int = 3
    max_hours: float = 0.0                    # resource limit (0 = none); a stop is reported as UNTESTED / incomplete

    def to_dict(self):
        return asdict(self)


RESOURCE_FIELDS_4C = ("max_hours",)


# =========================================================================== rendering helpers
def activated(p, K):
    n = p["xyz"].shape[0]
    return dict(xyz=p["xyz"], rot=torch.nn.functional.normalize(p["rot"], dim=-1), scale=torch.exp(p["log_scale"]),
                opacity=torch.sigmoid(p["opacity_logit"]), dc=p["dc"], rest=torch.zeros(n, K, 3, device=p["xyz"].device))


def cat_t(a, b):
    return {k: torch.cat([a[k], b[k]], 0) for k in a}


def render_t(sc, vr, t, lab, transport=None):
    return render_articulated(vr.cam, (transport or sc.transport).at(vr.frame), None, lab, sc.bg, sc.sh_mode,
                              sc.sh_degree, tensors=t)


def aux_t(vr, ft, t, lab, colors):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    cam = vr.cam
    mu, q = posed(t["xyz"], t["rot"], lab, ft)
    dev = mu.device
    R = GaussianRasterizer(GaussianRasterizationSettings(
        image_height=cam.H, image_width=cam.W, tanfovx=cam.tanfovx, tanfovy=cam.tanfovy, bg=torch.zeros(3, device=dev),
        scale_modifier=1.0, viewmatrix=cam.V.to(dev), projmatrix=cam.P.to(dev), sh_degree=0, campos=cam.campos.to(dev),
        prefiltered=False, debug=False, antialiasing=False))
    c, *_ = R.forward_rgbd(mu, torch.zeros(mu.shape[0], 3, device=dev), t["opacity"], colors_precomp=colors(mu).float(),
                           scales=t["scale"], rotations=q)
    return c


class Auth:
    """The authoritative scene G^k: the 4A originals (never modified) + committed births."""

    def __init__(self, scene, labels0: torch.Tensor, status0: np.ndarray):
        from .material_completion import Phase4Scene
        self.base_scene = scene
        self.g0 = scene.g
        self.n0 = len(scene.g)
        self.labels0 = labels0.clone()
        self.status0 = status0
        self.sc = Phase4Scene(scene.g, scene.joint, scene.views, scene.loss_cfg, sh_mode=scene.sh_mode,
                              sh_degree=scene.sh_degree, device=scene.device)
        self.sc.bg = scene.bg
        self.labels = labels0.clone()
        self.births = []            # dicts: patch_uid, rows (a, b), uids, params (raw), cells meta
        self.next_uid = int(self.g0.lineage["gaussian_uid"].max()) + 1

    @property
    def K(self):
        return self.sc.tensors()["rest"].shape[1]

    def tensors(self):
        return self.sc.tensors()

    def commit(self, cand_uid, p_raw, extra):
        from .material_completion import Phase4Scene
        n = p_raw["xyz"].shape[0]
        uids = np.arange(self.next_uid, self.next_uid + n, dtype=np.int64)
        self.next_uid += n
        add = GaussianTensors(p_raw["xyz"].detach(), p_raw["rot"].detach(), p_raw["log_scale"].detach(),
                              p_raw["opacity_logit"].detach(), p_raw["dc"].detach(),
                              torch.zeros(n, self.g0._features_rest.shape[1], 3, device=p_raw["xyz"].device),
                              self.g0.max_sh_degree,
                              {"material": np.full(n, MOVING, np.int8), "gaussian_uid": uids,
                               "parent_uid": np.full(n, -1, np.int64), "original_id": np.full(n, -1, np.int64),
                               "provenance": np.full(n, PROV_PHASE4_BIRTH, np.int8)})
        a = len(self.sc.g)
        g = GaussianTensors.concat(self.sc.g, add)
        g.active_sh_degree = self.g0.active_sh_degree
        sc = Phase4Scene(g, self.sc.joint, self.sc.views, self.sc.loss_cfg, sh_mode=self.sc.sh_mode,
                         sh_degree=self.sc.sh_degree, device=self.sc.device)
        sc.bg = self.sc.bg
        self.sc = sc
        self.labels = torch.cat([self.labels, torch.ones(n, dtype=torch.bool, device=self.labels.device)])
        self.births.append({"patch_uid": cand_uid, "rows": (a, a + n), "uids": uids, "params": {k: v.detach().clone()
                            for k, v in p_raw.items()}, **extra})

    def withdraw(self, patch_uids):
        """Remove whole committed patches (leave-one-out failures): rebuild G from the originals + remaining births."""
        from .material_completion import Phase4Scene
        keep = [b for b in self.births if b["patch_uid"] not in set(patch_uids)]
        g = self.g0
        labels = self.labels0.clone()
        rows = []
        for b in keep:
            a = len(g)
            n = b["params"]["xyz"].shape[0]
            add = GaussianTensors(b["params"]["xyz"], b["params"]["rot"], b["params"]["log_scale"],
                                  b["params"]["opacity_logit"], b["params"]["dc"],
                                  torch.zeros(n, self.g0._features_rest.shape[1], 3, device=b["params"]["xyz"].device),
                                  self.g0.max_sh_degree,
                                  {"material": np.full(n, MOVING, np.int8), "gaussian_uid": b["uids"],
                                   "parent_uid": np.full(n, -1, np.int64), "original_id": np.full(n, -1, np.int64),
                                   "provenance": np.full(n, PROV_PHASE4_BIRTH, np.int8)})
            g = GaussianTensors.concat(g, add)
            g.active_sh_degree = self.g0.active_sh_degree
            labels = torch.cat([labels, torch.ones(n, dtype=torch.bool, device=labels.device)])
            rows.append((a, a + n))
        for b, r in zip(keep, rows):
            b["rows"] = r
        sc = Phase4Scene(g, self.sc.joint, self.sc.views, self.sc.loss_cfg, sh_mode=self.sc.sh_mode,
                         sh_degree=self.sc.sh_degree, device=self.sc.device)
        sc.bg = self.sc.bg
        self.sc, self.labels, self.births = sc, labels, keep

    def without(self, rows: np.ndarray):
        keep = torch.ones(len(self.labels), dtype=torch.bool, device=self.labels.device)
        keep[torch.as_tensor(rows, device=keep.device)] = False
        return {k: v[keep] for k, v in self.tensors().items()}, self.labels[keep]

    def trusted(self, moving_confirmed_codes):
        tr = torch.zeros(len(self.labels), dtype=torch.bool, device=self.labels.device)
        tr[:self.n0] = torch.as_tensor(np.isin(self.status0, moving_confirmed_codes), device=tr.device)
        tr[self.n0:] = True
        return tr

    def undecided(self, codes):
        u = torch.zeros(len(self.labels), dtype=torch.bool, device=self.labels.device)
        u[:self.n0] = torch.as_tensor(np.isin(self.status0, codes), device=u.device)
        return u


# =========================================================================== per-candidate building blocks
def cells_params(cells, cfg: Validate4C, dev):
    return {k: v.to(dev) for k, v in cells_to_gaussian_params(cells, cfg.voxel, tangent_factor=cfg.tangent_factor).items()}


def version_rects(auth: Auth, versions, transport=None):
    """[len(versions), V, 4] padded tile rects of each candidate version alone (one render per view, all versions)."""
    V = len(auth.sc.views)
    if not versions:
        return np.zeros((0, V, 4), np.int16)
    K = auth.K
    ts = [activated(v["p"], K) for v in versions]
    t = {k: torch.cat([x[k] for x in ts], 0) for k in ts[0]}
    lab = torch.cat([torch.full((x["xyz"].shape[0],), bool(v["label"]), device=t["xyz"].device) for x, v in zip(ts, versions)])
    gid = torch.cat([torch.full((x["xyz"].shape[0],), k, device=t["xyz"].device) for k, x in enumerate(ts)])
    out = np.zeros((len(versions), V, 4), np.int16)
    ar = torch.arange(len(lab), device=lab.device)
    with torch.no_grad():
        for vi, vr in enumerate(auth.sc.views):
            o = render_t(auth.sc, vr, t, lab, transport)
            r = FP.tile_rects(vr.cam, o["means"], o["radii"])
            out[:, vi] = FP.aggregate(r[None], ar, gid, len(versions))[0].cpu().numpy()
    return out


def fit_version(auth: Auth, pv, roles, cells, label, view_ids, cfg: Validate4C, loss_cfg):
    dev = auth.labels.device
    p = {k: v.clone().requires_grad_(k != "xyz" or cfg.lr_xyz > 0) for k, v in cells_params(cells, cfg, dev).items()}
    groups = [{"params": [p["rot"]], "lr": cfg.lr_rot}, {"params": [p["log_scale"]], "lr": cfg.lr_scale},
              {"params": [p["opacity_logit"]], "lr": cfg.lr_opacity}, {"params": [p["dc"]], "lr": cfg.lr_dc}]
    if cfg.lr_xyz > 0:
        groups.append({"params": [p["xyz"]], "lr": cfg.lr_xyz})
    opt = torch.optim.Adam(groups)
    base = auth.tensors()
    tg = {}
    losses = []
    for step in range(cfg.refine_steps if view_ids else 0):
        i = view_ids[step % len(view_ids)]
        vr = auth.sc.views[i]
        if i not in tg:
            tg[i] = (ViewTargets(pv.gpu_view(i, dev), loss_cfg), (roles.get(i, dev) == FIT).double())
        tgt, w = tg[i]
        pa = activated(p, auth.K)
        lab = torch.cat([auth.labels, torch.full((pa["xyz"].shape[0],), bool(label), device=dev)])
        m = tgt.full_maps(render_t(auth.sc, vr, cat_t(base, pa), lab))
        loss = sum((m[c] * w).sum() for c in CHANNELS)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return {k: v.detach() for k, v in p.items()}, losses


def _mixed_reliable(tgt, noise, cam_name, k, gap, L):
    from .phase4b_surface import mixed_depth_map
    sig = noise.sigma(cam_name, tgt.D)
    return tgt.V & ~mixed_depth_map(tgt.D, tgt.V, sig, k, gap, L), noise.tol(cam_name, tgt.D)


def evaluate(auth: Auth, pv, roles, items, cfg: Validate4C, loss_cfg, noise, cfg4b, transport=None, view_filter=None,
             removed_rows=None, log=print, tag="evaluate"):
    """View-major exact evaluation. items: dicts with p (raw params), label, rects [V,4].
    removed_rows: evaluate 'scene WITH the item' vs 'scene WITHOUT these rows' (the item IS those rows; leave-one-out).
    Returns per item: list of per-view rows (all views, OUT_OF_FRUSTUM included) and contradiction pixel coords."""
    dev = auth.labels.device
    K = auth.K
    V = len(auth.sc.views)
    res = [{"rows": [], "contra_px": {}} for _ in items]
    ts = [activated(it["p"], K) for it in items]
    t0 = last = time.time()
    for vi, vr in enumerate(auth.sc.views):
        act = [k for k, it in enumerate(items) if FP.rect_nonempty(it["rects"][vi]) and
               (view_filter is None or vi in view_filter[k])]
        for k, it in enumerate(items):
            if k not in act:
                res[k]["rows"].append({"view": vi, "camera": vr.cam.name, "frame": int(vr.frame), "class": "OUT_OF_FRUSTUM"})
        if not act:
            continue
        with torch.no_grad():
            tgt = ViewTargets(pv.gpu_view(vi, dev), loss_cfg)
            role = roles.get(vi, dev)
            rel, tol = _mixed_reliable(tgt, noise, vr.cam.name, cfg4b.k_sigma, cfg4b.mixed_gap, cfg4b.mixed_len)
            ft = (transport or auth.sc.transport).at(vr.frame)
            if removed_rows is None:
                bt, bl = auth.tensors(), auth.labels
                base = render_t(auth.sc, vr, bt, bl, transport)
                bm = tgt.full_maps(base)
            for k in act:
                it = items[k]
                if removed_rows is not None:
                    bt, bl = auth.without(removed_rows[k])
                    base = render_t(auth.sc, vr, bt, bl, transport)
                    bm = tgt.full_maps(base)
                n = ts[k]["xyz"].shape[0]
                if removed_rows is None:
                    t_alt = cat_t(bt, ts[k])
                    l_alt = torch.cat([bl, torch.full((n,), bool(it["label"]), device=dev)])
                    is_p = torch.cat([torch.zeros(len(bl), device=dev), torch.ones(n, device=dev)])
                else:
                    t_alt, l_alt = auth.tensors(), auth.labels
                    is_p = torch.zeros(len(l_alt), device=dev)
                    is_p[torch.as_tensor(removed_rows[k], device=dev)] = 1.0
                alt = render_t(auth.sc, vr, t_alt, l_alt, transport)
                boxes = FP.boxes_for(it["rects"][vi], FP.EMPTY, tgt.W, tgt.H, SSIM_R)
                d_all, ch, _ = box_delta(alt, base, bm, tgt, boxes)
                d_val = box_delta_weighted(alt, base, bm, tgt, boxes, role == VALIDATION)
                z_cam = lambda mu: torch.stack([is_p, is_p * (mu @ vr.cam.V.to(dev)[:3, :3] + vr.cam.V.to(dev)[3, :3])[:, 2],
                                                torch.zeros_like(is_p)], 1)
                c = aux_t(vr, ft, t_alt, l_alt, z_cam)
                cnt = dict.fromkeys(VIS[:5], 0)
                chg = valp = visp = 0
                cm_all = []
                for (y0, y1, x0, x1) in boxes:
                    sl = (slice(y0, y1), slice(x0, x1))
                    AP = c[0][sl]
                    vis = AP >= cfg.alpha_vis
                    M, D, R_, T = tgt.M[sl], tgt.D[sl], rel[sl], tol[sl]
                    Da, Db = alt["depth"][0][sl], base["depth"][0][sl]
                    Aa, Ab = alt["alpha"][0][sl], base["alpha"][0][sl]
                    # the renderer's full composite with the candidate vs the measurement (the candidate attributed by
                    # its own visible responsibility A_P); a contradiction must be NEW (not already in the baseline)
                    bgc = vis & ~M & (Aa >= 0.5) & ~(Ab >= 0.5)
                    front_a = Da < D - T
                    dep = vis & R_ & front_a & ~((Ab >= 0.5) & (Db < D - T))
                    occ = vis & R_ & (Da > D + T)
                    mat = vis & R_ & ~front_a & ~occ
                    inv = vis & M & ~R_
                    for nm, m in zip(VIS[:5], (mat, dep, bgc, occ, inv)):
                        cnt[nm] += int(m.sum())
                    visp += int(vis.sum())
                    cm = dep | bgc
                    if bool(cm.any()):
                        yy, xx = torch.nonzero(cm, as_tuple=True)
                        cm_all.append(torch.stack([yy + y0, xx + x0], 1).cpu().numpy())
                    chm = (alt["color"][:, y0:y1, x0:x1] != base["color"][:, y0:y1, x0:x1]).any(0) | \
                        (alt["alpha"][0][sl] != base["alpha"][0][sl]) | (alt["zdepth"][0][sl] != base["zdepth"][0][sl])
                    chg += int(chm.sum())
                    valp += int((chm & (role[sl] == VALIDATION)).sum())
                ncon = cnt["VISIBLE_DEPTH_CONTRADICTION"] + cnt["VISIBLE_BACKGROUND_CONTRADICTION"]
                bad = ncon > max(cfg.contradiction_px, cfg.contradiction_frac * max(visp, 1))
                cls = "NOT_VISIBLE_IN_MODEL" if visp == 0 else ("CONTRADICTION" if bad else max(
                    ("VISIBLE_MATCH", "OBSERVED_NEARER_OCCLUDER", "INVALID_OR_MIXED_DEPTH"), key=lambda q: cnt[q]))
                HW = tgt.HW
                res[k]["rows"].append({"view": vi, "camera": vr.cam.name, "frame": int(vr.frame), "class": cls,
                                       "d_all_px": float(d_all.sum()) * HW, "d_val_px": float(d_val.sum()) * HW,
                                       "d_all": d_all.cpu().numpy().tolist(), "changed_px": chg, "val_px": valp,
                                       "visible_px": visp, **cnt, "contradiction": bool(bad)})
                if bad and cm_all:
                    res[k]["contra_px"][vi] = np.concatenate(cm_all)
        if time.time() - last > 60:
            last = time.time()
            log(f"    4C {tag}: view {vi + 1}/{V} ({(vi + 1) / (last - t0):.2f} views/s), {len(items)} items")
    return res


def summarise(rows, n_gauss, cfg: Validate4C):
    sc = [r for r in rows if "d_all_px" in r]
    val_px = sum(r["val_px"] for r in sc)
    gain = -sum(r["d_val_px"] for r in sc) - cfg.complexity_px_per_gaussian * n_gauss
    all_gain = -sum(r["d_all_px"] for r in sc)
    con = [r for r in sc if r["contradiction"]]
    worst = max(con, key=lambda r: r["VISIBLE_DEPTH_CONTRADICTION"] + r["VISIBLE_BACKGROUND_CONTRADICTION"], default=None)
    classes = {}
    for r in rows:
        classes[r["class"]] = classes.get(r["class"], 0) + 1
    return {"validation_pixels": val_px, "validation_gain_px": gain, "all_view_gain_px": all_gain,
            "views_scored": len(sc), "views_contradicting": len(con), "view_classes": classes,
            "worst_contradiction": None if worst is None else {"camera": worst["camera"], "frame": worst["frame"],
                                                               "depth_px": worst["VISIBLE_DEPTH_CONTRADICTION"],
                                                               "background_px": worst["VISIBLE_BACKGROUND_CONTRADICTION"]},
            "passes": bool(not con and gain > cfg.margin_px and all_gain > 0 and val_px >= cfg.min_val_pixels)}


def offending_cells(auth: Auth, it, contra_px: dict, cfg: Validate4C):
    from scipy.spatial import cKDTree
    pa = activated(it["p"], auth.K)
    n = pa["xyz"].shape[0]
    bad = np.zeros(n, bool)
    with torch.no_grad():
        for vi, pts in contra_px.items():
            vr = auth.sc.views[vi]
            ft = auth.sc.transport.at(vr.frame)
            mu, _ = posed(pa["xyz"], pa["rot"], torch.full((n,), bool(it["label"]), device=pa["xyz"].device), ft)
            px, py = FP.project_means(vr.cam, mu)
            P = torch.stack([py, px], 1).cpu().numpy()
            d, _ = cKDTree(pts).query(P)
            bad |= d <= 2.0
    return bad


def duplicate_fracs(auth: Auth, X: np.ndarray, undecided_mask: torch.Tensor, cfg: Validate4C):
    from .phase4b_surface import on_support
    t = auth.tensors()
    Py = np.repeat((cfg.dup_tol ** 2 * np.eye(3))[None], len(X), 0)
    und = undecided_mask
    fa = float(on_support(X, Py, t["xyz"][und], t["rot"][und], t["scale"][und], 8, cfg.support_k).mean()) \
        if bool(und.any()) and len(X) else 0.0
    dec = ~und
    fs = float(on_support(X, Py, t["xyz"][dec], t["rot"][dec], t["scale"][dec], 8, cfg.support_k).mean()) \
        if bool(dec.any()) and len(X) else 0.0
    return fa, fs


def shadow_gain(auth: Auth, pv, roles, cand_X: np.ndarray, view_ids, cfg: Validate4C, loss_cfg, shift_bound=None):
    """Bounded existing-geometry correction: fit the nearby existing Gaussians (frozen copy), exact validation gain over
    every view where their footprint before or after is non-empty. Discarded afterwards."""
    from scipy.spatial import cKDTree
    t0 = auth.tensors()
    dev = auth.labels.device
    xyz = t0["xyz"].cpu().numpy()
    reach = cfg.shadow_reach_voxels * cfg.voxel
    d, _ = cKDTree(cand_X).query(xyz) if len(cand_X) else (np.full(len(xyz), np.inf), None)
    near = np.nonzero(d <= reach)[0]
    if len(near) > cfg.shadow_max:
        near = near[np.argsort(d[near])[:cfg.shadow_max]]
    if not len(near) or not view_ids:
        return {"n_shadow": int(len(near)), "validation_gain_px": 0.0, "views": 0}
    g = auth.sc.g
    ix = torch.as_tensor(near, device=dev)
    raw = {"xyz": g._xyz[ix].detach().clone(), "rot": g._rotation[ix].detach().clone(),
           "log_scale": g._scaling[ix].detach().clone(), "opacity_logit": g._opacity[ix].detach().clone(),
           "dc": g._features_dc[ix].detach().clone()}
    # spec 5.3: an existing-geometry correction changes LOCATION, SCALE and OPACITY of existing Gaussians -- not their
    # appearance (a recoloured, double-sided front face could otherwise imitate a missing back face)
    p = {k: v.clone().requires_grad_(k in ("xyz", "log_scale", "opacity_logit")) for k, v in raw.items()}
    opt = torch.optim.Adam([{"params": [p["xyz"]], "lr": cfg.shadow_lr_xyz}, {"params": [p["log_scale"]], "lr": cfg.lr_scale},
                            {"params": [p["opacity_logit"]], "lr": cfg.lr_opacity}])

    def replaced(pp):
        pa = activated(pp, auth.K)
        t = {k: v.clone() for k, v in t0.items()}
        for k in ("xyz", "rot", "scale", "opacity", "dc"):
            t[k] = t[k].index_put((ix,), pa[k])
        return t
    tg = {}
    for step in range(cfg.shadow_steps):
        i = view_ids[step % len(view_ids)]
        if i not in tg:
            tg[i] = (ViewTargets(pv.gpu_view(i, dev), loss_cfg), (roles.get(i, dev) == FIT).double())
        tgt, w = tg[i]
        m = tgt.full_maps(render_t(auth.sc, auth.sc.views[i], replaced(p), auth.labels))
        loss = sum((m[c] * w).sum() for c in CHANNELS)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():                    # projection onto the declared physical bounds
            sh = cfg.shadow_max_shift_voxels * cfg.voxel if shift_bound is None else min(shift_bound,
                                                                                       cfg.shadow_max_shift_voxels * cfg.voxel)
            p["xyz"].copy_(torch.max(torch.min(p["xyz"], raw["xyz"] + sh), raw["xyz"] - sh))
            ls = float(np.log(cfg.shadow_scale_factor))
            p["log_scale"].copy_(torch.max(torch.min(p["log_scale"], raw["log_scale"] + ls), raw["log_scale"] - ls))
    gain, nv = 0.0, 0
    with torch.no_grad():
        ta = replaced({k: v.detach() for k, v in p.items()})
        for vi, vr in enumerate(auth.sc.views):
            base = render_t(auth.sc, vr, t0, auth.labels)
            rb = FP.tile_rects(vr.cam, base["means"][ix], base["radii"][ix])
            if not FP.rect_nonempty(rb.cpu().numpy()).any():
                alt = render_t(auth.sc, vr, ta, auth.labels)
                ra = FP.tile_rects(vr.cam, alt["means"][ix], alt["radii"][ix])
                if not FP.rect_nonempty(ra.cpu().numpy()).any():
                    continue
            else:
                alt = render_t(auth.sc, vr, ta, auth.labels)
                ra = FP.tile_rects(vr.cam, alt["means"][ix], alt["radii"][ix])
            gi = torch.zeros(len(near), dtype=torch.long, device=dev)
            ar = torch.arange(len(near), device=dev)
            ub = FP.aggregate(rb[None], ar, gi, 1)[0, 0].cpu().numpy()
            ua = FP.aggregate(ra[None], ar, gi, 1)[0, 0].cpu().numpy()
            tgt = ViewTargets(pv.gpu_view(vi, dev), loss_cfg)
            bm = tgt.full_maps(base)
            boxes = FP.boxes_for(ub, ua, tgt.W, tgt.H, SSIM_R)
            dv = box_delta_weighted(alt, base, bm, tgt, boxes, roles.get(vi, dev) == VALIDATION)
            gain -= float(dv.sum()) * tgt.HW
            nv += 1
    return {"n_shadow": int(len(near)), "validation_gain_px": gain, "views": nv}


def joint_sensitivity(auth: Auth, pv, roles, it, cfg: Validate4C, loss_cfg, noise, cfg4b, view_ids):
    from .articulated_scene import JointTransport
    from .phase4a_exact import Policy4A, perturbed_joints
    pol = Policy4A(sens_axis_deg=cfg.sens_axis_deg, sens_point_m=cfg.sens_point_m, sens_q_rel=cfg.sens_q_rel)
    out = {}
    vf = [set(view_ids)]
    for name, j in perturbed_joints(auth.sc.joint, pol):
        tr = JointTransport(j, auth.sc.device)
        r = evaluate(auth, pv, roles, [it], cfg, loss_cfg, noise, cfg4b, transport=tr, view_filter=vf, tag=f"sens {name}")
        s = summarise(r[0]["rows"], it["p"]["xyz"].shape[0], cfg)
        out[name] = {"validation_gain_px": s["validation_gain_px"], "views_contradicting": s["views_contradicting"]}
    return out


# =========================================================================== one growth cycle
def _version_item(p, label, rects):
    return {"p": p, "label": label, "rects": rects}


def validate_cycle(auth: Auth, pv, roles, cands, cfg: Validate4C, loss_cfg, noise, cfg4b, undecided_codes, t_start,
                   log=print):
    """Decide every candidate of this cycle against G^k; commit accepted non-interacting patches. Returns decisions."""
    dev = auth.labels.device
    decisions = []
    live = []
    und = auth.undecided(undecided_codes)
    for c in cands:
        d = {"patch_uid": c["patch_uid"], "cycle": c["cycle"], "layer_uid": c["layer_uid"], "search_mode": c["search_mode"],
             "support_groups": c["support_groups"], "views": c["views"], "frames": c["frames"],
             "first_observed_frame": c["first_observed_frame"], "n_sources": int(len(c["sources"])),
             "n_moving_cells": len(c["M_cells"]), "n_static_cells": len(c["S_cells"]),
             "split_cell_frac": c["split_cell_frac"], "source_motion_p90_m": c["source_motion_p90_m"],
             "frontier_parent_uids": c["frontier_parent_uids"], "evidence": c["evidence"]}
        decisions.append(d)
        if cfg.max_hours and (time.time() - t_start) / 3600 > cfg.max_hours:
            d["status"], d["reason"] = UNTESTED, f"max_hours={cfg.max_hours} reached before this candidate"
            continue
        X = np.array([m["mu"] for m in c["M_cells"]])
        fa, fs = duplicate_fracs(auth, X, und, cfg)
        d.update(undecided_original_overlap=fa, existing_overlap=fs)
        if fa >= cfg.dup_frac:
            d["status"], d["reason"] = CONFLICT, "cells inside the render support of an UNDECIDED original: return to 4A"
            continue
        if fs >= cfg.dup_frac:
            d["status"], d["reason"] = DUP, "cells inside the render support of existing decided geometry"
            continue
        if c["split_cell_frac"] >= cfg.mixed_cell_frac:
            d["status"], d["reason"] = MIXED, "most cells hold two depth layers (layer-mixed measurement)"
            continue
        live.append((c, d))
    if not live:
        return decisions, []
    # fit both versions; footprints of every version in one render per view
    versions = []
    for c, d in live:
        for lab, key in ((True, "M_cells"), (False, "S_cells")):
            if c[key]:
                versions.append({"cand": c, "dec": d, "label": lab, "cells": list(c[key]),
                                 "p": cells_params(c[key], cfg, dev)})
    rects0 = version_rects(auth, versions)
    for v, r in zip(versions, rects0):
        vis = [vi for vi in range(r.shape[0]) if FP.rect_nonempty(r[vi])]
        v["p"], v["fit_losses"] = fit_version(auth, pv, roles, v["cells"], v["label"], vis, cfg, loss_cfg)
    rects = version_rects(auth, versions)
    items = [_version_item(v["p"], v["label"], rects[k]) for k, v in enumerate(versions)]
    ev = evaluate(auth, pv, roles, items, cfg, loss_cfg, noise, cfg4b, log=log, tag="all versions")
    pruned = [0] * len(versions)
    for rnd in range(cfg.prune_rounds):
        redo = [k for k, e in enumerate(ev) if any(r.get("contradiction") for r in e["rows"])]
        changed = []
        for k in redo:
            bad = offending_cells(auth, items[k], ev[k]["contra_px"], cfg)
            if bad.any() and (~bad).sum() >= 2:
                versions[k]["cells"] = [cc for cc, b in zip(versions[k]["cells"], bad) if not b]
                pruned[k] += int(bad.sum())
                changed.append(k)
        if not changed:
            break
        sub = [versions[k] for k in changed]
        for v in sub:
            v["p"] = cells_params(v["cells"], cfg, dev)
        r0 = version_rects(auth, sub)
        for v, r in zip(sub, r0):
            vis = [vi for vi in range(r.shape[0]) if FP.rect_nonempty(r[vi])]
            v["p"], v["fit_losses"] = fit_version(auth, pv, roles, v["cells"], v["label"], vis, cfg, loss_cfg)
        r1 = version_rects(auth, sub)
        sub_items = [_version_item(v["p"], v["label"], r1[k]) for k, v in enumerate(sub)]
        sub_ev = evaluate(auth, pv, roles, sub_items, cfg, loss_cfg, noise, cfg4b, log=log, tag=f"prune round {rnd + 1}")
        for k, it, e in zip(changed, sub_items, sub_ev):
            items[k], ev[k] = it, e
    # decisions
    by_cand = {}
    for k, v in enumerate(versions):
        s = summarise(ev[k]["rows"], v["p"]["xyz"].shape[0], cfg)
        s["pruned_contradicting_cells"] = pruned[k]
        s["fit_loss_first_last"] = [v["fit_losses"][0], v["fit_losses"][-1]] if v["fit_losses"] else None
        s["n_gaussians"] = int(v["p"]["xyz"].shape[0])
        v["summary"] = s
        v["dec"]["_rows_" + ("M" if v["label"] else "S")] = ev[k]["rows"]
        by_cand.setdefault(v["dec"]["patch_uid"], {})["M" if v["label"] else "S"] = (k, v)
    pending = []
    for c, d in live:
        vv = by_cand.get(c["patch_uid"], {})
        m = vv.get("M")
        s = vv.get("S")
        sm = m[1]["summary"] if m else None
        ss = s[1]["summary"] if s else None
        d["moving"], d["static"] = sm, ss
        if m is None:
            d["status"], d["reason"] = MIXED, "no moving cells after fusion"
            continue
        if c["support_groups"] < cfg.min_support_groups or sm["validation_pixels"] < cfg.min_val_pixels:
            d["status"] = INSUFF
            d["reason"] = (f"{c['support_groups']} independent source group(s), {sm['validation_pixels']} changed validation "
                           f"pixels (needs {cfg.min_support_groups} / {cfg.min_val_pixels}); reported, not rejected as absent")
            continue
        s_pass = ss is not None and ss["passes"]
        if sm["views_contradicting"]:
            d["status"] = STATIC_ALT if s_pass else CONTRA
            d["reason"] = "the moving version contradicts measured surfaces" + (
                "; the static competitor passes" if s_pass else "")
            continue
        gm = sm["validation_gain_px"]
        if not sm["passes"]:
            d["status"] = STATIC_ALT if s_pass else NOIMP
            d["reason"] = f"moving validation gain {gm:.4g} px-equiv does not clear the margin" + (
                "; the static competitor passes" if s_pass else "")
            continue
        if s_pass:
            gs = ss["validation_gain_px"]
            if gs > gm * (1 + cfg.ambiguity_rel):
                d["status"], d["reason"] = STATIC_ALT, f"static gain {gs:.4g} > moving {gm:.4g} by > ambiguity_rel"
                continue
            if abs(gs - gm) <= cfg.ambiguity_rel * max(gs, gm):
                d["status"], d["reason"] = AMBIG, f"static {gs:.4g} and moving {gm:.4g} within ambiguity_rel"
                continue
        if c["source_motion_p90_m"] < cfg.identifiable_voxels * cfg.voxel:
            d["status"] = AMBIG
            d["reason"] = f"S/M source divergence {c['source_motion_p90_m']:.4f} m < {cfg.identifiable_voxels} voxels"
            continue
        pending.append((c, d, m[0]))
    # shadow + joint sensitivity for provisional passes
    accepted = []
    for c, d, k in pending:
        it = items[k]
        view_ids = [vi for vi in range(it["rects"].shape[0]) if FP.rect_nonempty(it["rects"][vi])]
        gm = d["moving"]["validation_gain_px"]
        if cfg.shadow:
            # the correction may move an existing surface only within the measurement tolerance at the patch depth
            zc = torch.as_tensor([float(np.median(c.get("source_depth", [2.0])))], device=dev)
            bound = float(noise.tol(auth.sc.views[view_ids[0]].cam.name, zc)[0]) if view_ids else None
            sh = shadow_gain(auth, pv, roles, np.array([cc["mu"] for cc in versions[k]["cells"]]), view_ids, cfg, loss_cfg,
                             shift_bound=bound)
            sh["shift_bound_m"] = bound
            d["existing_geometry_correction_shadow"] = sh
            if sh["validation_gain_px"] >= cfg.shadow_frac * gm:
                d["status"] = SHADOW
                d["reason"] = f"refitting {sh['n_shadow']} existing Gaussians gains {sh['validation_gain_px']:.4g} >= " \
                              f"{cfg.shadow_frac} x {gm:.4g}"
                continue
        if cfg.joint_sensitivity:
            js = joint_sensitivity(auth, pv, roles, it, cfg, loss_cfg, noise, cfg4b, view_ids)
            d["joint_sensitivity"] = js
            gs_nom = d["static"]["validation_gain_px"] if (d.get("static") or {}).get("passes") else -np.inf
            fails = [n for n, v in js.items() if v["validation_gain_px"] <= max(cfg.margin_px, gs_nom)]
            if fails:
                d["status"] = JSENS
                d["reason"] = f"under joint perturbations {fails} the moving gain vanishes or no longer beats STATIC"
                continue
        accepted.append((c, d, k))
    # commit: non-interacting greedily by gain; interacting ones are re-tested next round against the updated scene
    accepted.sort(key=lambda x: -x[1]["moving"]["validation_gain_px"])
    committed_rects = []
    deferred = []
    for c, d, k in accepted:
        r = items[k]["rects"]
        inter = any(FP.rects_overlap(r, cr, dil=1).any() for cr in committed_rects)
        if inter:
            deferred.append((c, d))
            continue
        d["status"], d["reason"] = ACC, "all declared criteria met"
        p = items[k]["p"]
        auth.commit(c["patch_uid"], p, {"layer_uid": c["layer_uid"], "cycle": c["cycle"],
                                        "frontier_parent_uids": c["frontier_parent_uids"],
                                        "first_observed_frame": c["first_observed_frame"],
                                        "cells": [{"vox": cc["vox"], "count": cc["count"], "members": cc["members"]}
                                                  for cc in versions[k]["cells"]],
                                        "sources": c["sources"], "source_view": c.get("source_view"),
                                        "source_pix": c.get("source_pix")})
        b = auth.births[-1]
        d.update(added=int(b["rows"][1] - b["rows"][0]), uid_range=[int(b["uids"][0]), int(b["uids"][-1]) + 1])
        committed_rects.append(r)
    for c, d in deferred:
        d["status"], d["reason"] = "DEFERRED_INTERACTION", "footprint interacts with a patch committed this round"
    return decisions, [c for c, d in deferred]


# =========================================================================== driver
def run_4c(auth: Auth, pv, roles, cands0, cfg: Validate4C, loss_cfg, noise, cfg4b, log=print):
    """Growth cycles (6.4), then leave-one-out against the final scene (interaction audit)."""
    from . import phase4b_surface as B
    from .phase4a_exact import CORE, MOVING_S, UNDECIDED
    t_start = time.time()
    decisions = {}
    order = []
    cycles = []
    cands = cands0
    new_front = None
    for cycle in range(cfg.max_cycles):
        if cycle > 0:
            und = auth.undecided(UNDECIDED)
            prop = B.run_proposals(auth.sc, pv, auth.labels, new_front, und, cfg4b, noise, roles, log=log, cycle=cycle)
            cands = prop["candidates"]
            cycles[-1]["next_scan"] = {"counters": prop["counters"], "report": prop["report"]}
        n0b = len(auth.births)
        log(f"4C cycle {cycle}: {len(cands)} candidates against G^{cycle} ({len(auth.labels)} Gaussians, "
            f"{n0b} birth patches)")
        pending = list(cands)
        rounds = 0
        while pending:
            decs, deferred = validate_cycle(auth, pv, roles, pending, cfg, loss_cfg, noise, cfg4b, UNDECIDED, t_start, log=log)
            for d in decs:
                if d["patch_uid"] not in decisions:
                    order.append(d["patch_uid"])
                decisions[d["patch_uid"]] = d
            rounds += 1
            pending = deferred
        added = auth.births[n0b:]
        counts = {}
        for u in order:
            if decisions[u]["cycle"] == cycle:
                counts[decisions[u]["status"]] = counts.get(decisions[u]["status"], 0) + 1
        cycles.append({"cycle": cycle, "candidates": len(cands), "rounds": rounds, "accepted_patches": len(added),
                       "accepted_gaussians": int(sum(b["rows"][1] - b["rows"][0] for b in added)), "statuses": counts})
        log(f"4C cycle {cycle}: {counts}; {len(added)} patches committed")
        if not added:
            break
        new_front = torch.zeros(len(auth.labels), dtype=torch.bool, device=auth.labels.device)
        for b in added:
            new_front[b["rows"][0]:b["rows"][1]] = True
    else:
        if cycles and cycles[-1]["accepted_patches"]:
            cycles[-1]["note"] = f"max_cycles={cfg.max_cycles} reached with new growth in the last cycle"
    # leave-one-out against the FINAL scene
    loo = {}
    if auth.births:
        items = [{"p": b["params"], "label": True} for b in auth.births]
        r = version_rects(auth, items)
        for it, rr in zip(items, r):
            it["rects"] = rr
        rows = [np.arange(*b["rows"]) for b in auth.births]
        ev = evaluate(auth, pv, roles, items, cfg, loss_cfg, noise, cfg4b, removed_rows=rows, log=log, tag="leave-one-out")
        withdraw = []
        for b, e in zip(auth.births, ev):
            s_ = summarise(e["rows"], b["rows"][1] - b["rows"][0], cfg)
            loo[b["patch_uid"]] = {k: v for k, v in s_.items()}
            loo[b["patch_uid"]]["_rows"] = e["rows"]
            if not s_["passes"]:
                withdraw.append(b["patch_uid"])
                d = decisions[b["patch_uid"]]
                d["status"] = NOIMP
                d["reason"] = "withdrawn: no longer clears the margin against the final scene (leave-one-out)"
        if withdraw:
            auth.withdraw(withdraw)
        log(f"  4C leave-one-out: {len(withdraw)} of {len(items)} committed patches withdrawn")
    return [decisions[u] for u in order], cycles, loo, time.time() - t_start


def full_audit(sc, labels, pv, loss_cfg, tol=0.01):
    from .exact_objective import view_objective
    from .phase4_artifacts import view_audit_row
    rows = []
    with torch.no_grad():
        for i, vr in enumerate(sc.views):
            tgt = ViewTargets(pv.gpu_view(i, sc.device), loss_cfg)
            out = render_t(sc, vr, sc.tensors(), labels)
            r = view_audit_row(out, tgt, tol, view_objective(out, tgt).cpu().numpy())
            rows.append({"camera": vr.cam.name, "frame": int(vr.frame), **r})
    return rows
