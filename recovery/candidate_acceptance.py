"""Phase 4C: nonlinear whole-sequence acceptance of proposed patches (spec section 6).

A candidate surface = the static patch and the moving patch fused from the SAME source pixels (paired by the
overlap of their source-pixel sets; a patch without a partner is tested under its own hypothesis only).
For h in {OMIT, STATIC, MOVING}:
  * OMIT leaves the current model unchanged
  * STATIC / MOVING add the patch with a HARD label (never a soft blend), then a short local refinement of
    the patch's (mu, quat, log-scale, opacity logit, DC) only, J and q fixed, original Gaussians frozen,
    rendering the ENTIRE scene at every step, on the fit views
Scores (per-pixel RGB-D-alpha objective, affected pixels):
  * validation gain_h = E_OMIT - E_h on the validation views (positive = the patch helps)
  * all-view contradiction audit: the fraction of all views whose affected pixels get worse; a patch that
    would sit in front of measured surfaces in early frames is rejected here
Data roles are fixed BEFORE any candidate is inspected: validation = Phase-3 processing rows with odd index,
fit = even rows. Phase 3 used every frame, so validation is conditional on the upstream joint, not
independent scientific evidence (recorded in the outputs).

Statuses: ACCEPTED_STATIC, ACCEPTED_MOVING, REJECTED_DUPLICATE, REJECTED_OCCLUSION_CONTRADICTION,
AMBIGUOUS_MATERIAL, INSUFFICIENT_OBSERVATION, JOINT_UNRESOLVED, INVALID_DEPTH_SOURCE.
Only accepted patches become Gaussians (provenance PHASE4_BIRTH, original_id -1, new uid). The full scene is
re-rendered after each acceptance (the next candidate is scored against the updated scene).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .articulated_scene import render_articulated
from .canonical_fusion import cells_to_gaussian_params
from .gaussians import MOVING, PROV_PHASE4_BIRTH, STATIC, GaussianTensors
from .rgbd_loss import pixel_loss_map

ACCEPTED_STATIC, ACCEPTED_MOVING = "ACCEPTED_STATIC", "ACCEPTED_MOVING"
REJECTED_DUPLICATE, REJECTED_OCCLUSION = "REJECTED_DUPLICATE", "REJECTED_OCCLUSION_CONTRADICTION"
AMBIGUOUS, INSUFFICIENT = "AMBIGUOUS_MATERIAL", "INSUFFICIENT_OBSERVATION"
JOINT_UNRESOLVED, INVALID_DEPTH = "JOINT_UNRESOLVED", "INVALID_DEPTH_SOURCE"


@dataclass
class AcceptConfig:
    refine_steps: int = 20
    lr_xyz: float = 1e-3
    lr_rot: float = 1e-3
    lr_scale: float = 5e-3
    lr_opacity: float = 5e-2
    lr_dc: float = 2.5e-2
    margin_abs: float = 1e-3          # minimal validation gain (loss units x pixels)
    margin_rel: float = 0.02          # ... relative to the OMIT loss on the affected validation pixels
    ambiguity_rel: float = 0.25       # S and M both pass and differ by less than this -> AMBIGUOUS_MATERIAL
    contradiction_frac: float = 0.34  # max fraction of affected views that may get worse
    worse_tol_rel: float = 0.02       # a view counts as worse if its affected-pixel loss rises by > this
    dup_tol: float = 0.003            # m, isotropic inflation of the duplicate test (Mahalanobis 3)
    dup_frac: float = 0.5
    pair_overlap: float = 0.5         # source-pixel overlap that pairs an S and an M patch
    max_patch_cells: int = 4000
    min_patch_cells: int = 2

    def to_dict(self):
        return asdict(self)


def data_roles(joint):
    """frame -> 'fit' | 'val' by Phase-3 processing-row parity (fixed before any candidate is seen)."""
    return {int(f): ("val" if k % 2 else "fit") for k, f in enumerate(joint.frames)}


def _activated(p):
    return dict(xyz=p["xyz"], rot=torch.nn.functional.normalize(p["rot"], dim=-1), scale=torch.exp(p["log_scale"]),
                opacity=torch.sigmoid(p["opacity_logit"]), dc=p["dc"],
                rest=torch.zeros(p["xyz"].shape[0], 0, 3, device=p["xyz"].device))


def render_with_patch(scene, vr, moving, patch, label):
    t = scene.tensors()
    pa = _activated(patch)
    K = t["rest"].shape[1]
    pa["rest"] = torch.zeros(pa["xyz"].shape[0], K, 3, device=t["xyz"].device)
    tt = {k: torch.cat([t[k], pa[k]], 0) for k in t}
    mv = torch.cat([moving, torch.full((pa["xyz"].shape[0],), label == MOVING, device=moving.device)])
    return render_articulated(vr.cam, scene.transport.at(vr.frame), None, mv, scene.bg, scene.sh_mode,
                              scene.sh_degree, tensors=tt)


def patch_views(scene, X: torch.Tensor, label):
    from .material_completion import project_px
    out = []
    for i, vr in enumerate(scene.views):
        ft = scene.transport.at(vr.frame)
        Xw = X @ ft.Q.T + ft.p if label == MOVING else X
        u, v, z = project_px(vr.cam, Xw)
        if bool(((z > 0.2) & (u > -0.5) & (u < vr.cam.W) & (v > -0.5) & (v < vr.cam.H)).any()):
            out.append(i)
    return out


def is_duplicate(scene, moving, X: np.ndarray, label, cfg: AcceptConfig):
    """A cell re-explains existing geometry if it lies within Mahalanobis 3 of an existing Gaussian of the same
    material (canonical frame, covariance R S^2 R^T + dup_tol^2 I). Anisotropic: a parallel layer behind a
    flat splat (the other face of a thin panel) is not a duplicate of it."""
    from scipy.spatial import cKDTree
    from .articulated_scene import quat_to_rotmat
    t = scene.tensors()
    same = torch.as_tensor(moving.cpu().numpy() == (label == MOVING), device=t["xyz"].device)
    if not bool(same.any()):
        return False, 0.0
    mu, rot, sc = t["xyz"][same], t["rot"][same], t["scale"][same]
    k = min(8, len(mu))
    _, j = cKDTree(mu.cpu().numpy()).query(X, k=k)
    j = torch.as_tensor(np.asarray(j).reshape(len(X), k), device=mu.device)
    R = quat_to_rotmat(rot[j])
    Sig = R @ torch.diag_embed(sc[j] ** 2) @ R.transpose(-1, -2) + cfg.dup_tol ** 2 * torch.eye(3, device=mu.device)
    dv = torch.as_tensor(X, device=mu.device, dtype=mu.dtype)[:, None, :] - mu[j]
    m2 = (dv[..., None, :] @ torch.linalg.solve(Sig, dv[..., None]))[..., 0, 0].min(1).values
    frac = float((m2 <= 9.0).float().mean())
    return frac >= cfg.dup_frac, frac


def test_hypothesis(scene, base, moving, cells, label, roles, cfg: AcceptConfig, voxel: float):
    """Refine the patch under one hard label on fit views; score validation gain and all-view audit."""
    dev = scene.device
    p0 = cells_to_gaussian_params(cells, voxel)
    patch = {k: v.to(dev).clone().requires_grad_(True) for k, v in p0.items()}
    vids = patch_views(scene, patch["xyz"].detach(), label)
    fit = [i for i in vids if roles[scene.views[i].frame] == "fit"]
    val = [i for i in vids if roles[scene.views[i].frame] == "val"]
    if not val or not fit:
        return {"status": INSUFFICIENT, "reason": "no fit or no validation view sees the patch", "gain": 0.0}
    opt = torch.optim.Adam([{"params": [patch["xyz"]], "lr": cfg.lr_xyz}, {"params": [patch["rot"]], "lr": cfg.lr_rot},
                            {"params": [patch["log_scale"]], "lr": cfg.lr_scale},
                            {"params": [patch["opacity_logit"]], "lr": cfg.lr_opacity},
                            {"params": [patch["dc"]], "lr": cfg.lr_dc}])
    for step in range(cfg.refine_steps):
        i = fit[step % len(fit)]
        vr = scene.views[i]
        out = render_with_patch(scene, vr, moving, patch, label)
        loss = pixel_loss_map(out, scene.view(vr), scene.loss_cfg).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    rows = []
    with torch.no_grad():
        for i in vids:
            vr = scene.views[i]
            pb, _ = base.get(i)
            pa = pixel_loss_map(render_with_patch(scene, vr, moving, patch, label), scene.view(vr), scene.loss_cfg).cpu()
            d = pa - pb
            aff = d.abs() > 1e-7
            rows.append({"view": i, "role": roles[vr.frame], "delta": float(d[aff].sum()),
                         "omit_affected": float(pb[aff].sum()), "affected": int(aff.sum())})
    v = [r for r in rows if r["role"] == "val" and r["affected"]]
    gain = -sum(r["delta"] for r in v)
    omit_aff = sum(r["omit_affected"] for r in v)
    aff_views = [r for r in rows if r["affected"]]
    worse = [r for r in aff_views if r["delta"] > cfg.worse_tol_rel * max(r["omit_affected"], 1e-9)]
    frac_worse = len(worse) / max(1, len(aff_views))
    passes = gain > cfg.margin_abs and gain > cfg.margin_rel * omit_aff
    return {"status": None, "gain": gain, "omit_affected_val": omit_aff, "frac_worse": frac_worse,
            "passes": bool(passes), "contradiction": bool(frac_worse > cfg.contradiction_frac),
            "fit_views": len(fit), "val_views": len(v), "all_views_affected": len(aff_views),
            "patch": {k: t.detach() for k, t in patch.items()}}


def pair_patches(cells_S, lab_S, cells_M, lab_M, cfg: AcceptConfig):
    def groups(cells, lab):
        out = []
        for l in np.unique(lab):
            cs = [cells[i] for i in np.nonzero(lab == l)[0]]
            out.append((cs, np.unique(np.concatenate([c["src"] for c in cs]))))
        return out
    gS, gM = groups(cells_S, lab_S), groups(cells_M, lab_M)
    used_M, cands = set(), []
    for cs, ss in gS:
        best, bo = None, 0.0
        for j, (cm, sm) in enumerate(gM):
            if j in used_M:
                continue
            o = len(np.intersect1d(ss, sm, assume_unique=True)) / max(1, min(len(ss), len(sm)))
            if o > bo:
                best, bo = j, o
        if best is not None and bo >= cfg.pair_overlap:
            used_M.add(best)
            cands.append({STATIC: cs, MOVING: gM[best][0], "overlap": bo})
        else:
            cands.append({STATIC: cs, MOVING: None, "overlap": bo})
    for j, (cm, _) in enumerate(gM):
        if j not in used_M:
            cands.append({STATIC: None, MOVING: cm, "overlap": 0.0})
    cands.sort(key=lambda c: -max(len(c[STATIC] or []), len(c[MOVING] or [])))
    return cands


def run_phase4c(scene, moving, cands, cfg: AcceptConfig, voxel: float, log=print):
    """Sequential acceptance; returns (new GaussianTensors, decisions)."""
    from .material_completion import BaselineCache
    roles = data_roles(scene.joint)
    decisions = []
    base = BaselineCache(scene, moving)
    next_uid = int(scene.g.lineage["gaussian_uid"].max()) + 1
    for ci, cand in enumerate(cands):
        res = {"candidate": ci, "overlap": cand["overlap"]}
        tested = {}
        for label in (STATIC, MOVING):
            cells = cand[label]
            if not cells or len(cells) < cfg.min_patch_cells:
                continue
            cells = cells[: cfg.max_patch_cells]
            X = np.array([c["mu"] for c in cells])
            dup, frac = is_duplicate(scene, moving, X, label, cfg)
            if dup:
                tested[label] = {"status": REJECTED_DUPLICATE, "dup_frac": frac, "passes": False, "gain": 0.0}
                continue
            tested[label] = test_hypothesis(scene, base, moving, cells, label, roles, cfg, voxel)
        name = {STATIC: "static", MOVING: "moving"}
        res.update({name[k]: {kk: vv for kk, vv in v.items() if kk != "patch"} for k, v in tested.items()})
        ok = {k: v for k, v in tested.items() if v.get("passes") and not v.get("contradiction")}
        if not tested:
            res["status"] = INSUFFICIENT
        elif len(ok) == 2:
            gs, gm = ok[STATIC]["gain"], ok[MOVING]["gain"]
            if abs(gs - gm) < cfg.ambiguity_rel * max(gs, gm):
                res["status"] = AMBIGUOUS
            else:
                res["status"] = ACCEPTED_MOVING if gm > gs else ACCEPTED_STATIC
        elif len(ok) == 1:
            res["status"] = ACCEPTED_MOVING if MOVING in ok else ACCEPTED_STATIC
        elif any(v.get("passes") and v.get("contradiction") for v in tested.values()):
            res["status"] = REJECTED_OCCLUSION
        elif all(v["status"] == REJECTED_DUPLICATE for v in tested.values()):
            res["status"] = REJECTED_DUPLICATE
        else:
            res["status"] = INSUFFICIENT
        if res["status"] in (ACCEPTED_MOVING, ACCEPTED_STATIC):
            label = MOVING if res["status"] == ACCEPTED_MOVING else STATIC
            p = tested[label]["patch"]
            n = p["xyz"].shape[0]
            K = scene.g._features_rest.shape[1]
            add = GaussianTensors(p["xyz"], p["rot"], p["log_scale"], p["opacity_logit"], p["dc"],
                                  torch.zeros(n, K, 3, device=p["xyz"].device), scene.g.max_sh_degree,
                                  {"material": np.full(n, label, np.int8),
                                   "gaussian_uid": np.arange(next_uid, next_uid + n, dtype=np.int64),
                                   "parent_uid": np.full(n, -1, np.int64), "original_id": np.full(n, -1, np.int64),
                                   "provenance": np.full(n, PROV_PHASE4_BIRTH, np.int8)})
            next_uid += n
            first = min(c["first_frame"] for c in cand[label])
            res.update(added=n, first_visible_frame_provenance=first)
            scene.g = GaussianTensors.concat(scene.g, add)
            scene.invalidate()
            moving = torch.cat([moving, torch.full((n,), label == MOVING, device=moving.device)])
            base = BaselineCache(scene, moving)            # the full scene is re-rendered for the next candidate
        decisions.append(res)
        log(f"  4C candidate {ci}: {res['status']}" + (f" (+{res.get('added')} Gaussians)" if "added" in res else ""))
    return scene.g, moving, decisions
