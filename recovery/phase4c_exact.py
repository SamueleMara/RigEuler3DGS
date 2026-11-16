"""Phase 4C, corrected: birth acceptance -- precise, held-out, complete (directive 2026-11-11, section 6).

Runs only on a complete 4B of the same run. A candidate is a proposed physical patch; its STATIC and MOVING
versions come from the same source pixels (4B pairing). For H in {OMIT, STATIC, MOVING}:

  * duplicates first (no birth): patch cells inside the kernel 3-sigma render support (inflated by the
    declared dup_tol), in canonical coordinates (label-independent: both materials share the anchor frame), of
    (a) an UNDECIDED original -> ORIGINAL_MATERIAL_AMBIGUOUS (that original may be this surface with the other
    material), (b) any decided original or earlier birth -> REJECTED_DUPLICATE
  * fit: canonical parameters of the patch only (mean, rotation, log-scale, opacity logit, DC), hard label,
    J and q fixed, every original frozen, `refine_steps` Adam steps cycling over the views that see the patch,
    loss = exact objective on FIT-role pixels only (fixed full-view denominators), full-frame rendering
    (ROI gradients are not used for fitting)
  * exact evaluation over EVERY view that the patch can touch (footprint from the kernel radii; views where it
    is culled in both renders are provably neutral): all-pixel delta, VALIDATION-role delta, validation
    support (changed validation pixels) and occlusion contradictions = valid observed-depth pixels that the
    patch NEWLY covers in front of the measurement (D_hat < D - tol with alpha >= alpha_cov, not so before)
Decision (DECLARED provisional policy, uncalibrated):
  PROVISIONAL_INSUFFICIENT_EVIDENCE  fewer than min_val_pixels changed validation pixels
  REJECTED_OCCLUSION_CONTRADICTION   some view has contradictions > max(contradiction_px, frac x footprint)
  REJECTED_NO_IMPROVEMENT            validation gain (pixel-equivalent, minus complexity penalty) < margin, or
                                     the all-view delta is not an improvement
  AMBIGUOUS_MATERIAL                 STATIC and MOVING both pass and differ by < ambiguity_rel
  ACCEPTED_STATIC / ACCEPTED_MOVING
Accepted patches enter the scene immediately (the next candidate is scored against it). After all candidates,
each accepted patch is re-checked leave-one-out against the FINAL scene (REJECTED_REDUNDANT_IN_FINAL if its
validation gain no longer clears the margin) and the final model is rendered on every view (global audit).
A birth is a physical material element: canonical anchor coordinates, permanent material, no birth time
(its first observed frame is provenance only).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import footprint as FP
from .articulated_scene import render_articulated
from .canonical_fusion import cells_to_gaussian_params
from .data_roles import FIT, INDEPENDENCE_LABEL, VALIDATION, RoleConfig, Roles
from .exact_objective import CHANNELS, SSIM_R, ExactLossConfig, ViewTargets, box_delta, box_delta_weighted
from .gaussians import MOVING, PROV_PHASE4_BIRTH, STATIC, GaussianTensors
from .runlog import atomic_json

ACC_S, ACC_M = "ACCEPTED_STATIC", "ACCEPTED_MOVING"
INSUFF, OCCL, NOIMP = "PROVISIONAL_INSUFFICIENT_EVIDENCE", "REJECTED_OCCLUSION_CONTRADICTION", "REJECTED_NO_IMPROVEMENT"
AMBIG, DUP, ORIG_AMB = "AMBIGUOUS_MATERIAL", "REJECTED_DUPLICATE", "ORIGINAL_MATERIAL_AMBIGUOUS"
REDUNDANT, SMALL = "REJECTED_REDUNDANT_IN_FINAL", "REJECTED_TOO_SMALL"
EXPLAINED_SRC = "REJECTED_SOURCES_ALREADY_EXPLAINED"


@dataclass
class AcceptConfig4C:
    refine_steps: int = 20
    lr_xyz: float = 0.0                   # positions stay at the MEASURED cell medians (no drift off the surface)
    tangent_factor: float = 0.45          # tangential sd = factor x voxel (kernel 3-sigma spill past an edge ~1.35 voxel)
    lr_rot: float = 1e-3
    lr_scale: float = 5e-3
    lr_opacity: float = 5e-2
    lr_dc: float = 2.5e-2
    margin_px: float = 1e-3               # minimal validation gain, pixel-equivalent loss units (declared)
    complexity_px_per_gaussian: float = 1e-4   # declared redundancy penalty per added Gaussian (uncalibrated)
    ambiguity_rel: float = 0.25
    min_val_pixels: int = 50              # changed VALIDATION pixels needed for any decision
    contradiction_px: int = 16            # declared, uncalibrated
    contradiction_frac: float = 0.05      # of the patch's changed pixels in that view
    alpha_cov: float = 0.5
    depth_tol: float = 0.01               # m, ASSUMED unless 4B recorded a calibrated sensor
    dup_tol: float = 0.003                # m, declared inflation of the duplicate containment test (uncalibrated)
    dup_frac: float = 0.5
    support_k: float = 3.0
    max_patch_cells: int = 4000
    min_patch_cells: int = 2
    voxel: float = 0.01
    identifiable_voxels: float = 2.0      # one-sided acceptance needs the sources' S-M divergence >= this x voxel
    prune_rounds: int = 2                 # remove cells that project onto contradiction pixels, refit, re-score
    edge_px: int = 2                      # contradictions are not counted in the observed depth-edge band
    depth_jump: float = 0.03              # (same declared edge rule as 4B)

    def to_dict(self):
        return asdict(self)


def _activated(p, K):
    n = p["xyz"].shape[0]
    return dict(xyz=p["xyz"], rot=torch.nn.functional.normalize(p["rot"], dim=-1), scale=torch.exp(p["log_scale"]),
                opacity=torch.sigmoid(p["opacity_logit"]), dc=p["dc"], rest=torch.zeros(n, K, 3, device=p["xyz"].device))


class Scene4C:
    """Original Gaussians (frozen) + accepted births, with hard render labels."""

    def __init__(self, scene, labels: torch.Tensor):
        self.sc = scene
        self.labels = labels.clone()

    def tensors(self):
        return {k: v.detach() for k, v in self.sc.tensors().items()}

    def render(self, vr, patch=None, label=None, only_patch=False):
        t = self.tensors()
        mv = self.labels
        if patch is not None:
            pa = _activated(patch, t["rest"].shape[1])
            pm = torch.full((pa["xyz"].shape[0],), label == MOVING, device=mv.device)
            if only_patch:
                t, mv = pa, pm
            else:
                t = {k: torch.cat([t[k], pa[k]], 0) for k in t}
                mv = torch.cat([mv, pm])
        return render_articulated(vr.cam, self.sc.transport.at(vr.frame), None, mv, self.sc.bg, self.sc.sh_mode,
                                  self.sc.sh_degree, tensors=t)

    def render_without(self, vr, rows: np.ndarray):
        t = self.tensors()
        keep = torch.ones(len(self.labels), dtype=torch.bool, device=self.labels.device)
        keep[torch.as_tensor(rows, device=keep.device)] = False
        t = {k: v[keep] for k, v in t.items()}
        return render_articulated(vr.cam, self.sc.transport.at(vr.frame), None, self.labels[keep], self.sc.bg,
                                  self.sc.sh_mode, self.sc.sh_degree, tensors=t)

    def add(self, patch, label, lineage):
        p = {k: v.detach() for k, v in patch.items()}
        n = p["xyz"].shape[0]
        K = self.sc.g._features_rest.shape[1]
        add = GaussianTensors(p["xyz"], p["rot"], p["log_scale"], p["opacity_logit"], p["dc"],
                              torch.zeros(n, K, 3, device=p["xyz"].device), self.sc.g.max_sh_degree, lineage)
        self.sc.g = GaussianTensors.concat(self.sc.g, add)
        self.sc.invalidate()
        self.labels = torch.cat([self.labels, torch.full((n,), label == MOVING, device=self.labels.device)])

    def remove(self, rows: np.ndarray):
        keep = np.setdiff1d(np.arange(len(self.sc.g)), rows)
        self.sc.g = self.sc.g.subset(keep)
        self.sc.invalidate()
        self.labels = self.labels[torch.as_tensor(keep, device=self.labels.device)]


def duplicate_test(s4: Scene4C, X: np.ndarray, label, undecided: torch.Tensor, cfg: AcceptConfig4C):
    """Fractions of patch cells (canonical coordinates) inside the kernel 3-sigma support (inflated by dup_tol)
    of (a) an UNDECIDED original and (b) any decided original or earlier birth. Canonical coordinates do not
    depend on the material label -- at the anchor frame static and moving Gaussians share one world frame --
    so a candidate that coincides with ANY existing Gaussian there is a physical duplicate at the anchor."""
    from .phase4b_exact import on_support
    t = s4.tensors()
    n0 = len(undecided)
    Py = np.repeat((cfg.dup_tol ** 2 * np.eye(3))[None], len(X), 0)
    und = torch.zeros(len(s4.labels), dtype=torch.bool, device=s4.labels.device)
    und[:n0] = undecided
    fa = fs = 0.0
    if bool(und.any()):
        fa = float(on_support(X, Py, t["xyz"][und], t["rot"][und], t["scale"][und], 8, cfg.support_k).mean())
    dec = ~und
    if bool(dec.any()):
        fs = float(on_support(X, Py, t["xyz"][dec], t["rot"][dec], t["scale"][dec], 8, cfg.support_k).mean())
    return fa, fs


def patch_views(s4: Scene4C, patch, label):
    """Views whose padded kernel footprint of the patch is non-empty, with the tile rect."""
    out = []
    with torch.no_grad():
        for i, vr in enumerate(s4.sc.views):
            o = s4.render(vr, patch, label, only_patch=True)
            r = FP.tile_rects(vr.cam, o["means"], o["radii"]).cpu().numpy()
            ne = FP.rect_nonempty(r)
            if ne.any():
                rr = r[ne].astype(np.int32)
                out.append((i, np.array([rr[:, 0].min(), rr[:, 1].min(), rr[:, 2].max(), rr[:, 3].max()], np.int16)))
    return out


def fit_patch(s4: Scene4C, pv, roles, cells, label, cfg: AcceptConfig4C, loss_cfg: ExactLossConfig, vis):
    dev = s4.labels.device
    p0 = cells_to_gaussian_params(cells, cfg.voxel, tangent_factor=cfg.tangent_factor)
    patch = {k: v.to(dev).clone().requires_grad_(True) for k, v in p0.items()}
    groups = [{"params": [patch["xyz"]], "lr": cfg.lr_xyz}] if cfg.lr_xyz > 0 else []
    if cfg.lr_xyz <= 0:
        patch["xyz"].requires_grad_(False)
    opt = torch.optim.Adam(groups + [{"params": [patch["rot"]], "lr": cfg.lr_rot},
                            {"params": [patch["log_scale"]], "lr": cfg.lr_scale},
                            {"params": [patch["opacity_logit"]], "lr": cfg.lr_opacity},
                            {"params": [patch["dc"]], "lr": cfg.lr_dc}])
    losses = []
    for step in range(cfg.refine_steps if vis else 0):
        i = vis[step % len(vis)][0]
        vr = s4.sc.views[i]
        tgt = ViewTargets(pv.gpu_view(i, dev), loss_cfg)
        w = (roles.get(i, dev) == FIT).double()
        m = tgt.full_maps(s4.render(vr, patch, label))
        loss = sum((m[c] * w).sum() for c in CHANNELS)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return {k: v.detach() for k, v in patch.items()}, losses


def evaluate(s4: Scene4C, pv, roles, patch, label, vis, loss_cfg: ExactLossConfig, cfg: AcceptConfig4C, rows_in_scene=None,
             masks=None):
    """Exact per-view deltas of adding `patch`; with rows_in_scene, of the current scene WITH those rows versus
    without them (leave-one-out: the rows already are the patch)."""
    dev = s4.labels.device
    rows = []
    with torch.no_grad():
        for i, rect in vis:
            vr = s4.sc.views[i]
            view = pv.gpu_view(i, dev)
            tgt = ViewTargets(view, loss_cfg)
            if rows_in_scene is None:
                base, alt = s4.render(vr), s4.render(vr, patch, label)
            else:
                base, alt = s4.render_without(vr, rows_in_scene), s4.render(vr)
            bm = tgt.full_maps(base)
            boxes = FP.boxes_for(rect, np.zeros(4, np.int16), tgt.W, tgt.H, SSIM_R)
            d_all, ch, _ = box_delta(alt, base, bm, tgt, boxes)
            role = roles.get(i, dev)
            d_val = box_delta_weighted(alt, base, bm, tgt, boxes, role == VALIDATION)
            from .phase4b_exact import edge_band
            reliable = tgt.V & ~edge_band(tgt.M, tgt.D, tgt.V, cfg)
            # occlusion contradictions and validation support inside the boxes
            con = val_px = chg = 0
            for (y0, y1, x0, x1) in boxes:
                sl = (slice(y0, y1), slice(x0, x1))
                V, D = reliable[sl], tgt.D[sl]
                Ab, Aa = base["alpha"][0][sl], alt["alpha"][0][sl]
                Db, Da = base["depth"][0][sl], alt["depth"][0][sl]
                front_a = V & (Aa >= cfg.alpha_cov) & (Da < D - cfg.depth_tol)
                front_b = V & (Ab >= cfg.alpha_cov) & (Db < D - cfg.depth_tol)
                cm = front_a & ~front_b
                con += int(cm.sum())
                if masks is not None and bool(cm.any()):
                    yy, xx = torch.nonzero(cm, as_tuple=True)
                    masks.setdefault(i, []).append(torch.stack([yy + y0, xx + x0], 1).cpu().numpy())
                c = (alt["color"][:, y0:y1, x0:x1] != base["color"][:, y0:y1, x0:x1]).any(0) | (Aa != Ab) | \
                    (alt["zdepth"][0][sl] != base["zdepth"][0][sl])
                chg += int(c.sum())
                val_px += int((c & (role[sl] == VALIDATION)).sum())
            HW = tgt.HW
            rows.append({"view": i, "camera": vr.cam.name, "frame": int(vr.frame),
                         "d_all_px": float(d_all.sum()) * HW, "d_val_px": float(d_val.sum()) * HW,
                         "d_all": d_all.cpu().numpy().tolist(), "changed_px": chg, "val_px": val_px,
                         "contradiction_px": con})
    return rows


def offending_cells(s4: Scene4C, patch, label, rows, masks, cfg: AcceptConfig4C):
    """Patch rows whose projected centre (kernel projection) falls within 2 px of a contradiction pixel in a
    view whose contradictions exceed the rule."""
    bad = np.zeros(patch["xyz"].shape[0], bool)
    with torch.no_grad():
        for r in rows:
            if r["contradiction_px"] <= max(cfg.contradiction_px, cfg.contradiction_frac * r["changed_px"]):
                continue
            pts = np.concatenate(masks.get(r["view"], [np.zeros((0, 2), int)]))
            if not len(pts):
                continue
            vr = s4.sc.views[r["view"]]
            o = s4.render(vr, patch, label, only_patch=True)
            px, py = FP.project_means(vr.cam, o["means"])
            P = torch.stack([py, px], 1).cpu().numpy()
            from scipy.spatial import cKDTree
            d, _ = cKDTree(pts).query(P)
            bad |= (d <= 2.0) & (o["radii"].cpu().numpy() > 0)
    return bad


def decide_h(rows, n_gauss, cfg: AcceptConfig4C):
    val_px = sum(r["val_px"] for r in rows)
    gain = -sum(r["d_val_px"] for r in rows) - cfg.complexity_px_per_gaussian * n_gauss
    all_gain = -sum(r["d_all_px"] for r in rows)
    worst = max(rows, key=lambda r: r["contradiction_px"] - max(cfg.contradiction_px, cfg.contradiction_frac * r["changed_px"]),
                default=None)
    contra = worst is not None and worst["contradiction_px"] > max(cfg.contradiction_px, cfg.contradiction_frac * worst["changed_px"])
    st = {"validation_pixels": val_px, "validation_gain_px": gain, "all_view_gain_px": all_gain,
          "views_touched": len(rows), "max_contradiction_px": max((r["contradiction_px"] for r in rows), default=0),
          "contradiction_view": None if not contra else {"camera": worst["camera"], "frame": worst["frame"],
                                                          "px": worst["contradiction_px"]}}
    if val_px < cfg.min_val_pixels:
        return INSUFF, st
    if contra:
        return OCCL, st
    if gain < cfg.margin_px or all_gain <= 0:
        return NOIMP, st
    return "PASS", st


def run_4c(s4: Scene4C, pv, roles, cands, undecided, cfg: AcceptConfig4C, loss_cfg, log=print):
    decisions, accepted = [], []
    used_src = np.zeros(0, np.int64)
    next_uid = int(s4.sc.g.lineage["gaussian_uid"].max()) + 1
    t0 = time.time()
    last = t0
    for ci, cand in enumerate(cands):
        res = {"candidate": ci, "overlap": float(cand["overlap"])}
        tested = {}
        for label in (STATIC, MOVING):
            cells = cand[label]
            name = "static" if label == STATIC else "moving"
            if not cells or len(cells) < cfg.min_patch_cells:
                continue
            cells = cells[: cfg.max_patch_cells]
            src_h = np.unique(np.concatenate([c["src"] for c in cells]))
            f_used = float(np.isin(src_h, used_src).mean()) if len(used_src) else 0.0
            if f_used >= cfg.dup_frac:
                tested[label] = {"status": EXPLAINED_SRC, "sources_already_explained_frac": f_used}
                continue
            X = np.array([c["mu"] for c in cells])
            fa, fs = duplicate_test(s4, X, label, undecided, cfg)
            if fa >= cfg.dup_frac:
                tested[label] = {"status": ORIG_AMB, "undecided_alternate_frac": fa, "decided_existing_frac": fs}
                continue
            if fs >= cfg.dup_frac:
                tested[label] = {"status": DUP, "undecided_alternate_frac": fa, "decided_existing_frac": fs}
                continue
            p0 = {k: v.to(s4.labels.device) for k, v in cells_to_gaussian_params(cells, cfg.voxel, tangent_factor=cfg.tangent_factor).items()}
            vis = patch_views(s4, p0, label)
            pruned = 0
            for rnd in range(cfg.prune_rounds + 1):
                patch, losses = fit_patch(s4, pv, roles, cells, label, cfg, loss_cfg, vis)
                vis = patch_views(s4, patch, label)
                masks = {}
                rows = evaluate(s4, pv, roles, patch, label, vis, loss_cfg, cfg, masks=masks)
                st, stats = decide_h(rows, len(cells), cfg)
                if st != OCCL or rnd == cfg.prune_rounds:
                    break
                bad = offending_cells(s4, patch, label, rows, masks, cfg)
                if not bad.any() or (~bad).sum() < cfg.min_patch_cells:
                    break
                cells = [c for c, b in zip(cells, bad) if not b]
                pruned += int(bad.sum())
                p0 = {k: v.to(s4.labels.device) for k, v in cells_to_gaussian_params(cells, cfg.voxel, tangent_factor=cfg.tangent_factor).items()}
                vis = patch_views(s4, p0, label)
            stats["pruned_contradicting_cells"] = pruned
            tested[label] = {"status": st, **stats, "fit_loss_first_last": [losses[0], losses[-1]] if losses else None,
                             "n_gaussians": len(cells), "undecided_alternate_frac": fa, "decided_existing_frac": fs,
                             "patch": patch, "cells": cells}
            res[name] = {k: v for k, v in tested[label].items() if k not in ("patch", "cells")}
        passed = {k: v for k, v in tested.items() if v["status"] == "PASS"}
        if len(passed) == 2:
            gs, gm = passed[STATIC]["validation_gain_px"], passed[MOVING]["validation_gain_px"]
            res["status"] = AMBIG if abs(gs - gm) < cfg.ambiguity_rel * max(gs, gm) else (ACC_M if gm > gs else ACC_S)
        elif len(passed) == 1:
            (lp, vp), = passed.items()
            other = tested.get(MOVING if lp == STATIC else STATIC)
            if other is not None and other["status"] == OCCL and other.get("validation_gain_px", -1e300) > vp["validation_gain_px"]:
                res["status"] = AMBIG                 # the better-supported material was only stopped by contradictions
                res["reason"] = "the other material has the larger validation gain but contradicts measured surfaces"
            elif other is None and cand.get("source_motion_p90_m", np.inf) < cfg.identifiable_voxels * cfg.voxel:
                res["status"] = AMBIG                 # S and M nearly coincide on these sources: material not identifiable
                res["reason"] = (f"only {'STATIC' if lp == STATIC else 'MOVING'} was testable and the sources' S/M "
                                 f"divergence {cand['source_motion_p90_m']:.4f} m < {cfg.identifiable_voxels} voxels")
            else:
                res["status"] = ACC_M if lp == MOVING else ACC_S
        elif tested:
            order = (ORIG_AMB, OCCL, DUP, EXPLAINED_SRC, INSUFF, NOIMP)
            res["status"] = min((v["status"] for v in tested.values()), key=order.index)
        else:
            res["status"] = SMALL
        for k, v in tested.items():
            res.setdefault("static" if k == STATIC else "moving", {k2: v2 for k2, v2 in v.items() if k2 not in ("patch", "cells")})
        if res["status"] in (ACC_M, ACC_S):
            label = MOVING if res["status"] == ACC_M else STATIC
            tv = tested[label]
            n = tv["patch"]["xyz"].shape[0]
            rows0 = len(s4.sc.g)
            s4.add(tv["patch"], label, {"material": np.full(n, label, np.int8),
                                        "gaussian_uid": np.arange(next_uid, next_uid + n, dtype=np.int64),
                                        "parent_uid": np.full(n, -1, np.int64), "original_id": np.full(n, -1, np.int64),
                                        "provenance": np.full(n, PROV_PHASE4_BIRTH, np.int8)})
            next_uid += n
            src = np.unique(np.concatenate([c["src"] for c in tv["cells"]]))
            used_src = np.union1d(used_src, src)
            res.update(added=n, rows=[rows0, rows0 + n], uid_range=[next_uid - n, next_uid],
                       first_observed_frame_provenance=min(c["first_frame"] for c in tv["cells"]),
                       source_pixels=int(len(src)))
            accepted.append({"candidate": ci, "label": label, "rows": (rows0, rows0 + n), "patch": tv["patch"]})
        decisions.append(res)
        if time.time() - last > 60:
            last = time.time()
            log(f"    4C: candidate {ci + 1}/{len(cands)}, accepted {len(accepted)}")
    log(f"  4C: {len(cands)} candidates in {time.time() - t0:.1f} s; accepted {len(accepted)}")
    return decisions, accepted


def leave_one_out(s4: Scene4C, pv, roles, accepted, decisions, cfg, loss_cfg, log=print):
    """Each accepted birth must still clear the margin against the FINAL scene (all other births present):
    exact final scene with vs without its rows. Failing births are withdrawn together at the end."""
    withdraw = []
    for a in accepted:
        rows = np.arange(*a["rows"])
        vis = patch_views(s4, a["patch"], a["label"])
        ev = evaluate(s4, pv, roles, a["patch"], a["label"], vis, loss_cfg, cfg, rows_in_scene=rows)
        st, stats = decide_h(ev, len(rows), cfg)
        d = decisions[a["candidate"]]
        d["final_leave_one_out"] = {"status": st, **stats}
        if st != "PASS":
            d["status"] = REDUNDANT
            withdraw.append(rows)
    if withdraw:
        s4.remove(np.concatenate(withdraw))
    log(f"  4C leave-one-out against the final scene: {len(withdraw)} of {len(accepted)} births withdrawn")
    return [a["candidate"] for a, w in zip(accepted, [True] * len(accepted)) if decisions[a["candidate"]]["status"] == REDUNDANT]


def scene_audit(s4: Scene4C, pv, loss_cfg):
    from .exact_objective import view_objective
    out = []
    with torch.no_grad():
        for i, vr in enumerate(s4.sc.views):
            tgt = ViewTargets(pv.gpu_view(i, s4.labels.device), loss_cfg)
            out.append(view_objective(s4.render(vr), tgt).cpu().numpy())
    return np.array(out)


def stage_4c(run, cfg: AcceptConfig4C, loss_cfg: ExactLossConfig):
    from .phase4_stages_legacy import joint_files_proof, load_4a
    from .phase4a_exact import SUSPECT, UNRESOLVED, UNTESTED
    log = run.log
    if not (run.marker("4b") or {}).get("complete"):
        raise SystemExit("stage 4c requires a complete 4b in this run directory")
    status, render_label = load_4a(run)
    dev = run.scene.device
    j0 = joint_files_proof(run.up)
    prop = torch.load(run.dir / "discovery" / "proposals.pt", weights_only=False)
    dm = json.loads((run.dir / "discovery" / "discovery_manifest.json").read_text())
    rc = dm["roles"]["config"]
    roles = Roles(run.scene.views, RoleConfig(rc["block_px"], rc["guard_px"], rc["time_block"]))
    if dm["sensor"]["calibrated"]:
        log("4C: NOTE the contradiction depth tolerance is the declared depth_tol, not the calibrated sigma")
    undecided = torch.as_tensor(np.isin(status, [UNRESOLVED, SUSPECT, UNTESTED]), device=dev)
    s4 = Scene4C(run.scene, torch.as_tensor(render_label, device=dev))
    n_orig = len(run.scene.g)
    lin = run.scene.g.lineage
    lin["material"] = render_label.astype(np.int8)
    lin["original_id"] = np.arange(n_orig, dtype=np.int64)
    E_before = scene_audit(s4, run.pv, loss_cfg)
    cands = prop["candidates"]
    log(f"4C: {len(cands)} candidates; validation roles {INDEPENDENCE_LABEL}")
    decisions, accepted = run_4c(s4, run.pv, roles, cands, undecided, cfg, loss_cfg, log=log)
    removed = leave_one_out(s4, run.pv, roles, accepted, decisions, cfg, loss_cfg, log=log)
    E_after = scene_audit(s4, run.pv, loss_cfg)
    if joint_files_proof(run.up) != j0:
        raise RuntimeError("Phase-3 joint files changed during 4C")
    g = s4.sc.g
    g.lineage["material"] = s4.labels.cpu().numpy().astype(np.int8)
    g.check()
    n_acc = len(g) - n_orig
    assert np.array_equal(g.lineage["original_id"][:n_orig], np.arange(n_orig))
    acc_dir = run.dir / "acceptance"
    acc_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for d in decisions:
        counts[d["status"]] = counts.get(d["status"], 0) + 1
    atomic_json(acc_dir / "decisions.json", {"config": cfg.to_dict(), "loss": loss_cfg.to_dict(), "counts": counts,
                                             "validation": INDEPENDENCE_LABEL, "decisions": decisions,
                                             "withdrawn_leave_one_out": removed,
                                             "thresholds_status": "declared, uncalibrated (margin_px, "
                                             "complexity_px_per_gaussian, contradiction_px/frac, dup_tol, dup_frac)"})
    fc = run.dir / "final_canonical"
    fc.mkdir(parents=True, exist_ok=True)
    g.save_ply(fc / "point_cloud.ply")
    g.save_lineage(fc)
    np.savez_compressed(fc / "phase4_status.npz", status_4a=np.asarray(status, dtype="U32"),
                        note="rows >= n_original are 4C births (status in acceptance/decisions.json)")
    torch.save({"xyz": g._xyz.cpu(), "rotation": g._rotation.cpu(), "scaling": g._scaling.cpu(),
                "opacity": g._opacity.cpu(), "features_dc": g._features_dc.cpu(), "features_rest": g._features_rest.cpu(),
                "lineage": g.lineage, "max_sh_degree": g.max_sh_degree, "n_original": n_orig}, fc / "checkpoint.pt")
    (fc / "joint_reference.json").write_text(json.dumps(run.up.joint.to_dict(), indent=1))
    views = [{"camera": vr.cam.name, "frame": int(vr.frame)} for vr in run.scene.views]
    atomic_json(fc / "full_sequence_render_audit.json", {
        "views": views, "channels": list(CHANNELS), "before_4C": E_before.tolist(), "after_4C": E_after.tolist(),
        "E_before": float(E_before.sum()), "E_after": float(E_after.sum()),
        "views_worse": [views[i] | {"delta": float(E_after[i].sum() - E_before[i].sum())}
                        for i in np.nonzero(E_after.sum(1) > E_before.sum(1) + 1e-12)[0]],
        "validation_note": INDEPENDENCE_LABEL})
    atomic_json(fc / "counts.json", {"n_original": n_orig, "n_accepted": n_acc, "n_final": len(g),
                                     "moving": int(g.lineage["material"].sum()),
                                     "static": int((g.lineage["material"] == 0).sum()), "decisions": counts})
    marker = {"stage": "4c", "complete": True, "counts": counts, "n_original": n_orig, "n_accepted": n_acc,
              "E_before": float(E_before.sum()), "E_after": float(E_after.sum()), "joint_files_sha256": j0,
              "resources": log.resources(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "NOT_FINAL": bool(run.manifest.get("NOT_FINAL"))}
    atomic_json(run.dir / "stage4c_complete.json", marker)
    run.manifest.setdefault("stages", {})["4c"] = {"complete": True, "counts": counts, "n_accepted": n_acc}
    run.write_manifest()
    log(f"4C COMPLETE: {counts}; {n_orig} original + {n_acc} births; E {E_before.sum():.6g} -> {E_after.sum():.6g}")
    return marker
