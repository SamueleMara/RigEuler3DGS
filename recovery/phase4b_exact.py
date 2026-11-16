"""Phase 4B, corrected: missing geometry must be genuinely OBSERVED (directive 2026-11-11, section 5).

Runs only on a COMPLETE 4A (stage marker). The accepted 4A scene (render labels) is rendered once per view;
residual classes are computed once per view:

  0 EXPLAINED                 the model accounts for the observed foreground (|D - D_hat| <= tol, covered)
  1 COVERAGE_GAP              M = 1, valid depth, rendered alpha < alpha_cov
  2 FRONT_SURFACE             observed depth nearer than the prediction by > tol  (needs scrutiny)
  3 BEHIND_PREDICTION         observed surface behind the prediction (never a birth source)
  4 DEPTH_INVALID             M = 1 without valid depth (no 3D birth from it)
  5 MIXED_EDGE                within edge_px of the silhouette or an observed depth discontinuity
  6 BACKGROUND                M = 0 (no depth observation; still negative evidence for 4C)
  7 ORIGINAL_MATERIAL_AMBIGUOUS   a gap / front pixel that an UNRESOLVED (or SUSPECT / UNTESTED) original
                              Gaussian explains in its ALTERNATE pose: not a birth source -- a moving
                              duplicate of an undecided original must not be created
Birth sources = COVERAGE_GAP | FRONT_SURFACE pixels in the PROPOSAL role only (data_roles.py).

Depth tolerance per pixel: tol = max(depth_tol_floor, k_sigma sigma_D(D)) with sigma_D from a CALIBRATED
sensor model (utils/noise/sensor_model.py); without one the declared `depth_tol_assumed` is used and the
outputs say ASSUMED (not sensor calibration; no millimetre accuracy is claimed).

Back-projection of a measured axial depth d at pixel (u, v) of camera c, frame t:
    y_C = d K^-1 [u + 0.5, v + 0.5, 1]^T,   y_W = R_WC y_C + t_WC
    x_S = y_W,   x_M = Q_t^T (y_W - p_t)          (hypotheses; a moving point exists at EVERY frame)
Three separate uncertainties, never mixed:
  * measurement   P_y = R J diag(sigma_px^2, sigma_px^2, sigma_D(d)^2) J^T R^T   (propagated, per point)
  * shared joint  not propagated: every proposal is conditional on the Phase-3 joint (recorded)
  * rendering     the 3DGS covariance Sigma_g = R S^2 R^T of an existing Gaussian -- a FOOTPRINT, used only
                  to ask whether a measured point lies inside the kernel's 3-sigma render support:
                  (y - mu)^T (Sigma_g + P_y)^-1 (y - mu) <= support_k^2  (support_k = 3, the kernel cutoff)
                  ("on existing geometry", dropped and counted). This replaces the pre-correction
                  `explained_mahal`, which used the footprint as if it were sensor error.
Layer-aware fusion per hypothesis: cells keyed by (canonical voxel, signed normal bin) -- opposite faces of a
thin panel never share a cell -- and each cell split along its normal wherever the observations leave a gap
> layer_gap (two same-facing parallel layers in one voxel stay separate). Patches = touching cells of the same
normal bin whose plane offsets agree within max(layer_gap, 2 x their combined measured spread). Cells need >= min_obs observations from >= min_views
views and keep their exact source pixels (camera, frame, pixel) as lineage.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .canonical_fusion import AXES, backproject, point_covariance
from .data_roles import PROPOSAL, RoleConfig, Roles
from .runlog import atomic_json

CODES = {"EXPLAINED": 0, "COVERAGE_GAP": 1, "FRONT_SURFACE": 2, "BEHIND_PREDICTION": 3, "DEPTH_INVALID": 4,
         "MIXED_EDGE": 5, "BACKGROUND": 6, "ORIGINAL_MATERIAL_AMBIGUOUS": 7}
BIRTH = (CODES["COVERAGE_GAP"], CODES["FRONT_SURFACE"])


@dataclass
class DiscoveryConfig4B:
    alpha_cov: float = 0.5
    depth_tol_assumed: float = 0.01      # m, ASSUMED when no calibrated sensor model is given
    depth_tol_floor: float = 0.002       # m, floor under k_sigma sigma_D when calibrated
    k_sigma: float = 3.0
    edge_px: int = 2
    depth_jump: float = 0.03             # m, |second difference| of observed depth = discontinuity (declared)
    amb_alpha: float = 0.5               # alternate-pose coverage that makes a gap ORIGINAL_MATERIAL_AMBIGUOUS
    voxel: float = 0.01
    min_obs: int = 3
    min_views: int = 2
    sigma_px: float = 0.5
    sigma_depth_assumed: float = 0.005   # m, ASSUMED measurement sd when no calibrated model
    support_k: float = 3.0               # kernel 3-sigma render support (not a sensor statistic)
    support_knn: int = 8
    layer_gap: float = 0.008             # m, split / never link layers further apart along the normal
    max_pixels_per_view: int = 20000
    role_block_px: int = 64
    role_guard_px: int = 6
    role_time_block: int = 8

    def to_dict(self):
        return asdict(self)

    def roles(self):
        return RoleConfig(self.role_block_px, self.role_guard_px, self.role_time_block)


# ------------------------------------------------------------------ sensor
class DepthNoise:
    def __init__(self, path, cfg: DiscoveryConfig4B):
        from utils.noise.sensor_model import load_sensor_model
        self.model = load_sensor_model(str(path)) if path else None
        self.cfg = cfg
        self.policy = self.model.policy if self.model is not None else "assumed"
        self.calibrated = self.model is not None and self.model.is_physical

    def sigma(self, camera: str, z):
        if self.calibrated:
            v = self.model.depth_variance(camera, z)
            return v.sqrt() if torch.is_tensor(v) else np.sqrt(v)
        return torch.full_like(z, self.cfg.sigma_depth_assumed) if torch.is_tensor(z) else \
            np.full_like(z, self.cfg.sigma_depth_assumed, dtype=float)

    def tol(self, camera: str, D: torch.Tensor):
        if self.calibrated:
            return torch.clamp(self.cfg.k_sigma * self.sigma(camera, D), min=self.cfg.depth_tol_floor)
        return torch.full_like(D, self.cfg.depth_tol_assumed)

    def manifest(self):
        return {"policy": self.policy, "calibrated": self.calibrated,
                "depth_tolerance": "max(depth_tol_floor, k_sigma sigma_D)" if self.calibrated else
                f"ASSUMED constant {self.cfg.depth_tol_assumed} m (not a sensor calibration)",
                "sigma_depth": "calibrated sigma_D(z)" if self.calibrated else
                f"ASSUMED {self.cfg.sigma_depth_assumed} m", "model_id": self.model.model_id if self.model else None}


# ------------------------------------------------------------------ classification
def _dilate(m, px):
    if px <= 0:
        return m
    return F.max_pool2d(m.float()[None, None], 2 * px + 1, 1, px)[0, 0] > 0.5


def edge_band(M, D, V, cfg):
    sil = _dilate(M, cfg.edge_px) & _dilate(~M, cfg.edge_px)
    d = torch.where(V, D, torch.zeros_like(D))
    jump = torch.zeros_like(M)
    sx = (d[:, 2:] - 2 * d[:, 1:-1] + d[:, :-2]).abs() > cfg.depth_jump
    sy = (d[2:, :] - 2 * d[1:-1, :] + d[:-2, :]).abs() > cfg.depth_jump
    jump[:, 1:-1] |= sx & V[:, 2:] & V[:, 1:-1] & V[:, :-2]
    jump[1:-1, :] |= sy & V[2:, :] & V[1:-1, :] & V[:-2, :]
    hole = M & ~V
    return sil | _dilate(jump | (_dilate(hole, 1) & V), cfg.edge_px)


def classify(out, alt_u, view, tol, cfg):
    A, Dh = out["alpha"][0], out["depth"][0]
    M, V, D = view.mask, view.depth_valid, view.depth
    code = torch.full(M.shape, CODES["EXPLAINED"], dtype=torch.uint8, device=M.device)
    code[~M] = CODES["BACKGROUND"]
    code[M & ~V] = CODES["DEPTH_INVALID"]
    cov = A >= cfg.alpha_cov
    gap = M & V & ~cov
    front = M & V & cov & (D < Dh - tol)
    code[gap] = CODES["COVERAGE_GAP"]
    code[front] = CODES["FRONT_SURFACE"]
    code[M & V & cov & (D > Dh + tol)] = CODES["BEHIND_PREDICTION"]
    if alt_u is not None:
        Au, Du = alt_u["alpha"][0], alt_u["depth"][0]
        amb = (gap | front) & (Au >= cfg.amb_alpha) & ((Du - D).abs() <= tol)
        code[amb] = CODES["ORIGINAL_MATERIAL_AMBIGUOUS"]
    code[M & V & edge_band(M, D, V, cfg)] = CODES["MIXED_EDGE"]
    return code


# ------------------------------------------------------------------ support containment (render footprint)
def on_support(y: np.ndarray, Py: np.ndarray, mu: torch.Tensor, rot: torch.Tensor, scale: torch.Tensor, k: int, kmax: float):
    """Bool [K]: the measured point lies inside the kernel's kmax-sigma support of one of its k nearest posed
    Gaussians, with the footprint inflated by the point's own measurement covariance."""
    if len(y) == 0 or len(mu) == 0:
        return np.zeros(len(y), bool)
    from scipy.spatial import cKDTree
    from .articulated_scene import quat_to_rotmat
    k = min(k, len(mu))
    _, j = cKDTree(mu.cpu().numpy()).query(y, k=k)
    j = torch.as_tensor(np.asarray(j).reshape(len(y), k), device=mu.device)
    R = quat_to_rotmat(rot[j])
    Sg = R @ torch.diag_embed(scale[j] ** 2) @ R.transpose(-1, -2)
    S = Sg + torch.as_tensor(Py, device=mu.device, dtype=mu.dtype)[:, None]
    dv = torch.as_tensor(y, device=mu.device, dtype=mu.dtype)[:, None, :] - mu[j]
    m2 = (dv[..., None, :] @ torch.linalg.solve(S, dv[..., None]))[..., 0, 0]
    return (m2.min(1).values <= kmax ** 2).cpu().numpy()


# ------------------------------------------------------------------ fusion
def fuse_layers(x, n, rgb, view, frame, src, cfg: DiscoveryConfig4B):
    """Cells keyed by (voxel, signed normal bin), split along the normal at gaps > layer_gap."""
    good = np.isfinite(n).all(1)
    x, n, rgb, view, frame, src = x[good], n[good], rgb[good], view[good], frame[good], src[good]
    if not len(x):
        return []
    nbin = np.argmax(n @ AXES.T, 1)
    key = np.concatenate([np.floor(x / cfg.voxel).astype(np.int64), nbin[:, None]], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    cells = []
    for c in range(len(uniq)):
        idx = order[bounds[c]:bounds[c + 1]]
        ax = AXES[uniq[c][3]]
        off = x[idx] @ ax
        srt = idx[np.argsort(off, kind="stable")]
        o = x[srt] @ ax
        cuts = np.nonzero(np.diff(o) > cfg.layer_gap)[0] + 1
        for li, part in enumerate(np.split(srt, cuts)):
            nv = len(np.unique(view[part]))
            if len(part) < cfg.min_obs or nv < cfg.min_views:
                continue
            xc = x[part]
            mu = np.median(xc, 0)
            nm = n[part].mean(0); nm /= max(np.linalg.norm(nm), 1e-12)
            r = (xc - mu) @ nm
            cells.append({"key": tuple(int(v) for v in uniq[c]), "layer": li, "bin": int(uniq[c][3]), "mu": mu,
                          "normal": nm, "rgb": np.median(rgb[part], 0), "count": int(len(part)), "views": nv,
                          "frames": int(len(np.unique(frame[part]))), "first_frame": int(frame[part].min()),
                          "src": src[part], "thickness": float(np.sqrt(np.mean(r ** 2)))})
    return cells


def patches(cells, cfg: DiscoveryConfig4B):
    """Touching cells (26-neighbourhood) of the same normal bin whose plane offsets agree within
    max(layer_gap, 2 sqrt(t_i^2 + t_j^2)) -- t = each cell's measured rms spread along its normal, so noisy
    observations of ONE surface still link while resolved parallel layers (split inside a cell at gaps >
    layer_gap) do not."""
    if not cells:
        return np.zeros(0, int)
    by_key = {}
    for i, c in enumerate(cells):
        by_key.setdefault(c["key"][:3] + (c["bin"],), []).append(i)
    parent = list(range(len(cells)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, c in enumerate(cells):
        kx, ky, kz, _ = c["key"]
        n = c["normal"]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in by_key.get((kx + dx, ky + dy, kz + dz, c["bin"]), ()):
                        tol = max(cfg.layer_gap, 2.0 * float(np.hypot(c["thickness"], cells[j]["thickness"])))
                        if j == i or abs(float(n @ (cells[j]["mu"] - c["mu"]))) > tol:
                            continue
                        a, b = find(i), find(j)
                        if a != b:
                            parent[max(a, b)] = min(a, b)
    lab = np.array([find(i) for i in range(len(cells))])
    return np.unique(lab, return_inverse=True)[1]


# ------------------------------------------------------------------ the stage
def run_4b(scene, pv, labels: torch.Tensor, status: np.ndarray, cfg: DiscoveryConfig4B, noise: DepthNoise, log=print):
    """Returns (codes [V] uint8 CPU tensors, per-view stats, observations per hypothesis, counters)."""
    from .phase4a_exact import JOINT_INVALID, PosedPair, SUSPECT, UNRESOLVED, UNTESTED
    dev = scene.device
    roles = Roles(scene.views, cfg.roles())
    undecided = torch.as_tensor(np.isin(status, [UNRESOLVED, SUSPECT, UNTESTED]), device=dev)
    alt_labels = torch.where(undecided, ~labels, labels)
    t = scene.tensors()
    codes, stats = [], []
    obs = {"S": [], "M": []}
    cnt = {"eligible_proposal_px": 0, "subsampled": 0, "on_existing_support": 0, "on_undecided_alternate": 0,
           "no_normal": 0, "backprojected": 0}
    t0 = time.time()
    last = t0
    for vi, vr in enumerate(scene.views):
        view = pv.gpu_view(vi, dev)
        pp = PosedPair(scene, vr, scene.transport)
        out = pp.render(labels)
        alt_u = pp.render(alt_labels, keep=undecided) if bool(undecided.any()) else None
        tol = noise.tol(vr.cam.name, view.depth)
        code = classify(out, alt_u, view, tol, cfg)
        role = roles.get(vi, dev)
        codes.append(code.cpu())
        b = torch.bincount(code.reshape(-1).long(), minlength=len(CODES)).cpu().numpy()
        st = {"camera": vr.cam.name, "frame": int(vr.frame), **{k: int(b[i]) for k, i in CODES.items()}}
        sel_t = torch.isin(code, torch.tensor(BIRTH, device=dev)) & (role == PROPOSAL)
        sel = sel_t.cpu().numpy()
        n_el = int(sel.sum())
        st["birth_source_px"] = n_el
        cnt["eligible_proposal_px"] += n_el
        if n_el:
            if n_el > cfg.max_pixels_per_view:
                idx = np.flatnonzero(sel)
                keep = idx[np.linspace(0, len(idx) - 1, cfg.max_pixels_per_view).round().astype(int)]
                sel = np.zeros_like(sel); sel.reshape(-1)[keep] = True
                cnt["subsampled"] += n_el - cfg.max_pixels_per_view
            depth = pv.depth[vi]
            yw, nw, pc = backproject(vr.cam, depth, sel)
            sd = noise.sigma(vr.cam.name, pc[:, 2])
            Py = point_covariance(vr.cam, pc, cfg.sigma_px, np.asarray(sd, float))
            mu = torch.where(labels[:, None], pp.muM, pp.muS)
            q = torch.where(labels[:, None], pp.qM, pp.qS)
            drop = on_support(yw, Py, mu, q, t["scale"], cfg.support_knn, cfg.support_k)
            cnt["on_existing_support"] += int(drop.sum())
            if bool(undecided.any()):
                ua = undecided.nonzero()[:, 0]
                mu_a = torch.where(alt_labels[:, None], pp.muM, pp.muS)[ua]
                q_a = torch.where(alt_labels[:, None], pp.qM, pp.qS)[ua]
                amb = on_support(yw, Py, mu_a, q_a, t["scale"][ua], cfg.support_knn, cfg.support_k) & ~drop
                cnt["on_undecided_alternate"] += int(amb.sum())
                drop |= amb
            keep = ~drop & np.isfinite(nw).all(1)
            cnt["no_normal"] += int((~np.isfinite(nw).all(1) & ~drop).sum())
            st["kept_points"] = int(keep.sum())
            if keep.any():
                idx = np.flatnonzero(sel)[keep]
                yw, nw = yw[keep], nw[keep]
                rgb = pv.cpu_view(vi).rgb.permute(1, 2, 0).numpy().reshape(-1, 3)[idx]
                Q, p = scene.joint.transform(vr.frame)
                src = vi * depth.size + idx
                for h in ("S", "M"):
                    x, nn = (yw, nw) if h == "S" else ((yw - p) @ Q, nw @ Q)
                    obs[h].append((x, nn, rgb, np.full(len(x), vi), np.full(len(x), vr.frame), src))
                cnt["backprojected"] += int(keep.sum())
        stats.append(st)
        if time.time() - last > 60:
            last = time.time()
            log(f"    4B: view {vi + 1}/{len(scene.views)}, birth-source px {cnt['eligible_proposal_px']}")
    log(f"  4B classified {len(scene.views)} views in {time.time() - t0:.1f} s; {cnt}")
    return codes, stats, obs, cnt, roles


def build_candidates(obs, cfg: DiscoveryConfig4B, pair_overlap=0.5):
    """Source-equivalent candidates (directive 6): every patch found under either hypothesis defines a SET OF
    SOURCE PIXELS; its STATIC and MOVING versions are both fused from exactly those pixels (the S and M
    observation arrays are pixel-aligned). Patches are taken largest first; a patch whose sources are already
    >= pair_overlap covered by an earlier candidate is not repeated. A hypothesis that smears (no cell reaches
    min_obs / min_views from those pixels) simply has no cells -- that is evidence, not a missing pair."""
    from .gaussians import MOVING, STATIC
    cat = {}
    for h in ("S", "M"):
        cat[h] = [np.concatenate([o[k] for o in obs[h]]) for k in range(6)] if obs[h] else None
    cells, lab = {}, {}
    for h in ("S", "M"):
        cells[h] = fuse_layers(*cat[h], cfg) if cat[h] is not None else []
        lab[h] = patches(cells[h], cfg)
    found = []
    for h in ("S", "M"):
        for l in (np.unique(lab[h]) if len(lab[h]) else []):
            cs = [cells[h][i] for i in np.nonzero(lab[h] == l)[0]]
            found.append((len(cs), h, cs, np.unique(np.concatenate([c["src"] for c in cs]))))
    found.sort(key=lambda f: (-f[0], f[1]))
    covered = np.zeros(0, np.int64)
    cands = []
    for n, h, cs, src in found:
        if len(covered) and np.isin(src, covered).mean() >= pair_overlap:
            continue
        o = "M" if h == "S" else "S"
        other = []
        if cat[o] is not None:
            m = np.isin(cat[o][5], src)
            if m.any():
                other = fuse_layers(*(a[m] for a in cat[o]), cfg)
        ms = np.isin(cat[h][5], src)
        disp = np.linalg.norm(cat["S"][0][ms] - cat["M"][0][ms], axis=1) if cat["S"] is not None and cat["M"] is not None \
            else np.zeros(0)
        cands.append({STATIC: cs if h == "S" else other, MOVING: cs if h == "M" else other, "overlap": 1.0,
                      "found_under": h, "sources": int(len(src)),
                      "source_motion_p90_m": float(np.percentile(disp, 90)) if len(disp) else 0.0})
        covered = np.union1d(covered, src)
    return cells, lab, cands


def stage_4b(run, cfg: DiscoveryConfig4B, loss_cfg=None):
    from .phase4_stages_legacy import load_4a
    log = run.log
    if not (run.marker("4a") or {}).get("complete"):
        raise SystemExit("stage 4b requires a complete 4a in this run directory")
    status, render_label = load_4a(run)
    labels = torch.as_tensor(render_label, device=run.scene.device)
    noise = DepthNoise(run.up.sensor_model, cfg)
    log(f"4B: residual discovery on the 4A scene; sensor {noise.manifest()['policy']} "
        f"({'calibrated' if noise.calibrated else 'ASSUMED tolerances'})")
    t0 = time.time()
    codes, stats, obs, cnt, roles = run_4b(run.scene, run.pv, labels, status, cfg, noise, log=log)
    cells, lab, cands = build_candidates(obs, cfg)
    d = run.dir / "discovery"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "residual_codes.npz", codes=torch.stack(codes).numpy(), legend=json.dumps(CODES))
    torch.save({"cells": cells, "patch_label": lab, "candidates": cands, "config": cfg.to_dict()}, d / "proposals.pt")
    summary = {h: {"cells": len(cells[h]), "patches": int(len(np.unique(lab[h]))) if len(lab[h]) else 0,
                   "mean_frames_per_cell": float(np.mean([c["frames"] for c in cells[h]])) if cells[h] else 0.0}
               for h in ("S", "M")}
    atomic_json(d / "discovery_manifest.json", {
        "config": cfg.to_dict(), "sensor": noise.manifest(), "roles": roles.manifest(), "counters": cnt,
        "per_view": stats, "proposals": summary, "candidates": len(cands),
        "birth_sources": "COVERAGE_GAP | FRONT_SURFACE in the PROPOSAL role only",
        "uncertainty": {"measurement": "P_y propagated per point", "shared_joint": "not propagated; "
                        "conditional_on_phase3_joint", "rendering": "3DGS footprint, used only for the render-support "
                        "containment test (support_k = kernel 3-sigma)"},
        "thresholds_status": "declared assumptions (alpha_cov, depth_jump, layer_gap, voxel), not calibrated",
        "seconds": time.time() - t0})
    marker = {"stage": "4b", "complete": True, "candidates": len(cands), "proposals": summary, "counters": cnt,
              "resources": log.resources(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "NOT_FINAL": bool(run.manifest.get("NOT_FINAL"))}
    atomic_json(run.dir / "stage4b_complete.json", marker)
    run.manifest.setdefault("stages", {})["4b"] = {"complete": True, "candidates": len(cands)}
    run.write_manifest()
    log(f"4B COMPLETE: {len(cands)} candidates; {summary}")
    return marker
