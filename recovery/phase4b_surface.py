"""Phase 4B (unified spec 2026-11-11, section 5): NEW moving-surface PROPOSALS only -- zero births.

Input: an immutable, complete 4A artifact (or, inside a 4C growth cycle, the frozen authoritative scene G^k).
Output: candidate canonical surface patches with full provenance. Nothing is added to any scene here.

1. Single view-major scan (5.2). Per (camera, original frame): ONE native render of the complete scene plus two
   auxiliary passes of the same depth-sorted compositing with indicator colours (background 0):
       A_M(p) = sum_{i moving}  w_i(p)        visible moving responsibility (all static + moving occluders)
       A_T(p) = sum_{i trusted} w_i(p)        trusted moving support (core + MOVING_CONFIRMED + accepted births)
       s_Z(p) = sqrt(E_w[z^2] - E_w[z]^2)     spread of the rendered contributors' depths: the expected depth
                                               Dhat = Z / A of a pixel with large s_Z is a LAYER MIXTURE, not a surface
2. Residual taxonomy (5.3), per observed pixel, codes below. Tolerances: tol(p) = k sqrt(sigma_D(D)^2 + sigma_model^2)
   with sigma_D CALIBRATED (utils/noise/sensor_model.py) or the declared ASSUMED value (outputs say which).
     MIXED_EDGE_AMBIGUOUS -- observation-side mixed return, decided from the measured depth image alone: along any
       of 4 directions the least-squares lines through the samples on the two sides (offsets gap+1..gap+L) disagree
       at the pixel (a discontinuity) AND the pixel's depth fits NEITHER line, both at k sigma_D (prediction variance
       propagated). A slanted plane (lines agree) and a crease (pixel fits one side) are never flagged. No fixed
       mask-erosion band, no blanket border rejection; flagged pixels are kept in the audit, never back-projected.
     FRONT_SURFACE if D < Dhat - tol (- render_layer_k s_Z when that declared option is > 0). Dhat of a pixel with a
       large rendered spread s_Z is a layer mixture, so FRONT is never trusted by itself: every GAP / FRONT point is
       tested in 3D against the render support of the actual Gaussians (next item); s_Z > tol is counted.
     A GAP / FRONT point inside the kernel 3-sigma render support of an existing Gaussian (inflated by its own
       measurement covariance) is ORDINARY_GEOMETRY_RESIDUAL (existing-geometry correction alternative, no birth);
       inside the support of an UNDECIDED original's alternate pose it is ORIGINAL_MEMBERSHIP_CONFLICT (a return to
       4A, never a duplicate).
   Birth sources = COVERAGE_GAP | FRONT_SURFACE pixels of the PROPOSAL role (data_roles.py, assigned before any residual).
3. Canonical accumulation (5.5): y_W = R_WC d K^-1 [u+.5, v+.5, 1] + t_WC;  x_M = Q_t^T (y_W - p_t);  x_S = y_W.
   Measurement covariance propagated per point (P_y); the shared joint is NOT propagated per point (conditional).
4. Layer identity (5.6), no global min-layer-distance:
     fragments  per view, 4-neighbour pixels linked only if BOTH one-sided 5-sample line fits predict the pixel across
                within k sigma_D sqrt(Var + 1) (a depth step = a different layer IN THAT VIEW)
     cells      (view, fragment, canonical voxel); normal = plane fit over every sample within normal_radius seen from
                the same side (camera on the same side), oriented to the observing camera (sidedness), with a propagated
                angular sd; cells with an unreliable normal or < min_cell_link samples never link;
                split in two along the normal when a 2-means split exceeds the single-layer null (1.6 s + k s sqrt(1/n1+1/n2))
     co-layer   cells of adjacent voxels with agreeing normals (n_i . n_j >= normal_cos > 0, so opposite faces never
                link), never two different fragments of one view, and plane offsets agreeing at CELL precision:
                max(|n_i.(mu_j - mu_i)|, |n_j.(mu_i - mu_j)|) <= k [sqrt(se_i^2 + se_j^2 + 2 sigma_layer^2) + |dmu_t| theta]
                with theta = the larger propagated normal sd; cells with an unreliable normal (strips, edges) never link
     corner     adjacent cells with different normals whose centres lie within corner_reach of each other's plane
                (supported 90-degree edges; a separate graph, recorded, used for frontier reporting only)
   patches = connected components of the co-layer graph.
5. Frontier (5.4): a patch is proposed only if one of its cells lies within frontier_reach of the 3-sigma render
   support of a TRUSTED moving Gaussian (core, MOVING_CONFIRMED, or an accepted birth). Unresolved originals never
   seed. `disconnected_search` (off by default) is a separately labelled second mode with a stricter support rule.
6. Candidates (5.7): MOVING primary (one Gaussian per (patch, voxel)); the source-equivalent STATIC competitor fused
   from exactly the same samples in world coordinates; provenance, support groups (camera x time block), frontier
   parents, layer ids, conflicts. Patches with fewer than min_support_groups independent groups are still emitted
   (evidence reported honestly); 4C decides.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .canonical_fusion import backproject, point_covariance

CODES = {"EXPLAINED": 0, "APPEARANCE_ONLY": 1, "ORDINARY_GEOMETRY_RESIDUAL": 2, "COVERAGE_GAP": 3, "FRONT_SURFACE": 4,
         "BEHIND_PREDICTION": 5, "MIXED_EDGE_AMBIGUOUS": 6, "DEPTH_INVALID": 7, "OUTSIDE_OBJECT_MASK": 8,
         "ORIGINAL_MEMBERSHIP_CONFLICT": 9}
BIRTH = (CODES["COVERAGE_GAP"], CODES["FRONT_SURFACE"])


@dataclass
class Surface4B:
    alpha_cov: float = 0.5
    sigma_depth_assumed: float = 0.005     # m, ASSUMED sensor sd when no calibrated model is given
    sigma_model: float = -1.0              # m, rendering / model discrepancy; < 0 = AUTO: max(sigma_model_floor, robust
                                           # 1.4826 MAD of (Dhat - D) on covered reliable pixels of a view subset)
    sigma_model_floor: float = 0.001
    sigma_layer: float = -1.0              # m, discrepancy BETWEEN co-layer cells of different views; < 0 = AUTO (estimated
                                           # in link_cells from the data, floor sigma_layer_floor)
    sigma_layer_floor: float = 0.001
    k_sigma: float = 3.0
    mixed_gap: int = 1                     # px skipped next to the pixel on each side
    mixed_len: int = 3                     # px per side in the line fits
    rgb_tol: float = 0.1                   # mean |I - Ihat| above this on a geometrically explained pixel: APPEARANCE_ONLY
    render_layer_k: float = 0.0          # >0: FRONT only if D < Dhat - tol - k s_Z. Default 0: the rendered-layer
                                         # spread is a recorded diagnostic and the 3D render-support containment
                                         # test (ORDINARY_GEOMETRY_RESIDUAL) decides model-side mixtures
    voxel: float = 0.01
    min_samples: int = 6                   # per patch
    min_support_groups: int = 2            # (camera, time block) groups; fewer -> emitted but flagged
    normal_cos: float = 0.8
    normal_err: float = 0.05               # rad, declared cell-normal uncertainty in the co-layer test
    normal_radius_voxels: float = 1.5      # cell normal: plane fit over its fragment's samples within this radius
    normal_max_err: float = 0.05           # rad; a cell whose propagated normal sd exceeds this never links
    min_cell_link: int = 2                 # samples a cell needs to take part in co-layer / corner links
    corner_reach_voxels: float = 1.0
    frontier_reach_voxels: float = 2.0
    support_k: float = 3.0                 # kernel 3-sigma render support (a footprint, not a sensor statistic)
    support_knn: int = 8
    sigma_px: float = 0.5
    max_pixels_per_view: int = 20000
    role_block_px: int = 64
    role_guard_px: int = 6
    role_time_block: int = 8
    disconnected_search: bool = False

    def to_dict(self):
        return asdict(self)

    def roles(self):
        from .data_roles import RoleConfig
        return RoleConfig(self.role_block_px, self.role_guard_px, self.role_time_block)


class SensorNoise:
    def __init__(self, path, cfg: Surface4B, sigma_model=None):
        self.sigma_model = cfg.sigma_model if sigma_model is None else float(sigma_model)
        self.model = None
        if path:
            from utils.noise.sensor_model import load_sensor_model
            self.model = load_sensor_model(str(path))
        self.cfg = cfg
        self.calibrated = self.model is not None and self.model.is_physical

    def sigma(self, camera: str, z: torch.Tensor) -> torch.Tensor:
        if self.calibrated:
            v = self.model.depth_variance(camera, z)
            return (v if torch.is_tensor(v) else torch.as_tensor(v, device=z.device)).float().sqrt()
        return torch.full_like(z, self.cfg.sigma_depth_assumed)

    def tol(self, camera, D):
        s = self.sigma(camera, D)
        if self.sigma_model < 0:
            raise RuntimeError("sigma_model is AUTO and has not been estimated (call estimate_sigma_model)")
        return self.cfg.k_sigma * torch.sqrt(s * s + self.sigma_model ** 2)

    def manifest(self):
        return {"calibrated": self.calibrated,
                "sigma_depth": "calibrated sigma_D(z)" if self.calibrated else
                f"ASSUMED {self.cfg.sigma_depth_assumed} m (not a sensor calibration)",
                "sigma_model_m": self.sigma_model, "sigma_model_source": getattr(self, "sigma_model_source", "declared"),
                "k_sigma": self.cfg.k_sigma,
                "model_id": getattr(self.model, "model_id", None)}


# =========================================================================== observation-side mixed depth
def _shift(x, dy, dx, fill):
    """out[y, x] = x[y + dy, x + dx] (fill outside)."""
    H, W = x.shape
    out = torch.full_like(x, fill)
    ys, yd = (slice(dy, H), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, H))
    xs, xd = (slice(dx, W), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, W))
    out[yd, xd] = x[ys, xs]
    return out


def mixed_depth_map(D: torch.Tensor, V: torch.Tensor, sigma: torch.Tensor, k=3.0, gap=1, L=3) -> torch.Tensor:
    """Bool [H,W]: measured depth inconsistent with a single surface (see module doc, item 2)."""
    D = D.double()
    sig = sigma.double()
    flag = torch.zeros_like(V)
    offs = torch.arange(gap + 1, gap + L + 1, dtype=torch.float64)
    xm = float(offs.mean())
    xc = offs - xm
    sxx = float((xc ** 2).sum())
    var_f = 1.0 / L + xm ** 2 / sxx                       # Var(line prediction at 0) / sigma^2
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        preds, oks = [], []
        for s in (1, -1):
            Y, O = [], []
            for kk in range(gap + 1, gap + L + 1):
                Y.append(_shift(D, s * kk * dy, s * kk * dx, 0.0))
                O.append(_shift(V, s * kk * dy, s * kk * dx, False))
            Y = torch.stack(Y)
            b = (xc[:, None, None].to(D.device) * Y).sum(0) / sxx
            preds.append(Y.mean(0) - b * xm)              # line evaluated at offset 0 (offsets measured outward)
            oks.append(torch.stack(O).all(0))
        ok = V & oks[0] & oks[1]
        disagree = (preds[0] - preds[1]).abs() > k * sig * np.sqrt(2 * var_f)
        fa = (D - preds[0]).abs() <= k * sig * np.sqrt(var_f + 1)
        fb = (D - preds[1]).abs() <= k * sig * np.sqrt(var_f + 1)
        flag |= ok & disagree & ~fa & ~fb
    return flag


# =========================================================================== support containment
def on_support(y: np.ndarray, Py: np.ndarray, mu: torch.Tensor, rot: torch.Tensor, scale: torch.Tensor, k: int,
               kmax: float) -> np.ndarray:
    """Bool [K]: y within the kmax-sigma kernel support of one of its k nearest Gaussians, footprint inflated by Py."""
    if len(y) == 0 or len(mu) == 0:
        return np.zeros(len(y), bool)
    from scipy.spatial import cKDTree
    from .articulated_scene import quat_to_rotmat
    k = min(k, len(mu))
    _, j = cKDTree(mu.detach().cpu().numpy()).query(y, k=k)
    j = torch.as_tensor(np.asarray(j).reshape(len(y), k), device=mu.device)
    out = np.zeros(len(y), bool)
    for a in range(0, len(y), 65536):
        jj = j[a:a + 65536]
        R = quat_to_rotmat(rot[jj])
        S = R @ torch.diag_embed(scale[jj] ** 2) @ R.transpose(-1, -2) + \
            torch.as_tensor(Py[a:a + 65536], device=mu.device, dtype=mu.dtype)[:, None]
        dv = torch.as_tensor(y[a:a + 65536], device=mu.device, dtype=mu.dtype)[:, None, :] - mu[jj]
        m2 = (dv[..., None, :] @ torch.linalg.solve(S, dv[..., None]))[..., 0, 0]
        out[a:a + 65536] = (m2.min(1).values <= kmax ** 2).cpu().numpy()
    return out


# =========================================================================== auxiliary compositing passes
def aux_render(pp, labels: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
    """Same Gaussians / poses / opacities / depth order as pp.render(labels), colours replaced, background 0.
    Returns [3,H,W] = sum_i w_i(p) colors_i."""
    from diff_gaussian_rasterization import GaussianRasterizer
    m = labels[:, None]
    mu = torch.where(m, pp.muM, pp.muS)
    q = torch.where(m, pp.qM, pp.qS)
    R0 = GaussianRasterizer(pp.R.raster_settings._replace(bg=torch.zeros(3, device=mu.device)))
    c, *_ = R0.forward_rgbd(mu, torch.zeros(mu.shape[0], 3, device=mu.device), pp.t["opacity"],
                            colors_precomp=colors.float().contiguous(), scales=pp.t["scale"], rotations=q)
    return c


def camera_z(cam, mu):
    V = cam.V.to(mu.device)
    return (mu @ V[:3, :3] + V[3, :3])[:, 2]


def estimate_sigma_model(scene, pv, labels, noise: SensorNoise, cfg: Surface4B, max_views=48):
    """AUTO model discrepancy: robust spread of (Dhat - D) on covered, valid, non-mixed pixels of an evenly spaced view
    subset of the 4A scene. Floor: sigma_model_floor. Recorded as an estimate, not a sensor calibration."""
    from .phase4a_exact import PosedPair
    if cfg.sigma_model >= 0:
        noise.sigma_model, noise.sigma_model_source = cfg.sigma_model, "declared"
        return {"sigma_model_m": cfg.sigma_model, "source": "declared"}
    ids = np.unique(np.linspace(0, len(scene.views) - 1, min(max_views, len(scene.views))).round().astype(int))
    res = []
    for vi in ids:
        vr = scene.views[vi]
        view = pv.gpu_view(int(vi), scene.device)
        with torch.no_grad():
            out = PosedPair(scene, vr, scene.transport).render(labels)
        D, V = view.depth, view.depth_valid
        m = V & ~mixed_depth_map(D, V, noise.sigma(vr.cam.name, D), cfg.k_sigma, cfg.mixed_gap, cfg.mixed_len) & \
            (out["alpha"][0] >= cfg.alpha_cov)
        r = (out["depth"][0] - D)[m]
        if r.numel():
            res.append(r[torch.randperm(r.numel(), device=r.device)[:20000]].cpu().numpy())
    r = np.concatenate(res) if res else np.zeros(0)
    mad = float(1.4826 * np.median(np.abs(r - np.median(r)))) if len(r) else 0.0
    noise.sigma_model = max(cfg.sigma_model_floor, mad)
    noise.sigma_model_source = "AUTO: max(floor, robust MAD of 4A-scene depth residuals)"
    return {"sigma_model_m": noise.sigma_model, "mad_m": mad, "floor_m": cfg.sigma_model_floor, "views": len(ids),
            "pixels": int(len(r)), "source": noise.sigma_model_source}


# =========================================================================== the scan
def scan(scene, pv, labels: torch.Tensor, trusted: torch.Tensor, undecided: torch.Tensor, cfg: Surface4B,
         noise: SensorNoise, roles, log=print, keep_codes=True):
    """One view-major pass. Returns (codes list[uint8 CPU], per-view stats, samples dict of arrays, counters)."""
    from .phase4a_exact import PosedPair
    dev = scene.device
    t = scene.tensors()
    alt_labels = torch.where(undecided, ~labels, labels)
    frames = list(scene.joint.frames)
    codes, stats = [], []
    S = {k: [] for k in ("view", "cam", "frame", "row", "pix", "D", "sigma", "rgb", "yW", "nW", "xM", "nM", "camS",
                         "camM", "group")}
    cnt = {k: 0 for k in ("birth_source_px", "subsampled", "on_existing_support", "membership_conflict", "no_normal",
                          "backprojected", "model_layer_mixture_px", "mixed_px")}
    cam_names = []
    t0 = last = time.time()
    for vi, vr in enumerate(scene.views):
        if vr.cam.name not in cam_names:
            cam_names.append(vr.cam.name)
        view = pv.gpu_view(vi, dev)
        pp = PosedPair(scene, vr, scene.transport)
        with torch.no_grad():
            out = pp.render(labels)
            mu = torch.where(labels[:, None], pp.muM, pp.muS)
            z = camera_z(vr.cam, mu)
            c1 = aux_render(pp, labels, torch.stack([labels.float(), trusted.float(), z], 1))
            c2 = aux_render(pp, labels, torch.stack([z * z, undecided.float(), torch.zeros_like(z)], 1))
        A, Dh = out["alpha"][0], out["depth"][0]
        Af = A.clamp_min(1e-6)
        sZ = torch.sqrt(torch.clamp(c2[0] / Af - (c1[2] / Af) ** 2, min=0.0))
        M, V, D, I = view.mask, view.depth_valid, view.depth, view.rgb
        sig = noise.sigma(vr.cam.name, D)
        tol = noise.tol(vr.cam.name, D)
        mixed = mixed_depth_map(D, V, sig, cfg.k_sigma, cfg.mixed_gap, cfg.mixed_len)
        cov = A >= cfg.alpha_cov
        code = torch.full(M.shape, CODES["EXPLAINED"], dtype=torch.uint8, device=dev)
        code[~M] = CODES["OUTSIDE_OBJECT_MASK"]
        code[M & ~V] = CODES["DEPTH_INVALID"]
        good = M & V & ~mixed
        code[M & V & mixed] = CODES["MIXED_EDGE_AMBIGUOUS"]
        gap = good & ~cov
        front_any = good & cov & (D < Dh - tol)
        front = front_any & (D < Dh - tol - cfg.render_layer_k * sZ)
        behind = good & cov & (D > Dh + tol)
        app = good & cov & ~front_any & ~behind & ((out["color"] - I).abs().mean(0) > cfg.rgb_tol)
        code[app] = CODES["APPEARANCE_ONLY"]
        code[front_any & ~front] = CODES["ORDINARY_GEOMETRY_RESIDUAL"]
        code[behind] = CODES["BEHIND_PREDICTION"]
        code[gap] = CODES["COVERAGE_GAP"]
        code[front] = CODES["FRONT_SURFACE"]
        cnt["model_layer_mixture_px"] += int((front_any & (sZ > tol)).sum())
        cnt["mixed_px"] += int((M & V & mixed).sum())
        role = roles.get(vi, dev)
        from .data_roles import PROPOSAL
        sel_t = (gap | front) & (role == PROPOSAL)
        n_el = int(sel_t.sum())
        cnt["birth_source_px"] += n_el
        st = {"camera": vr.cam.name, "frame": int(vr.frame), "birth_source_px": n_el,
              "moving_responsibility_in_mask": float(c1[0][M].mean()) if bool(M.any()) else 0.0,
              "trusted_responsibility_in_mask": float(c1[1][M].mean()) if bool(M.any()) else 0.0,
              "undecided_responsibility_in_mask": float(c2[1][M].mean()) if bool(M.any()) else 0.0}
        if n_el:
            sel = sel_t.cpu().numpy()
            if n_el > cfg.max_pixels_per_view:
                idx = np.flatnonzero(sel)
                keep = idx[np.linspace(0, len(idx) - 1, cfg.max_pixels_per_view).round().astype(int)]
                sel = np.zeros_like(sel)
                sel.reshape(-1)[keep] = True
                cnt["subsampled"] += n_el - cfg.max_pixels_per_view
            depth = pv.depth[vi] if not isinstance(pv.depth[vi], torch.Tensor) else pv.depth[vi].numpy()
            depth = np.asarray(depth, np.float32)
            yw, nw, pc = backproject(vr.cam, depth, sel)
            sd = sig.cpu().numpy()[sel]
            Py = point_covariance(vr.cam, pc, cfg.sigma_px, sd.astype(float))
            q = torch.where(labels[:, None], pp.qM, pp.qS)
            drop = on_support(yw, Py, mu, q, t["scale"], cfg.support_knn, cfg.support_k)
            conflict = np.zeros(len(yw), bool)
            if bool(undecided.any()):
                ua = undecided.nonzero()[:, 0]
                mu_a = torch.where(alt_labels[:, None], pp.muM, pp.muS)[ua]
                q_a = torch.where(alt_labels[:, None], pp.qM, pp.qS)[ua]
                conflict = on_support(yw, Py, mu_a, q_a, t["scale"][ua], cfg.support_knn, cfg.support_k) & ~drop
            idx_all = np.flatnonzero(sel)
            flat = code.view(-1)
            if drop.any():
                flat[torch.as_tensor(idx_all[drop], device=dev)] = CODES["ORDINARY_GEOMETRY_RESIDUAL"]
            if conflict.any():
                flat[torch.as_tensor(idx_all[conflict], device=dev)] = CODES["ORIGINAL_MEMBERSHIP_CONFLICT"]
            cnt["on_existing_support"] += int(drop.sum())
            cnt["membership_conflict"] += int(conflict.sum())
            okn = np.isfinite(nw).all(1)
            cnt["no_normal"] += int((~okn & ~drop & ~conflict).sum())
            keep = ~drop & ~conflict & okn
            st["kept_samples"] = int(keep.sum())
            if keep.any():
                idx = idx_all[keep]
                Q, p = scene.joint.transform(vr.frame)
                Q, p = np.asarray(Q, float), np.asarray(p, float)
                C = vr.cam.c2w[:3, 3].astype(float)
                rgbv = pv.cpu_view(vi).rgb.permute(1, 2, 0).numpy().reshape(-1, 3)[idx]
                row = int(roles.row[vi])
                n = len(idx)
                S["view"].append(np.full(n, vi)); S["cam"].append(np.full(n, cam_names.index(vr.cam.name)))
                S["frame"].append(np.full(n, int(vr.frame))); S["row"].append(np.full(n, row))
                S["pix"].append(idx); S["D"].append(depth.reshape(-1)[idx]); S["sigma"].append(sd[keep])
                S["rgb"].append(rgbv); S["yW"].append(yw[keep]); S["nW"].append(nw[keep])
                S["xM"].append((yw[keep] - p) @ Q); S["nM"].append(nw[keep] @ Q)
                S["camS"].append(np.repeat(C[None], n, 0)); S["camM"].append(np.repeat(((C - p) @ Q)[None], n, 0))
                S["group"].append(np.full(n, cam_names.index(vr.cam.name) * 100000 + row // cfg.role_time_block))
                cnt["backprojected"] += n
        b = torch.bincount(code.reshape(-1).long(), minlength=len(CODES)).cpu().numpy()
        st.update({k: int(b[i]) for k, i in CODES.items()})
        stats.append(st)
        codes.append(code.cpu() if keep_codes else None)
        if time.time() - last > 60:
            last = time.time()
            log(f"    4B scan: view {vi + 1}/{len(scene.views)}, birth-source px {cnt['birth_source_px']} "
                f"({(vi + 1) / (last - t0):.1f} views/s)")
    samples = {k: (np.concatenate(v) if v else np.zeros((0,) + ((3,) if k in ("rgb", "yW", "nW", "xM", "nM", "camS", "camM")
                                                                  else ()))) for k, v in S.items()}
    log(f"  4B scan: {len(scene.views)} views in {time.time() - t0:.1f} s; {cnt}")
    return codes, stats, samples, cnt


# =========================================================================== layers: fragments, cells, patches
def _line_weights(L: int, at: float):
    """Least-squares line through samples at positions 0, -1, ..., -(L-1), evaluated at `at`: weights and Var/sigma^2."""
    x = -np.arange(L, dtype=float)
    X = np.stack([np.ones(L), x], 1)
    w = np.array([1.0, at]) @ np.linalg.inv(X.T @ X) @ X.T
    return w, float((w ** 2).sum())


def fragments(samples: dict, depth_of_view, valid_of_view, sigma_of_view, H_of_view, W_of_view, cfg: Surface4B, L: int = 5):
    """Per-view range-image fragments of the kept samples (int array aligned with samples).

    4-neighbours p, q = p + e are linked only if BOTH one-sided lines agree: the least-squares line through
    p, p - e, ..., p - (L-1) e predicts D(q), and the line through q, q + e, ..., q + (L-1) e predicts D(p), each within
    k sigma sqrt(Var_pred + 1). A depth step must therefore fool two independent L-sample fits at once; a slanted
    plane or a crease passes. Where a line is unavailable (fewer than L valid samples), only the direct test
    |D(p) - D(q)| <= k sqrt(2) sigma links (conservative)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    frag = np.full(len(samples["view"]), -1, np.int64)
    w, var = _line_weights(L, 1.0)
    nxt = 0
    for vi in np.unique(samples["view"]):
        sel = np.nonzero(samples["view"] == vi)[0]
        H, W = H_of_view(vi), W_of_view(vi)
        D = depth_of_view(vi).astype(float)
        V = valid_of_view(vi).astype(bool)
        sg = sigma_of_view(vi)
        pix = samples["pix"][sel]
        local = np.full(H * W, -1, np.int64)
        local[pix] = np.arange(len(sel))
        rows, cols = [], []
        for dy, dx in ((0, 1), (1, 0)):
            y, x = np.divmod(pix, W)
            yb, xb = y + dy, x + dx
            ok = (yb < H) & (xb < W)
            q = np.where(ok, local[np.clip(yb, 0, H - 1) * W + np.clip(xb, 0, W - 1)], -1)
            ok &= q >= 0
            a, b = np.nonzero(ok)[0], q[ok]
            ya, xa, yb, xb = y[a], x[a], yb[a], xb[a]
            s = np.maximum(sg[ya, xa], sg[yb, xb])

            def line(y0, x0, step):                         # samples y0 - k step, k = 0..L-1 (step = +e or -e)
                pred = np.zeros(len(y0))
                okl = np.ones(len(y0), bool)
                for k in range(L):
                    yy, xx = y0 - k * step[0], x0 - k * step[1]
                    inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                    yc, xc = np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)
                    okl &= inside & V[yc, xc]
                    pred += w[k] * D[yc, xc]
                return pred, okl
            pa, oka = line(ya, xa, (dy, dx))                # p side, predicts q
            pb, okb = line(yb, xb, (-dy, -dx))              # q side, predicts p
            da, db = D[ya, xa], D[yb, xb]
            tau = cfg.k_sigma * s * np.sqrt(var + 1.0)
            both = oka & okb
            link = np.where(both, (np.abs(pa - db) <= tau) & (np.abs(pb - da) <= tau),
                            np.abs(da - db) <= cfg.k_sigma * np.sqrt(2.0) * s)
            rows.append(a[link]); cols.append(b[link])
        r = np.concatenate(rows) if rows else np.zeros(0, int)
        c = np.concatenate(cols) if cols else np.zeros(0, int)
        n = len(sel)
        g = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
        k, lab = connected_components(g, directed=False)
        frag[sel] = nxt + lab
        nxt += k
    return frag


def _pca_normal(X, fallback, cam_center, sigma=0.001, max_err=0.1):
    """(normal oriented to the camera, angular sd estimate, reliable). Reliable = the samples spread in two directions
    (w1 >= 0.1 w2), are flat at this scale (w0 <= 0.3 w1) and the propagated tilt sd sigma / sqrt(w1) <= max_err
    (w = eigenvalues of the scatter matrix, ascending). Otherwise the mean measured normal, flagged unreliable."""
    rel, err = False, np.pi / 2
    n = fallback
    if len(X) >= 3:
        Xc = X - X.mean(0)
        w, v = np.linalg.eigh(Xc.T @ Xc)
        if w[2] > 1e-18 and w[1] >= 0.1 * w[2] and w[0] <= 0.3 * w[1]:
            err = float(sigma / np.sqrt(max(w[1], 1e-30)))
            n = v[:, 0]
            rel = err <= max_err
    n = n / max(np.linalg.norm(n), 1e-12)
    if n @ (cam_center - X.mean(0)) < 0:                            # sidedness: faces the observing camera
        n = -n
    return n, min(err, np.pi / 2), rel


def robust_normal(X, nb, loc, mc, fallback, cam_center, sg, cfg: Surface4B):
    """Plane through the pooled same-side neighbours `nb` of a cell, restricted to INLIERS of the current plane estimate
    (|n.(x - mc)| <= k sg): neighbours across a depth step (another layer) never tilt the normal. Two refinements from
    the cell's own estimate; unreliable if the inlier residual exceeds 2 sg (still not one plane)."""
    n, err, rel = _pca_normal(X[loc], fallback, cam_center, sg, cfg.normal_max_err)
    if len(nb) < 6:
        return n, err, rel
    for _ in range(2):
        r = (X[nb] - mc) @ n
        inl = nb[np.abs(r) <= cfg.k_sigma * sg]
        if len(inl) < 6:
            return n, np.pi / 2, False
        n2, err2, rel2 = _pca_normal(X[inl], n, cam_center, sg, cfg.normal_max_err)
        if not rel2:
            return n, err, False
        n, err = n2, err2
    rms = float(np.sqrt(np.mean(((X[inl] - X[inl].mean(0)) @ n) ** 2)))
    return n, err, bool(err <= cfg.normal_max_err and rms <= 2 * sg)


def build_cells(samples: dict, frag: np.ndarray, hyp: str, idx: np.ndarray, cfg: Surface4B):
    """Cells keyed by (view, fragment, voxel) of the samples `idx` under hypothesis 'M' or 'S'."""
    X = samples["xM" if hyp == "M" else "yW"][idx]
    N = samples["nM" if hyp == "M" else "nW"][idx]
    Cc = samples["camM" if hyp == "M" else "camS"][idx]
    vox = np.floor(X / cfg.voxel).astype(np.int64)
    key = np.concatenate([samples["view"][idx][:, None], frag[idx][:, None], vox], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    cells = []
    s_mod = max(cfg.sigma_layer, cfg.sigma_layer_floor)
    # cell normals: plane fit over ALL samples (every view) within normal_radius whose observing camera is on the same
    # side of the surface as the cell's camera (parallel layers share the normal, so pooling them is harmless; the side
    # test keeps the two faces of a thin panel apart). Per-pixel depth-gradient normals are only the last fallback.
    from scipy.spatial import cKDTree
    tree_all = cKDTree(X) if len(X) else None
    rad = cfg.normal_radius_voxels * cfg.voxel
    for c in range(len(uniq)):
        loc = order[bounds[c]:bounds[c + 1]]
        fb = N[loc].mean(0)
        if not np.isfinite(fb).all() or np.linalg.norm(fb) < 1e-9:
            continue
        mc = np.median(X[loc], 0)
        nb = np.asarray(tree_all.query_ball_point(mc, rad), np.int64)
        cside = Cc[loc].mean(0) - mc
        nb = nb[((Cc[nb] - mc) @ cside) > 0] if len(nb) else nb
        sg_c = float(np.sqrt(np.mean(samples["sigma"][idx][loc] ** 2) + s_mod ** 2))
        n, n_err, n_rel = robust_normal(X, nb, loc, mc, fb, Cc[loc].mean(0), sg_c, cfg)
        parts = [loc]
        if len(loc) >= 4:
            r = (X[loc] - X[loc].mean(0)) @ n
            o = np.argsort(r)
            rs = r[o]
            n_ = len(rs)
            c1, c2 = np.cumsum(rs), np.cumsum(rs ** 2)
            j = np.arange(2, n_ - 1)                       # split sizes (both parts >= 2)
            cut = None
            if len(j):
                s1, q1 = c1[j - 1], c2[j - 1]
                s2, q2 = c1[-1] - s1, c2[-1] - q1
                ss = (q1 - s1 ** 2 / j) + (q2 - s2 ** 2 / (n_ - j))
                cut = int(j[np.argmin(ss)])
            if cut is not None:
                a, b = rs[:cut], rs[cut:]
                s = float(np.sqrt(np.mean(samples["sigma"][idx][loc] ** 2) + s_mod ** 2))
                sep = b.mean() - a.mean()
                if sep > 1.6 * s + cfg.k_sigma * s * np.sqrt(1 / len(a) + 1 / len(b)):
                    parts = [loc[o[:cut]], loc[o[cut:]]]
        for li, part in enumerate(parts):
            Xp = X[part]
            mu = np.median(Xp, 0)
            nn = n
            r = (Xp - mu) @ nn
            s_meas = float(np.sqrt(np.mean(samples["sigma"][idx][part] ** 2)))
            rms = float(np.sqrt(np.mean(r ** 2)))
            se = max(rms, s_meas) / np.sqrt(len(part))
            cells.append({"view": int(uniq[c][0]), "frag": int(uniq[c][1]), "vox": tuple(int(v) for v in uniq[c][2:]),
                          "sub": li, "split": len(parts) > 1, "mu": mu, "normal": nn, "normal_err": n_err,
                          "normal_reliable": bool(n_rel), "n": int(len(part)), "se": se,
                          "thickness": max(rms, 1e-4), "rgb": np.median(samples["rgb"][idx][part], 0),
                          "members": idx[part]})
    return cells


def _candidate_pairs(cells):
    """All cell pairs in the same or 26-adjacent voxels (i < j), vectorised over the 27 offsets."""
    V = np.array([c["vox"] for c in cells], np.int64)
    Vs = V - V.min(0) + 1
    dims = Vs.max(0) + 2
    key = (Vs[:, 0] * dims[1] + Vs[:, 1]) * dims[2] + Vs[:, 2]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    I, J = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                k2 = ((Vs[:, 0] + dx) * dims[1] + (Vs[:, 1] + dy)) * dims[2] + (Vs[:, 2] + dz)
                lo = np.searchsorted(ks, k2, "left"); hi = np.searchsorted(ks, k2, "right")
                cnt = hi - lo
                if not cnt.any():
                    continue
                i = np.repeat(np.arange(len(cells)), cnt)
                j = order[np.concatenate([np.arange(a, b) for a, b in zip(lo, hi) if b > a])]
                m = i < j
                I.append(i[m]); J.append(j[m])
    if not I:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(I), np.concatenate(J)


def link_cells(cells, cfg: Surface4B, report=None):
    """Co-layer edges and corner edges between cells. sigma_layer < 0 (AUTO): the cross-view discrepancy is estimated
    from the data as the robust spread of plane-offset differences between reliable, same-oriented, adjacent cells of
    DIFFERENT views, minus their own standard errors (floor: sigma_layer_floor); recorded in `report`."""
    if not cells:
        return np.zeros((0, 2), int), np.zeros((0, 2), int)
    i, j = _candidate_pairs(cells)
    mu = np.array([c["mu"] for c in cells]); nr = np.array([c["normal"] for c in cells])
    se = np.array([c["se"] for c in cells])
    nerr = np.array([c["normal_err"] for c in cells])
    nrel = np.array([c["normal_reliable"] and c["n"] >= cfg.min_cell_link for c in cells])
    view = np.array([c["view"] for c in cells]); frag = np.array([c["frag"] for c in cells])
    if not len(i):
        return np.zeros((0, 2), int), np.zeros((0, 2), int)
    same_view = view[i] == view[j]
    ok_frag = ~same_view | (frag[i] == frag[j])
    d = mu[j] - mu[i]
    cosn = (nr[i] * nr[j]).sum(1)
    dij = np.abs((nr[i] * d).sum(1)); dji = np.abs((nr[j] * d).sum(1))
    tang = np.linalg.norm(d - (nr[i] * d).sum(1)[:, None] * nr[i], axis=1)
    both_rel = nrel[i] & nrel[j]
    s_layer = cfg.sigma_layer
    est = None
    if s_layer < 0:
        m = both_rel & ~same_view & (cosn >= cfg.normal_cos)
        if m.sum() >= 50:
            off = (nr[i][m] * d[m]).sum(1)
            sd = 1.4826 * np.median(np.abs(off - np.median(off)))
            var = sd ** 2 - np.median(se[i][m] ** 2 + se[j][m] ** 2)
            est = float(np.sqrt(max(var, 0.0) / 2.0))
        s_layer = max(cfg.sigma_layer_floor, est or 0.0)
    if report is not None:
        report["sigma_layer_m"] = s_layer
        report["sigma_layer_source"] = "declared" if cfg.sigma_layer >= 0 else \
            f"AUTO: max(floor {cfg.sigma_layer_floor}, cross-view offset spread estimate {est})"
    tol = cfg.k_sigma * (np.sqrt(se[i] ** 2 + se[j] ** 2 + 2 * s_layer ** 2) +
                         tang * np.maximum(np.maximum(nerr[i], nerr[j]), cfg.normal_err / cfg.k_sigma))
    c_ok = ok_frag & both_rel & (cosn >= cfg.normal_cos) & (np.maximum(dij, dji) <= tol)
    reach = cfg.corner_reach_voxels * cfg.voxel
    k_ok = ok_frag & both_rel & (cosn < cfg.normal_cos) & (cosn > -0.5) & (dij <= reach) & (dji <= reach)
    co = np.unique(np.stack([i[c_ok], j[c_ok]], 1), axis=0) if c_ok.any() else np.zeros((0, 2), int)
    corner = np.unique(np.stack([i[k_ok], j[k_ok]], 1), axis=0) if k_ok.any() else np.zeros((0, 2), int)
    return co, corner


def components(n, edges):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if n == 0:
        return np.zeros(0, int)
    g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)) if len(edges) else \
        coo_matrix((n, n))
    return connected_components(g, directed=False)[1]


def merge_voxels(cells_in_patch, cfg: Surface4B):
    """One Gaussian cell per voxel of a patch (cells of different views at the same voxel fused by sample count)."""
    by = {}
    for c in cells_in_patch:
        by.setdefault(c["vox"], []).append(c)
    out = []
    for v, cs in by.items():
        w = np.array([c["n"] for c in cs], float)
        mu = (np.array([c["mu"] for c in cs]) * w[:, None]).sum(0) / w.sum()
        nr = (np.array([c["normal"] for c in cs]) * w[:, None]).sum(0)
        nr /= max(np.linalg.norm(nr), 1e-12)
        mem = np.concatenate([c["members"] for c in cs])
        thick = float(np.sqrt((np.array([c["thickness"] for c in cs]) ** 2 * w).sum() / w.sum()))
        out.append({"vox": v, "mu": mu, "normal": nr, "rgb": np.median(np.array([c["rgb"] for c in cs]), 0),
                    "thickness": thick, "count": int(w.sum()), "members": mem, "split": any(c["split"] for c in cs)})
    return out


def patches_from(samples, frag, idx, hyp, cfg: Surface4B, report=None):
    cells = build_cells(samples, frag, hyp, idx, cfg)
    co, corner = link_cells(cells, cfg, report)
    lab = components(len(cells), co)
    return cells, lab, co, corner


# =========================================================================== candidates
def propose(samples, frag, trusted_xyz, trusted_rot, trusted_scale, trusted_uid, cfg: Surface4B, log=print, cycle=0):
    """Moving-surface candidate patches near the trusted frontier (+ source-equivalent static competitors)."""
    from .articulated_scene import quat_to_rotmat  # noqa: F401
    idx_all = np.arange(len(samples["view"]))
    lrep = {}
    cells, lab, co, corner = patches_from(samples, frag, idx_all, "M", cfg, lrep)
    out, report = [], {**lrep, "cells": len(cells), "co_layer_edges": int(len(co)), "corner_edges": int(len(corner)),
                       "patches": int(len(np.unique(lab))) if len(lab) else 0, "too_small": 0, "not_frontier": 0,
                       "disconnected_mode": 0, "split_cells": int(sum(c["split"] for c in cells))}
    if not cells:
        return out, report, cells, lab, co, corner
    reach = cfg.frontier_reach_voxels * cfg.voxel
    mu_c = np.array([c["mu"] for c in cells])
    near = on_support(mu_c, np.repeat((reach ** 2 * np.eye(3))[None], len(mu_c), 0), trusted_xyz, trusted_rot,
                      trusted_scale, cfg.support_knn, cfg.support_k) if len(trusted_xyz) else np.zeros(len(mu_c), bool)
    nearest = None
    if len(trusted_xyz):
        from scipy.spatial import cKDTree
        nearest = cKDTree(trusted_xyz.detach().cpu().numpy())
    for l in np.unique(lab):
        ci = np.nonzero(lab == l)[0]
        mem = np.unique(np.concatenate([cells[i]["members"] for i in ci]))
        groups = np.unique(samples["group"][mem])
        if len(mem) < cfg.min_samples:
            report["too_small"] += 1
            continue
        frontier = bool(near[ci].any())
        mode = "frontier"
        if not frontier:
            if cfg.disconnected_search and len(groups) >= cfg.min_support_groups + 1:
                mode = "disconnected"
                report["disconnected_mode"] += 1
            else:
                report["not_frontier"] += 1
                continue
        mcells = merge_voxels([cells[i] for i in ci], cfg)
        # source-equivalent STATIC competitor: the same samples, layer-aware cells in world coordinates
        s_cells, s_lab, _, _ = patches_from(samples, frag, mem, "S", cfg)
        scells = []
        for sl in (np.unique(s_lab) if len(s_lab) else []):
            sci = [s_cells[i] for i in np.nonzero(s_lab == sl)[0]]
            if sum(c["n"] for c in sci) >= cfg.min_samples:
                scells += merge_voxels(sci, cfg)
        disp = np.linalg.norm(samples["xM"][mem] - samples["yW"][mem], axis=1)
        parents = []
        if nearest is not None:
            _, j = nearest.query(np.array([c["mu"] for c in mcells]))
            parents = np.unique(trusted_uid[np.asarray(j)]).tolist()
        frames = np.unique(samples["frame"][mem])
        out.append({"cycle": cycle, "search_mode": mode, "layer_uid": f"c{cycle}_p{int(l)}", "M_cells": mcells,
                    "S_cells": scells, "sources": mem, "support_groups": int(len(groups)),
                    "views": int(len(np.unique(samples["view"][mem]))), "frames": frames.tolist(),
                    "first_observed_frame": int(samples["frame"][mem][np.argmin(samples["row"][mem])]),
                    "frontier_parent_uids": parents[:64], "n_frontier_parents": len(parents),
                    "frontier_cells": int(near[ci].sum()),
                    "corner_links_to_other_patches": int((np.isin(corner, ci).sum(1) == 1).sum()) if len(corner) else 0,
                    "split_cell_frac": float(np.mean([c["split"] for c in mcells])),
                    "source_motion_p90_m": float(np.percentile(disp, 90)) if len(disp) else 0.0,
                    "evidence": "MULTI_GROUP" if len(groups) >= cfg.min_support_groups else "SINGLE_GROUP",
                    "source_view": samples["view"][mem], "source_pix": samples["pix"][mem],
                    "source_depth": samples["D"][mem]})
    out.sort(key=lambda c: (-len(c["sources"]), c["layer_uid"]))
    for k, c in enumerate(out):
        c["patch_uid"] = f"c{cycle}_{k:05d}"
    return out, report, cells, lab, co, corner


def frontier_tensors(scene, trusted: torch.Tensor):
    t = scene.tensors()
    ix = trusted.nonzero()[:, 0]
    uid = scene.g.lineage["gaussian_uid"][ix.cpu().numpy()]
    return t["xyz"][ix], t["rot"][ix], t["scale"][ix], uid


def view_accessors(pv, scene, noise: SensorNoise):
    """Closures over the prepared views (CPU numpy) for the fragment builder."""
    def depth(vi):
        d = pv.depth[vi]
        return np.asarray(d.numpy() if torch.is_tensor(d) else d, np.float32)

    def valid(vi):
        v = pv.valid[vi]
        return np.asarray(v.numpy() if torch.is_tensor(v) else v, bool)

    def sigma(vi):
        d = torch.as_tensor(depth(vi))
        return noise.sigma(scene.views[vi].cam.name, d).cpu().numpy().astype(float)
    return depth, valid, sigma, (lambda vi: scene.views[vi].cam.H), (lambda vi: scene.views[vi].cam.W)


def run_proposals(scene, pv, labels, trusted, undecided, cfg: Surface4B, noise: SensorNoise, roles, log=print, cycle=0):
    """Scan + layers + frontier candidates for one scene version (no mutation of anything)."""
    if noise.sigma_model < 0:
        noise.sigma_estimate = estimate_sigma_model(scene, pv, labels, noise, cfg)
        log(f"  4B model discrepancy: {noise.sigma_estimate}")
    codes, stats, samples, cnt = scan(scene, pv, labels, trusted, undecided, cfg, noise, roles, log=log)
    acc = view_accessors(pv, scene, noise)
    frag = fragments(samples, *acc, cfg) if len(samples["view"]) else np.zeros(0, np.int64)
    txyz, trot, tsc, tuid = frontier_tensors(scene, trusted)
    cands, rep, cells, lab, co, corner = propose(samples, frag, txyz, trot, tsc, tuid, cfg, log=log, cycle=cycle)
    log(f"  4B proposals (cycle {cycle}): {len(cands)} candidates; {rep}")
    return {"codes": codes, "stats": stats, "samples": samples, "counters": cnt, "fragments": frag,
            "candidates": cands, "report": rep, "cells": cells, "cell_patch": lab, "co_edges": co,
            "corner_edges": corner}
