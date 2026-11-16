"""Phase 4B, step 2: masked RGB-D back-projection, inverse joint transport and layer-aware fusion (spec 5.2-5.5).

For an eligible pixel p = (u + 0.5, v + 0.5) with metric axial depth d (views.py pixel convention):

    r = K^-1 [u + 0.5, v + 0.5, 1]^T,   y_W = R_WC (d r) + t_WC
    x_S = y_W,                          x_M = Q_f^T (y_W - p_f)          (same anchor, same J_3, q_3)

Measurement covariance (not the render covariance):  P_y = R_WC J_back Sigma_uvd J_back^T R_WC^T,
P_xM = Q_f^T P_y Q_f, with J_back = [[d/fx, 0, (u+.5-cx)/fx], [0, d/fy, (v+.5-cy)/fy], [0, 0, 1]] and
Sigma_uvd = diag(sigma_px^2, sigma_px^2, sigma_D(d)^2). Repeated observations are NOT treated as independent
for shared calibration / joint error: a cell reports the mean P of its observations, never P / n.

Association (layer-aware, not nearest neighbour): observations are binned per hypothesis by
(canonical voxel, normal bin). The normal comes from the observed depth map (cross product of the
back-projected neighbours), oriented towards the camera, and is transported with the hypothesis
(Q_f^T n for M). Its bin is the dominant signed canonical axis, so the two faces of a thin panel (opposite
normals) never fuse, and a front and a back layer in the same voxel stay separate. A cell is a proposal only
if >= min_obs observations from >= min_views distinct (camera, frame) support it; cells sharing a normal bin
and touching (26-neighbourhood) form a patch, fitted by a robust weighted plane (mean, normal, two tangent
extents from the sampling pitch, thickness from the normal residual).

No physical birth time (5.5): a moving cell is stored in the canonical frame of the anchor and rendered at
every frame by (Q_f, p_f); the first frame that saw it is provenance only.
Ranking only (5.4.5): temporal coherence = distinct frames per cell under each hypothesis. Both hypotheses
are kept for the 4C full-render test; they never enter one physical scene together.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .residual_discovery import BIRTH_SOURCES


@dataclass
class FusionConfig:
    voxel: float = 0.01               # m, canonical cell size
    min_obs: int = 3
    min_views: int = 2
    sigma_px: float = 0.5             # declared pixel uncertainty (px)
    sigma_depth: float = 0.005        # m, used when the sensor model gives none
    max_pixels_per_view: int = 20000  # deterministic stride subsampling of eligible pixels, logged
    explained_mahal: float = 3.0      # a back-projected point within this Mahalanobis distance of an existing
                                      # posed Gaussian (covariance R S^2 R^T + sigma_depth^2 I) is existing
                                      # geometry whose EXPECTED depth is mixed with what lies behind it
                                      # (D_hat is not a first return), not a new surface: dropped and counted.
                                      # Anisotropic on purpose: a parallel layer 12 mm behind a flat splat
                                      # is not "on" it.
    explained_knn: int = 8

    def to_dict(self):
        return asdict(self)


AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)


def backproject(cam, depth: np.ndarray, sel: np.ndarray):
    """World points [K,3], camera-space normals [K,3] (oriented to the camera; NaN when undefined) of the
    selected pixels (bool [H,W])."""
    K = cam.K
    c2w = cam.c2w
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    X = (uu + 0.5 - K[0, 2]) / K[0, 0] * depth
    Y = (vv + 0.5 - K[1, 2]) / K[1, 1] * depth
    P = np.stack([X, Y, depth], -1)
    du = np.zeros_like(P); dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(du, dv)
    ok = np.zeros((H, W), bool)
    ok[1:-1, 1:-1] = (depth[1:-1, 2:] > 0) & (depth[1:-1, :-2] > 0) & (depth[2:, 1:-1] > 0) & (depth[:-2, 1:-1] > 0)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(ok[..., None] & (nn > 0), n / np.maximum(nn, 1e-12), np.nan)
    flip = (n * P).sum(-1) > 0                      # orient towards the camera (n . (0 - P) > 0)
    n[flip] *= -1
    pc, nc = P[sel], n[sel]
    yw = pc @ c2w[:3, :3].T + c2w[:3, 3]
    nw = nc @ c2w[:3, :3].T
    return yw, nw, pc


def point_covariance(cam, pc: np.ndarray, sigma_px: float, sigma_d: np.ndarray):
    """P_y per point (world), [K,3,3]."""
    K = cam.K
    d = pc[:, 2]
    u = pc[:, 0] / d * K[0, 0]                      # = u + 0.5 - cx
    v = pc[:, 1] / d * K[1, 1]
    J = np.zeros((len(pc), 3, 3))
    J[:, 0, 0] = d / K[0, 0]; J[:, 0, 2] = u / K[0, 0]
    J[:, 1, 1] = d / K[1, 1]; J[:, 1, 2] = v / K[1, 1]
    J[:, 2, 2] = 1.0
    S = np.zeros((len(pc), 3, 3))
    S[:, 0, 0] = S[:, 1, 1] = sigma_px ** 2
    S[:, 2, 2] = sigma_d ** 2
    # camera axes x, y, z = (u, v, d) mapped: p_cam = [u d / fx, v d / fy, d] -> J_back acts on (du, dv, dd)
    Pc = J @ S @ J.transpose(0, 2, 1)
    R = cam.c2w[:3, :3]
    return R @ Pc @ R.T


@dataclass
class Observation:
    x: np.ndarray        # canonical position under the hypothesis [K,3]
    n: np.ndarray        # canonical normal [K,3]
    rgb: np.ndarray      # [K,3]
    P: np.ndarray        # canonical measurement covariance [K,3,3]
    view: np.ndarray     # view index [K]
    frame: np.ndarray    # original frame [K]
    camera: list
    src: np.ndarray      # source pixel id view * H * W + pixel [K]


def collect(scene, maps, cfg: FusionConfig, sigma_depth_fn=None, moving=None):
    """Eligible pixels (COVERAGE_GAP | FRONT_SURFACE, never MIXED_EDGE / BEHIND / invalid depth) of every view,
    back-projected and inverse-transported under S and M. With `moving` (the current labels), points that
    fall on existing posed geometry are dropped (FusionConfig.explained_factor)."""
    from scipy.spatial import cKDTree
    from .articulated_scene import posed
    out = {"S": [], "M": []}
    log = []
    t = scene.tensors() if moving is not None else None
    for vi, (vr, code) in enumerate(zip(scene.views, maps)):
        sel = np.isin(code.numpy(), BIRTH_SOURCES)
        n_el = int(sel.sum())
        if n_el == 0:
            continue
        if n_el > cfg.max_pixels_per_view:
            idx = np.flatnonzero(sel)
            keep = idx[np.linspace(0, len(idx) - 1, cfg.max_pixels_per_view).round().astype(int)]
            sel = np.zeros_like(sel); sel.reshape(-1)[keep] = True
            log.append({"view": vi, "eligible": n_el, "kept": cfg.max_pixels_per_view})
        v = vr.get()
        depth = v.depth.numpy()
        if t is not None:
            from .articulated_scene import quat_to_rotmat
            mu, qq = posed(t["xyz"], t["rot"], moving, scene.transport.at(vr.frame))
            yw_all, _, _ = backproject(vr.cam, depth, sel)
            k = min(cfg.explained_knn, len(mu))
            _, j = cKDTree(mu.cpu().numpy()).query(yw_all, k=k)
            j = torch.as_tensor(j.reshape(len(yw_all), k), device=mu.device)
            R = quat_to_rotmat(qq[j])                                    # [K,k,3,3]
            S2 = torch.diag_embed(t["scale"][j] ** 2)
            Sig = R @ S2 @ R.transpose(-1, -2) + (cfg.sigma_depth ** 2) * torch.eye(3, device=mu.device)
            dv = torch.as_tensor(yw_all, device=mu.device, dtype=mu.dtype)[:, None, :] - mu[j]
            m2 = (dv[..., None, :] @ torch.linalg.solve(Sig, dv[..., None]))[..., 0, 0]
            expl = (m2.min(1).values <= cfg.explained_mahal ** 2).cpu().numpy()
            if expl.any():
                idx = np.flatnonzero(sel)
                sel = sel.copy(); sel.reshape(-1)[idx[expl]] = False
            log.append({"view": vi, "eligible": n_el, "explained_geometry": int(expl.sum())})
            if not sel.any():
                continue
        yw, nw, pc = backproject(vr.cam, depth, sel)
        rgb = v.rgb.permute(1, 2, 0).numpy()[sel]
        sd = sigma_depth_fn(pc[:, 2]) if sigma_depth_fn else np.full(len(pc), cfg.sigma_depth)
        Py = point_covariance(vr.cam, pc, cfg.sigma_px, sd)
        Q, p = scene.joint.transform(vr.frame)
        for h in ("S", "M"):
            if h == "S":
                x, n, P = yw, nw, Py
            else:
                x, n, P = (yw - p) @ Q, nw @ Q, Q.T @ Py @ Q
            src = vi * depth.size + np.flatnonzero(sel)
            out[h].append(Observation(x, n, rgb, P, np.full(len(x), vi), np.full(len(x), vr.frame), [vr.cam.name], src))
    return out, log


def fuse(obs_list, cfg: FusionConfig):
    """Cells keyed by (voxel, normal bin). Returns a list of cell dicts and the patch label per cell."""
    if not obs_list:
        return [], np.zeros(0, int)
    x = np.concatenate([o.x for o in obs_list]); n = np.concatenate([o.n for o in obs_list])
    rgb = np.concatenate([o.rgb for o in obs_list]); P = np.concatenate([o.P for o in obs_list])
    view = np.concatenate([o.view for o in obs_list]); frame = np.concatenate([o.frame for o in obs_list])
    src = np.concatenate([o.src for o in obs_list])
    good = np.isfinite(n).all(1)
    x, n, rgb, P, view, frame, src = x[good], n[good], rgb[good], P[good], view[good], frame[good], src[good]
    if not len(x):
        return [], np.zeros(0, int)
    nbin = np.argmax(n @ AXES.T, 1)
    key = np.concatenate([np.floor(x / cfg.voxel).astype(np.int64), nbin[:, None]], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    cells = []
    for c in range(len(uniq)):
        m = inv == c
        nv = len(np.unique(view[m]))
        if m.sum() < cfg.min_obs or nv < cfg.min_views:
            continue
        xc = x[m]
        mu = np.median(xc, 0)
        nm = n[m].mean(0); nm /= max(np.linalg.norm(nm), 1e-12)
        r = (xc - mu) @ nm
        cells.append({"key": tuple(uniq[c]), "mu": mu, "normal": nm, "rgb": np.median(rgb[m], 0),
                      "count": int(m.sum()), "views": nv, "frames": int(len(np.unique(frame[m]))),
                      "first_frame": int(frame[m].min()), "first_view": int(view[m].min()), "src": src[m],
                      "thickness": float(np.sqrt(np.mean(r ** 2))), "P_mean": P[m].mean(0)})
    # patches: touching cells in the same normal bin
    if not cells:
        return cells, np.zeros(0, int)
    index = {c["key"]: i for i, c in enumerate(cells)}
    parent = list(range(len(cells)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, c in enumerate(cells):
        kx, ky, kz, kb = c["key"]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    j = index.get((kx + dx, ky + dy, kz + dz, kb))
                    if j is not None:
                        a, b = find(i), find(j)
                        if a != b:
                            parent[max(a, b)] = min(a, b)
    lab = np.array([find(i) for i in range(len(cells))])
    return cells, np.unique(lab, return_inverse=True)[1]


def coherence(cells) -> float:
    """Ranking only: mean distinct frames per proposal cell (a surface placed by the right hypothesis
    re-observes the same canonical cell across frames; the wrong one smears it)."""
    return float(np.mean([c["frames"] for c in cells])) if cells else 0.0


def cells_to_gaussian_params(cells, voxel, opacity0=0.5, normal_floor=0.001, tangent_factor=0.6):
    """One flat Gaussian per cell: centre, tangent basis from the normal, tangential scale from the cell
    pitch, normal thickness from the fit, conservative opacity, DC from the observed colour."""
    from .articulated_scene import rotmat_to_quat
    xyz, quat, scale, dc = [], [], [], []
    for c in cells:
        nrm = c["normal"]
        t1 = np.cross(nrm, [1.0, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1.0, 0]); t1 /= np.linalg.norm(t1)
        t2 = np.cross(nrm, t1)
        quat.append(rotmat_to_quat(np.stack([t1, t2, nrm], 1)))
        xyz.append(c["mu"])
        scale.append([tangent_factor * voxel, tangent_factor * voxel, max(normal_floor, c["thickness"])])
        dc.append((np.clip(c["rgb"], 0.02, 0.98) - 0.5) / 0.28209479177387814)
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32)
    n = len(cells)
    return dict(xyz=t(xyz).reshape(n, 3), rot=t(quat).reshape(n, 4), log_scale=torch.log(t(scale).reshape(n, 3)),
                opacity_logit=torch.full((n, 1), float(np.log(opacity0 / (1 - opacity0)))), dc=t(dc).reshape(n, 1, 3))
