"""Surface-centred endpoint bootstrap: one synchronized multiview RGB-D frame ->
a compact, fixed-count 3DGS population with provenance and a geometry covariance.

RigEuler3DGS_Calibrated_Bootstrap_IEKF_Implementation_Spec.md section 3.

Only the endpoint frame's own observations are used (every camera, strictly
masked, raw finite depth > 0); nothing from another time / articulation state.

Samples. Pixel (u, v) of camera c with depth z gives

    y = R_wc (z r) + t_wc,   r = K^-1 [u, v, 1]^T,
    C_sensor = sigma_D,c^2(z) d d^T,   d = R_wc r            (rank one, along the ray)

Seeds. A voxel grid sized to the requested population (density from the visible
surface, not from any label), then for each seed a robust point-to-tangent-plane
fit over its k nearest samples from ALL cameras (one common surface, never
per-camera fused clouds):

    r_j   = n^T (y_j - mu)
    var_j = n^T C_j n + sigma_surface^2 = sigma_D^2 (n . d_j)^2 + sigma_surface^2
    w_j   = Student-t weight(r_j / sqrt(var_j)) / var_j            (IRLS)

mu is the weighted centroid, n the smallest weighted principal direction,
sigma_surface^2 the excess normal spread beyond the sensor term (a GEOMETRIC
approximation, never sensor noise), capped by the tangential footprint so the
covariance cannot inflate without bound. Samples from another layer (back side,
occluder) get ~zero weight instead of being averaged in; no pixel-space
interpolation is ever done, so hard mask edges are not smeared.

Gaussian: mu, R = [t1 t2 n], tangential scales from the local seed spacing
(measured coverage), normal scale from the fitted thickness, colour from the
weighted samples, opacity from the inlier fraction. Sigma_render = R diag(s^2) R^T
is the RENDERING footprint, not P_mu (the uncertainty of the centre), which is
exported separately (normal direction: 1 / sum_j 1/var_j; tangential: unobservable
from the samples alone -> declared prior).
"""
from __future__ import annotations

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# cameras and back-projection
# ---------------------------------------------------------------------------
def camera_geometry(cam) -> dict:
    """Intrinsics / extrinsics of a scene Camera in the OpenCV convention."""
    W, H = int(cam.image_width), int(cam.image_height)
    fx = W / (2.0 * math.tan(float(cam.FoVx) / 2.0))
    fy = H / (2.0 * math.tan(float(cam.FoVy) / 2.0))
    w2c = cam.world_view_transform.detach().double().cpu().T      # row-major glm -> 4x4
    c2w = torch.linalg.inv(w2c)
    return {"W": W, "H": H, "fx": fx, "fy": fy, "cx": W / 2.0, "cy": H / 2.0,
            "R_wc": c2w[:3, :3].numpy(), "t_wc": c2w[:3, 3].numpy(), "w2c": w2c.numpy()}


def backproject(depth: np.ndarray, valid: np.ndarray, geo: dict, stride: int = 1):
    """World points y [M,3], unit-depth ray directions d [M,3] (world), depth z [M],
    flat pixel ids [M]."""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    sel = valid[vv, uu]
    u, v = uu[sel].astype(np.float64), vv[sel].astype(np.float64)
    z = depth[vv, uu][sel].astype(np.float64)
    r = np.stack([(u + 0.5 - geo["cx"]) / geo["fx"], (v + 0.5 - geo["cy"]) / geo["fy"], np.ones_like(u)], 1)
    d = r @ geo["R_wc"].T
    y = z[:, None] * d + geo["t_wc"][None]
    return y, d, z, (vv[sel] * W + uu[sel]).astype(np.int64)


# ---------------------------------------------------------------------------
# seeds and the robust surface fit
# ---------------------------------------------------------------------------
def voxel_seeds(y: np.ndarray, target: int, iters: int = 30) -> np.ndarray:
    """~target seed positions: per-voxel means on a grid whose size is searched so the
    number of occupied voxels approaches `target` (density from coverage)."""
    lo, hi = y.min(0), y.max(0)
    a, b = 1e-5, float(np.max(hi - lo))
    best = None
    for _ in range(iters):
        h = math.sqrt(a * b)
        key = np.floor((y - lo) / h).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        n = cnt.size
        best = (h, inv.reshape(-1), cnt)
        if abs(n - target) <= 0.02 * target:
            break
        if n > target:
            a = h
        else:
            b = h
    h, inv, cnt = best
    seeds = np.zeros((cnt.size, 3))
    np.add.at(seeds, inv, y)
    return seeds / cnt[:, None], h


def student_t_weight(z2: torch.Tensor, nu: float = 5.0) -> torch.Tensor:
    return (nu + 1.0) / (nu + z2)


@torch.no_grad()
def fit_surface(y: np.ndarray, d: np.ndarray, var_sensor: np.ndarray, rgb: np.ndarray,
                seeds: np.ndarray, *, k: int = 24, iters: int = 4, nu: float = 5.0,
                device: str = "cuda", chunk: int = 20000) -> dict:
    """Robust point-to-tangent-plane fit around every seed (see module docstring)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(y)
    _, nn = tree.query(seeds, k=min(k, len(y)))
    sp = cKDTree(seeds).query(seeds, k=min(7, len(seeds)))[0][:, 1:].mean(1)   # local seed spacing
    out = {key: [] for key in ("mu", "normal", "tangent1", "tangent2", "eig", "sigma_surface2",
                               "var_normal", "inlier_fraction", "color", "num_samples")}
    for s in range(0, len(seeds), chunk):
        idx = torch.as_tensor(nn[s:s + chunk], device=device)
        Y = torch.as_tensor(y, device=device, dtype=torch.float64)[idx]                  # [n,k,3]
        D = torch.as_tensor(d, device=device, dtype=torch.float64)[idx]
        S2 = torch.as_tensor(var_sensor, device=device, dtype=torch.float64)[idx]       # [n,k]
        C = torch.as_tensor(rgb, device=device, dtype=torch.float64)[idx]               # [n,k,3]
        spacing = torch.as_tensor(sp[s:s + chunk], device=device, dtype=torch.float64)
        w = torch.ones(Y.shape[:2], device=device, dtype=torch.float64)
        surf2 = torch.zeros(Y.shape[0], device=device, dtype=torch.float64)
        for _ in range(iters):
            ws = w / w.sum(1, keepdim=True)
            mu = (ws[..., None] * Y).sum(1)
            X = Y - mu[:, None]
            M = torch.einsum("nk,nki,nkj->nij", ws, X, X)
            # 3x3 blocks: on the CPU (the batched CUDA solver asks for a multi-GiB workspace)
            ev, evec = (t.to(device) for t in torch.linalg.eigh(M.cpu()))               # ascending
            n = evec[:, :, 0]
            r = (X * n[:, None]).sum(-1)
            sens = S2 * (D * n[:, None]).sum(-1) ** 2
            var = sens + surf2[:, None]
            w = student_t_weight(r * r / var.clamp_min(1e-18), nu) / var.clamp_min(1e-18)
            wn = w / w.sum(1, keepdim=True)
            excess = (wn * (r * r - sens)).sum(1).clamp_min(0.0)
            surf2 = torch.minimum(excess, (0.5 * spacing) ** 2)                         # shape constraint
        z2 = r * r / var.clamp_min(1e-18)
        inl = (z2 < 9.0).double().mean(1)
        out["mu"].append(mu)
        out["normal"].append(n)
        out["tangent1"].append(evec[:, :, 2])
        out["tangent2"].append(evec[:, :, 1])
        out["eig"].append(ev)
        out["sigma_surface2"].append(surf2)
        out["var_normal"].append(1.0 / (1.0 / var.clamp_min(1e-18)).sum(1))
        out["inlier_fraction"].append(inl)
        out["color"].append((wn[..., None] * C).sum(1))
        out["num_samples"].append(torch.full_like(inl, idx.shape[1]))
    res = {k_: torch.cat(v).cpu().numpy() for k_, v in out.items()}
    res["seed_spacing"] = sp
    return res


def gaussian_parameters(fit: dict, min_thickness: float = 1e-4) -> dict:
    """mu, rotation (wxyz quaternion), scales, opacity, DC colour of every seed."""
    t1, t2, n = fit["tangent1"], fit["tangent2"], fit["normal"]
    Rm = np.stack([t1, t2, n], axis=2)                                    # columns
    det = np.linalg.det(Rm)
    Rm[det < 0, :, 2] *= -1.0
    h = fit["seed_spacing"]
    st1 = np.clip(np.sqrt(np.maximum(fit["eig"][:, 2], 0)), 0.3 * h, 1.0 * h)
    st2 = np.clip(np.sqrt(np.maximum(fit["eig"][:, 1], 0)), 0.3 * h, 1.0 * h)
    sn = np.clip(np.sqrt(fit["sigma_surface2"] + fit["var_normal"]), min_thickness, 0.5 * np.minimum(st1, st2))
    opacity = np.clip(fit["inlier_fraction"], 0.05, 0.95)
    return {"xyz": fit["mu"], "R": Rm, "quat_wxyz": rotmat_to_quat_wxyz(Rm),
            "scales": np.stack([st1, st2, sn], 1), "opacity": opacity,
            "rgb": np.clip(fit["color"], 0.0, 1.0)}


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Batched rotation matrices -> unit quaternions (w, x, y, z)."""
    R = np.asarray(R, float)
    q = np.empty((R.shape[0], 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for i in range(R.shape[0]):
        m = R[i]
        if tr[i] > 0:
            s = math.sqrt(tr[i] + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# provenance and initial covariance at the endpoint
# ---------------------------------------------------------------------------
@torch.no_grad()
def source_responsibility(lin) -> dict:
    """Exact compositing weights w = T_before * alpha of every (Gaussian, camera, pixel)
    from the production renderer's own contributor walk (output-only extraction),
    verified: sum_i w_i(p) == rendered accumulated alpha A(p)."""
    from observation_diagnostics.renderer_access import pixel_contributions
    g, c, p, w, err = [], [], [], [], []
    for ci, v in enumerate(lin.views):
        cn = pixel_contributions(v.cache)
        pix = torch.repeat_interleave(torch.arange(cn.H * cn.W, device=cn.gid.device), cn.counts.long())
        wt = cn.T * cn.alpha
        acc = torch.zeros(cn.H * cn.W, device=wt.device, dtype=torch.float64).index_add_(0, pix, wt.double())
        err.append(float((acc - v.alpha.reshape(-1).double()).abs().max()))
        keep = wt > 0
        g.append(cn.gid[keep].int().cpu())
        c.append(torch.full((int(keep.sum()),), ci, dtype=torch.int16))
        p.append(pix[keep].int().cpu())
        w.append(wt[keep].float().cpu())
    return {"gaussian_id": torch.cat(g).numpy(), "camera_id": torch.cat(c).numpy(),
            "flat_pixel_id": torch.cat(p).numpy(), "weight": torch.cat(w).numpy(),
            "max_abs_alpha_sum_error_per_camera": err}


def ownership(resp: dict, n: int, ncam: int) -> dict:
    """Per-Gaussian camera shares of responsibility (a DIAGNOSTIC, never a gate)."""
    share = np.zeros((n, ncam))
    np.add.at(share, (resp["gaussian_id"], resp["camera_id"].astype(np.int64)), resp["weight"])
    tot = share.sum(1, keepdims=True)
    frac = np.divide(share, tot, out=np.zeros_like(share), where=tot > 0)
    return {"camera_share": frac, "total_weight": tot[:, 0],
            "dominant_camera": np.where(tot[:, 0] > 0, frac.argmax(1), -1),
            "unseen": int((tot[:, 0] == 0).sum())}


def geometry_covariance(fisher: torch.Tensor, prior_sigma_pos: float, prior_sigma_rot: float) -> dict:
    """P0 = (F + Lambda_prior)^-1 per Gaussian (pose block: delta mu, delta phi).

    F is the renderer's per-Gaussian 6x6 Fisher at the endpoint under the shared
    sensor model; Lambda_prior a DECLARED weak prior so unobservable directions
    get a large (prior) variance instead of fake certainty. Per-Gaussian blocks
    ignore inter-Gaussian and shared-camera coupling (documented approximation)."""
    F = 0.5 * (fisher + fisher.transpose(1, 2)).double().cpu()
    lam = torch.diag(torch.tensor([prior_sigma_pos ** -2] * 3 + [prior_sigma_rot ** -2] * 3,
                                  dtype=torch.float64))
    A = F + lam
    P = torch.linalg.inv(A)
    P = 0.5 * (P + P.transpose(1, 2))
    evF = torch.linalg.eigvalsh(F)
    evP = torch.linalg.eigvalsh(P)
    return {"P0_geometry": P.numpy(), "fisher": F.numpy(), "fisher_eigenvalues": evF.numpy(),
            "P0_eigenvalues": evP.numpy(),
            "prior": {"sigma_pos_m": prior_sigma_pos, "sigma_rot_rad": prior_sigma_rot},
            "approximation": ("per-Gaussian 6x6 blocks of (F + Lambda_prior)^-1; inter-Gaussian and "
                              "shared-camera covariance are NOT represented")}


# ---------------------------------------------------------------------------
# bootstrap refinement objective (pure; CPU-testable)
# ---------------------------------------------------------------------------
def pixel_partition(height: int, width: int, heldout_fraction: float, seed: int, camera_index: int):
    """Deterministic, disjoint (train, held) split of the WHOLE image: drawn once per
    camera from (seed, camera_index) and never resampled. Returns two bool [H,W]."""
    rng = np.random.default_rng([int(seed), int(camera_index)])
    held = rng.random((height, width)) < float(heldout_fraction)
    return ~held, held


def bootstrap_loss_terms(rgb_render, alpha, invdepth_render, obs: dict, S, *, silhouette_sigma: float,
                         nu: float = 5.0) -> dict:
    """The bootstrap refinement terms on ONE pixel partition S ([H,W] bool).

    Supports (spec 4c88725 surgical review, section 1):
        alpha   all of S, foreground AND background:
                E_alpha = sum_{p in S} (A - M)^2 / (2 sigma_A^2 |S|)
                -- a soft loss on the hard binary object mask; opacity spilled onto
                M = 0 pixels is penalised (it was not, when S lay inside M)
        RGB     S AND M:  1/2 sum W_rgb |C - I|^2 / |S AND M|
        depth   S AND M AND observed-valid AND A > 0.5:
                Student-t NLL of inverse depth, Var(1/z) = Var(z)/z^4
    Background rays and invalid observations never enter RGB or depth. A term whose
    support is empty is exactly 0 and reported with count 0 (never NaN).

    obs: rgb [3,H,W], M [H,W] float in {0,1}, V [H,W] bool valid depth, invz, var_invz [H,W],
         w_rgb [3,1,1] fixed RGB precision."""
    S = S.bool()
    M = obs["M"]
    Mb = M > 0.5
    zero = alpha.sum() * 0.0
    n_a = int(S.sum())
    e_a = ((alpha - M) ** 2)[S].sum() / (2.0 * silhouette_sigma ** 2 * n_a) if n_a else zero
    Srgb = S & Mb
    n_r = int(Srgb.sum())
    e_r = (0.5 * (obs["w_rgb"] * (rgb_render - obs["rgb"]) ** 2).sum(0))[Srgb].sum() / n_r if n_r else zero
    inv = invdepth_render / alpha.clamp_min(1e-3)
    Sd = Srgb & obs["V"].bool() & (alpha.detach() > 0.5)
    n_d = int(Sd.sum())
    if n_d:
        z2 = (inv - obs["invz"]) ** 2 / obs["var_invz"]
        e_d = (0.5 * (nu + 1.0) * torch.log1p(z2 / nu))[Sd].sum() / n_d
    else:
        e_d = zero
    with torch.no_grad():
        bg = S & ~Mb
        stats = {"pixels": n_a, "foreground": int(Srgb.sum()), "background": int(bg.sum()), "depth": n_d,
                 "foreground_miss": float((M - alpha).clamp_min(0)[Srgb].mean()) if n_r else None,
                 "offmask_opacity": float(alpha[bg].mean()) if int(bg.sum()) else None}
    return {"rgb": e_r, "alpha": e_a, "depth": e_d, "stats": stats}
