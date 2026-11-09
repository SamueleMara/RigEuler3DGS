"""READ-ONLY wrong-camera force provenance for the Phase-1 IEKF.

CODEX_DEBUG_RigEuler3DGS_REAL_FRIDGE_WRONG_CAMERA_GAUSSIAN_DRIFT.md sections 3-5, 7.

Everything here works on ONE frozen inner iterate handed over by the IEKF's
`iteration_observer` hook (estimator/sparse_iekf.py): the same linearization
cache, the same per-camera CameraTerms (Student-t precision, calibrated
variances, masked RGB / alpha policy), the same prior information, damping and
canonical group map the production solve uses. Nothing is re-rendered, nothing
is written back.

Per camera c and channel ch in {rgb, depth, alpha} (sign: the production
H^T W r ascent convention):

    b[c,rgb]   = H_RGB,c^T (W_RGB,c r_RGB,c)
    b[c,depth] = H_D,c^T   (W_D,c   r_D,c)
    b[c,alpha] = H_A,c^T   (K_c     r_A,c)          (exact zero without the alpha term)
    A[c,ch](v) = H_ch,c^T (W_ch,c H_ch,c v)        matrix-free, keeps co-visibility coupling

with sum_{c,ch} b = production rhs_pose and sum_{c,ch} A = production meas_normal
asserted, not assumed. Leave-one-camera / leave-one-channel influences are exact
solutions of the frozen, UNCONSTRAINED quadratic in the production grouped
coordinates:

    A(v) = Lambda_prior v + sum A[c,ch](v) + lm v       (12D per Gaussian, then G^T . G)
    delta_all = A^-1 b,  delta_-k = (A - A_k)^-1 (b - b_k),  I_k = delta_all - delta_-k
    identity check: I_k == A^-1 [b_k - A_k(delta_-k)]

They are NOT the production trust-region step (a constrained, nonlinear step).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

CHANNELS = ("rgb", "depth", "alpha")


# ---------------------------------------------------------------------------
# per-view operators (the production kernels, one camera at a time)
# ---------------------------------------------------------------------------
def view_H(view, pose):
    from estimator.renderer_bridge import iekf_apply_H
    return iekf_apply_H(view.cache, pose.float().contiguous())


def view_HT(view, z_rgb, z_depth):
    from estimator.renderer_bridge import iekf_apply_HT
    return iekf_apply_HT(view.cache, z_rgb.contiguous(), z_depth.contiguous(), view.transmittance,
                         view.alpha, view.depth, view.n_contrib)


def _full_ray(view):
    H, W = view.alpha.shape
    return (torch.ones((H, W), dtype=torch.int32, device=view.alpha.device),
            torch.zeros((H, W), dtype=torch.float32, device=view.alpha.device))


def view_H_alpha(view, pose):
    from diff_gaussian_rasterization import iekf_vis_apply_H
    mode, zobs = _full_ray(view)
    return iekf_vis_apply_H(view.cache, mode, zobs, 0.0, pose.float().contiguous())


def view_HT_alpha(view, za):
    from diff_gaussian_rasterization import iekf_vis_apply_HT
    mode, zobs = _full_ray(view)
    return iekf_vis_apply_HT(view.cache, mode, zobs, 0.0, za.float().contiguous(), view.transmittance,
                             view.n_contrib)


def camera_channel_rhs(lin, terms) -> dict:
    """{(c, ch): [N,6] float64} from the frozen terms."""
    out = {}
    for c, (v, t) in enumerate(zip(lin.views, terms)):
        zr, zd = torch.zeros_like(t.r_rgb), torch.zeros_like(t.r_d)
        out[(c, "rgb")] = view_HT(v, t.p_rgb * t.r_rgb, zd).double()
        out[(c, "depth")] = view_HT(v, zr, t.p_d * t.r_d).double()
        out[(c, "alpha")] = (view_HT_alpha(v, t.kappa * t.r_a).double() if t.kappa is not None
                             else torch.zeros_like(out[(c, "rgb")]))
    return out


def camera_channel_normal(lin, terms, pose, keys=None) -> dict:
    """{(c, ch): [N,6] float64} = A[c,ch](pose), one H per camera, one H^T per (c, ch)."""
    out = {}
    for c, (v, t) in enumerate(zip(lin.views, terms)):
        want = [ch for ch in CHANNELS if keys is None or (c, ch) in keys]
        if not want:
            continue
        dC, dD = view_H(v, pose)
        zr, zd = torch.zeros_like(t.r_rgb), torch.zeros_like(t.r_d)
        if "rgb" in want:
            out[(c, "rgb")] = view_HT(v, t.p_rgb * dC, zd).double()
        if "depth" in want:
            out[(c, "depth")] = view_HT(v, zr, t.p_d * dD).double()
        if "alpha" in want:
            if t.kappa is not None:
                out[(c, "alpha")] = view_HT_alpha(v, t.kappa * view_H_alpha(v, pose)).double()
            else:
                out[(c, "alpha")] = torch.zeros((pose.shape[0], 6), dtype=torch.float64, device=pose.device)
    return out


def camera_channel_energy(terms) -> list:
    rows = []
    for c, t in enumerate(terms):
        for ch, (p, r) in {"rgb": (t.p_rgb, t.r_rgb), "depth": (t.p_d, t.r_d)}.items():
            m = p > 0
            rows.append({"camera_index": c, "channel": ch, "valid_precision_pixels": int(m.sum()),
                         "weighted_energy": 0.5 * float((p.double() * r.double().square()).sum()),
                         "objective_energy": t.e_rgb if ch == "rgb" else t.e_d,
                         "precision_median": float(p[m].median()) if bool(m.any()) else None})
        if t.kappa is not None:
            rows.append({"camera_index": c, "channel": "alpha", "valid_precision_pixels": int((t.kappa > 0).sum()),
                         "weighted_energy": t.e_a, "objective_energy": t.e_a,
                         "precision_median": float(t.kappa.median())})
        else:
            rows.append({"camera_index": c, "channel": "alpha", "valid_precision_pixels": 0,
                         "weighted_energy": 0.0, "objective_energy": 0.0, "precision_median": None})
    return rows


def additivity_and_symmetry(lin, terms, rhs_pose, meas_normal, n_probe=3, seed=0) -> dict:
    """sum b == production rhs; sum A == production normal; <v,A w> == <A v,w>; v^T A v >= 0."""
    b = camera_channel_rhs(lin, terms)
    tot = sum(b.values())
    ref = rhs_pose.double()
    rhs_rel = float((tot - ref).norm() / ref.norm().clamp_min(1e-30))
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = rhs_pose.shape[0]
    dev = rhs_pose.device
    norm_rel, sym, psd = [], {}, {}
    for k in range(n_probe):
        v = torch.randn((n, 6), generator=g).to(dev)
        w = torch.randn((n, 6), generator=g).to(dev)
        Av, Aw = camera_channel_normal(lin, terms, v), camera_channel_normal(lin, terms, w)
        prod = meas_normal(v.float()).double()
        norm_rel.append(float((sum(Av.values()) - prod).norm() / prod.norm().clamp_min(1e-30)))
        for key in Av:
            vAw, Avw = float((v.double() * Aw[key]).sum()), float((Av[key] * w.double()).sum())
            sym[key] = max(sym.get(key, 0.0), abs(vAw - Avw) / max(abs(vAw), abs(Avw), 1e-30))
            q = float((v.double() * Av[key]).sum())
            scale = float(Av[key].norm() * v.norm())
            psd[key] = min(psd.get(key, float("inf")), q / max(scale, 1e-30))
    return {"rhs_additivity_rel_err": rhs_rel,
            "normal_additivity_rel_err_max": max(norm_rel),
            "symmetry_rel_err": {f"{c}:{ch}": e for (c, ch), e in sym.items()},
            "min_vAv_over_norms": {f"{c}:{ch}": e for (c, ch), e in psd.items()},
            "probes": n_probe, "seed": seed}, b


# ---------------------------------------------------------------------------
# grouped frozen quadratic and leave-one-out solves
# ---------------------------------------------------------------------------
@dataclass
class FrozenSystem:
    lin: object
    terms: list
    Lambda_prior: torch.Tensor
    b12: torch.Tensor                 # production RHS (prior + measurement), [N,12]
    meas_normal: object
    lm: float
    grouped: object | None
    precond: object | None = None
    cache: dict = field(default_factory=dict)

    def G(self, z):
        if self.grouped is None:
            return z
        return torch.cat([self.grouped.expand(z[:, :6].float()), self.grouped.expand(z[:, 6:].float())], 1).double()

    def GT(self, v):
        if self.grouped is None:
            return v
        return torch.cat([self.grouped.reduce(v[:, :6].float().contiguous()),
                          self.grouped.reduce(v[:, 6:].float().contiguous())], 1).double()

    def A12_full(self, v12):
        from estimator.covariance import apply_prior
        from estimator.renderer_bridge import embed_pose_to_full_state, extract_pose
        meas = embed_pose_to_full_state(self.meas_normal(extract_pose(v12).float()).double())
        return apply_prior(self.Lambda_prior.double(), v12) + meas + self.lm * v12

    def A12_minus(self, v12, keys):
        """A minus the listed (camera, channel) pieces."""
        from estimator.renderer_bridge import embed_pose_to_full_state, extract_pose
        full = self.A12_full(v12)
        if not keys:
            return full
        parts = camera_channel_normal(self.lin, self.terms, extract_pose(v12), keys=set(keys))
        return full - embed_pose_to_full_state(sum(parts[k] for k in keys))

    def solve(self, keys=(), rhs12=None, tol=1e-7, max_iters=800):
        """Grouped solve of (A - A_keys) d = b - b_keys (or rhs12). Returns (d12 per Gaussian, d_group, info)."""
        from estimator.renderer_bridge import embed_pose_to_full_state
        if rhs12 is None:
            rhs12 = self.b12.double()
            if keys:
                bk = self.cache.setdefault("b", camera_channel_rhs(self.lin, self.terms))
                rhs12 = rhs12 - embed_pose_to_full_state(sum(bk[k] for k in keys))
        bg = self.GT(rhs12)
        A = lambda z: self.GT(self.A12_minus(self.G(z), keys))          # noqa: E731
        M = (lambda r: self.precond(r.float()).double()) if self.precond is not None else None
        d, info = _pcg64(A, bg, M, tol=tol, max_iters=max_iters)
        return self.G(d), d, info


def _pcg64(A, b, M=None, tol=1e-7, max_iters=800):
    """Float64 preconditioned CG on top of float32 operators; true residual reported."""
    x = torch.zeros_like(b)
    r = b.clone()
    z = M(r) if M is not None else r
    p = z.clone()
    rz = float((r * z).sum())
    bn = float(b.norm()) or 1.0
    it = 0
    for it in range(1, max_iters + 1):
        Ap = A(p)
        pAp = float((p * Ap).sum())
        if pAp <= 0:
            break
        a = rz / pAp
        x = x + a * p
        r = r - a * Ap
        if float(r.norm()) / bn < tol:
            break
        z = M(r) if M is not None else r
        rz_new = float((r * z).sum())
        p = z + (rz_new / rz) * p
        rz = rz_new
    true = float((b - A(x)).norm()) / bn
    # the operators are float32 CUDA kernels: a true relative residual below ~1e-5 is not
    # resolvable, so convergence is judged against max(10 tol, that floor) and both reported
    floor = 1e-5
    return x, {"iterations": it, "recursive_rel_residual": float(r.norm()) / bn, "true_rel_residual": true,
               "float32_operator_floor": floor, "converged": true <= max(10 * tol, floor)}


def leave_one_out(sys: FrozenSystem, n_cam: int, identity_check=True, tol=1e-7, max_iters=800) -> dict:
    """delta_all and I_k for every camera and every (camera, channel)."""
    d_all, _, info_all = sys.solve((), tol=tol, max_iters=max_iters)
    arms = {}
    for c in range(n_cam):
        arms[f"minus_cam{c}"] = [(c, ch) for ch in CHANNELS]
        for ch in CHANNELS:
            arms[f"minus_cam{c}_{ch}"] = [(c, ch)]
    out = {"delta_all": d_all, "solver_all": info_all, "arms": {}}
    from estimator.renderer_bridge import embed_pose_to_full_state, extract_pose
    for name, keys in arms.items():
        d_k, dg_k, info = sys.solve(keys, tol=tol, max_iters=max_iters)
        I = d_all - d_k
        rec = {"keys": [f"{c}:{ch}" for c, ch in keys], "influence": I, "solver": info}
        if identity_check and name.count("_") == 1:          # whole-camera arms
            bk = sys.cache.setdefault("b", camera_channel_rhs(sys.lin, sys.terms))
            Ak = camera_channel_normal(sys.lin, sys.terms, extract_pose(d_k), keys=set(keys))
            rhs = embed_pose_to_full_state(sum(bk[k] for k in keys) - sum(Ak[k] for k in keys))
            y, _, info_y = sys.solve((), rhs12=rhs, tol=tol, max_iters=max_iters)
            rec["identity_rel_err"] = float((y - I).norm() / I.norm().clamp_min(1e-30))
            rec["identity_solver"] = info_y
        out["arms"][name] = rec
    return out


# ---------------------------------------------------------------------------
# contributor traces (production CSR, exactly the composited contributors)
# ---------------------------------------------------------------------------
def _edge_distance(D: torch.Tensor, valid: torch.Tensor, jump_m=0.02) -> np.ndarray:
    """Pixels to the nearest observed-depth discontinuity (3x3 range > jump_m), reporting only."""
    from scipy.ndimage import distance_transform_edt
    Dv = torch.where(valid, D, torch.zeros_like(D))
    big = torch.where(valid, Dv, torch.full_like(Dv, -1e9))
    small = torch.where(valid, Dv, torch.full_like(Dv, 1e9))
    rng = (torch.nn.functional.max_pool2d(big[None, None], 3, 1, 1)
           + torch.nn.functional.max_pool2d(-small[None, None], 3, 1, 1))[0, 0]
    edge = ((rng > jump_m) & valid).cpu().numpy()
    return distance_transform_edt(~edge) if edge.any() else np.full(edge.shape, np.inf)


CLASSES = ("MATCH_COMPATIBLE", "BEHIND_FIRST_RETURN", "IN_FRONT_CONTRADICTION", "EDGE_OR_MIXED",
           "INVALID_OR_OUTSIDE_OBJECT")


def classify(d_over_sigma, valid, edge_px, k_sigma=3.0, edge_band_px=3.0):
    """REPORTING ONLY. Continuous d/sigma is always kept next to the label."""
    if not valid:
        return "INVALID_OR_OUTSIDE_OBJECT"
    if edge_px <= edge_band_px:
        return "EDGE_OR_MIXED"
    if abs(d_over_sigma) <= k_sigma:
        return "MATCH_COMPATIBLE"
    return "BEHIND_FIRST_RETURN" if d_over_sigma > 0 else "IN_FRONT_CONTRADICTION"


def contributor_traces(lin, obs, cfg, gaussian_ids, camera_names, sigma_fn, max_front=3) -> list:
    """Per (camera, pixel, selected Gaussian) records from the production CSR (the
    contributors the renderer actually composited, T_STOP included).
    sigma_fn(camera_index, D, valid) -> [H,W] effective depth sd (sensor + declared)."""
    from observation_diagnostics.renderer_access import pixel_contributions
    sel = torch.as_tensor(sorted(set(int(g) for g in gaussian_ids)), dtype=torch.int64)
    rows = []
    for c, view in enumerate(lin.views):
        ob = obs[c]
        dev = view.depth.device
        D = ob.depth_observed.to(dev).float()
        valid = ob.depth_valid.to(dev).bool() & torch.isfinite(D) & (D > 0)
        M = ob.object_coverage.to(dev) > 0.5
        sd = sigma_fn(c, D, valid)
        edge = _edge_distance(D, valid)
        cc = pixel_contributions(view.cache)
        gid_t = cc.gid.long()
        hit = torch.isin(gid_t, sel.to(gid_t.device))
        if not bool(hit.any()):
            continue
        H, W = view.alpha.shape
        np_ = lambda t: t.detach().cpu().numpy()                               # noqa: E731
        gid, al, T, off = np_(gid_t), np_(cc.alpha).astype(np.float64), np_(cc.T).astype(np.float64), np_(cc.offsets)
        pix = np_(cc.pixel)
        zc = np_(view.cache.depth.reshape(-1)).astype(np.float64)
        A, Dh, nc, fT = (np_(view.alpha.reshape(-1)), np_(view.depth.reshape(-1)),
                         np_(view.n_contrib.reshape(-1)), np_(cc.final_T))
        Dn, vn, Mn, sdn = np_(D.reshape(-1)), np_(valid.reshape(-1)), np_(M.reshape(-1)), np_(sd.reshape(-1))
        eps = float(view.cache.eps_depth)
        for e in np.nonzero(np_(hit))[0].tolist():
            p, g = int(pix[e]), int(gid[e])
            q = al[e] * T[e]
            z = zc[g]
            ok = bool(vn[p])
            Do = float(Dn[p]) if ok else None
            s = float(sdn[p]) if ok else float("nan")
            d = (z - Do) if ok else float("nan")
            ds = d / s if ok and s > 0 else float("nan")
            ex = float(edge[p // W, p % W])
            front = sorted(((al[f] * T[f], int(gid[f]), zc[int(gid[f])]) for f in range(int(off[p]), e)),
                           reverse=True)[:max_front]
            rows.append({"camera": camera_names[c], "camera_index": c, "u": p % W, "v": p // W, "gaussian_id": g,
                         "alpha": float(al[e]), "T": float(T[e]), "q": float(q), "beta": float(q / (A[p] + eps)),
                         "gaussian_depth": float(z), "observed_depth": Do, "depth_valid": ok,
                         "mask": bool(Mn[p]), "rendered_depth": float(Dh[p]), "A": float(A[p]),
                         "final_T": float(fT[p]), "n_contrib": int(nc[p]), "signed_difference": d,
                         "sigma_effective": s, "d_over_sigma": ds, "depth_edge_distance_px": ex,
                         "class": classify(ds, ok and bool(Mn[p]), ex),
                         "front_contributors": [{"gaussian_id": gg, "q": float(qq), "depth": float(zz)}
                                                for qq, gg, zz in front]})
    return rows


def summarize_traces(rows) -> dict:
    """Per (Gaussian, camera): pixels, sum q, beta-weighted class fractions, median d/sigma."""
    by = {}
    for r in rows:
        k = (r["gaussian_id"], r["camera"])
        e = by.setdefault(k, {"pixels": 0, "sum_q": 0.0, "sum_beta": 0.0, "class_beta": {c: 0.0 for c in CLASSES},
                              "d_over_sigma": []})
        e["pixels"] += 1; e["sum_q"] += r["q"]; e["sum_beta"] += r["beta"]
        e["class_beta"][r["class"]] += r["beta"]
        if r["d_over_sigma"] == r["d_over_sigma"]:
            e["d_over_sigma"].append(r["d_over_sigma"])
    out = {}
    for (g, cam), e in by.items():
        tot = max(e["sum_beta"], 1e-30)
        out.setdefault(str(g), {})[cam] = {
            "pixels": e["pixels"], "sum_q": e["sum_q"], "sum_beta": e["sum_beta"],
            "class_fraction_beta_weighted": {k: v / tot for k, v in e["class_beta"].items()},
            "median_d_over_sigma": float(np.median(e["d_over_sigma"])) if e["d_over_sigma"] else None}
    return out


def camera_centres(lin) -> torch.Tensor:
    return torch.stack([torch.linalg.inv(v.camera.viewmatrix.double())[3, :3] for v in lin.views])


def cos_rows(a, b):
    na, nb = a.norm(dim=1), b.norm(dim=1)
    return (a * b).sum(1) / (na * nb).clamp_min(1e-30)
