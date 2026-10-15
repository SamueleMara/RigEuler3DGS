"""Endpoint comparison of the two trained models and their Phase-1 transports.

  F0 = G_F at original frame 0 (the initial PLY, its own anchor)
  FT = G_F transported to frame T by the forward Phase-1 track
  B0 = G_B transported to frame 0 by the backward Phase-1 track
  BT = G_B at frame T (the final PLY, its own anchor)

Every state is rendered ALONE with the production IEKF renderer
(estimator.renderer_bridge -> iekf_build_cache/iekf_render) and its exact
contributors come from observation_diagnostics.renderer_access (the
production loop, verified per view). Pixel categories, per-Gaussian
accounting and pair sums reuse pixel_evidence / gaussian_evidence /
pair_evidence unchanged.

Joint-informed (J1) diagnostic states, built in RAM only:
  FT_J = F0 with the rows owned by an estimated moving part moved by that
         part's ESTIMATED motion T(T) T(0)^{-1} (positions and orientations);
         every other row stays at its frame-0 place (the joint model's static
         prediction). B0_J likewise from BT with T(0) T(T)^{-1}.
Ownership in the estimating source is the estimator's own support set; in
the other source it is INFERRED by proximity to the owned part at that
source's anchor frame and is labelled as such (low confidence).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import gaussian_evidence as ge
from . import pixel_evidence as pe
from .gt_firewall import assert_allowed
from .renderer_access import pixel_contributions

CENTER_ONLY = "CENTER_ONLY_FIXED_ORIENTATION"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# sources (GT-free loading route)
# ---------------------------------------------------------------------------

class EndpointSource:
    """A trained endpoint model + its chronological Phase-1 track, loaded through
    the scene loader's inference_only route (no GT part labels)."""

    def __init__(self, key, model_dir: Path, centers_dir: Path, dataset_dir: Path,
                 endpoint: str, direction: str, device="cuda"):
        from tools.run_iekf_scene import load_scene, state_from_gaussians
        self.key, self.endpoint, self.direction = key, endpoint, direction
        self.model = Path(model_dir)
        its = sorted((int(p.name.split("_")[-1]), p) for p in (self.model / "point_cloud").glob("iteration_*")
                     if (p / "point_cloud.ply").is_file())
        if not its:
            raise SystemExit(f"no trained PLY under {self.model}")
        self.ply = its[-1][1] / "point_cloud.ply"
        self.ply_sha256 = sha256_file(self.ply)
        _d, _p, g, scene = load_scene(str(self.model), str(dataset_dir), inference_only=True)
        self.cams_all = scene.getTrainCameras()
        self.sh_degree = int(getattr(g, "max_sh_degree", getattr(g, "active_sh_degree", 0)))
        self.sh_rest_dim = int(g.get_features_rest.shape[1]) if hasattr(g, "get_features_rest") else 0
        self.base = state_from_gaussians(g)
        self.N = int(self.base.mu.shape[0])
        del g, scene
        cdir = Path(centers_dir)
        man = cdir / "trajectory_manifest.json"
        assert_allowed(man)
        self.manifest = json.loads(man.read_text())
        m = self.manifest
        if m.get("trajectory_order") != "original_chronological":
            raise SystemExit(f"{cdir}: trajectory_order {m.get('trajectory_order')!r}")
        if m.get("canonical_endpoint") != endpoint or m.get("direction") != direction:
            raise SystemExit(f"{cdir}: endpoint/direction {m.get('canonical_endpoint')}/"
                             f"{m.get('direction')}, expected {endpoint}/{direction}")
        files = sorted(cdir.glob("centers_frame_*.npy"))
        self.frames = [int(f.stem.split("_")[-1]) for f in files]
        if self.frames != list(range(len(files))):
            raise SystemExit(f"{cdir}: centre files are not 0..{len(files) - 1}")
        self.mu = np.stack([np.load(f).astype(np.float32) for f in files])
        if self.mu.shape[1] != self.N:
            raise SystemExit(f"{cdir}: {self.mu.shape[1]} rows, PLY has {self.N}")
        idx = cdir / "original_gaussian_indices.npy"
        self.row_mapping = "absent: rows are PLY rows"
        if idx.is_file():
            a = np.load(idx)
            if not np.array_equal(a, np.arange(self.N)):
                raise SystemExit(f"{idx} is not the identity; refusing to reorder rows")
            self.row_mapping = "identity (original_gaussian_indices.npy)"
        self.anchor = int(m["canonical_original_frame"])
        ply_xyz = self.base.mu.detach().cpu().numpy()
        self.anchor_error = float(np.abs(self.mu[self.anchor] - ply_xyz).max())
        if self.anchor_error > 1e-5:
            raise SystemExit(f"{cdir}: centres at the anchor frame {self.anchor} differ from the "
                             f"PLY by {self.anchor_error}")
        self.track_sha256 = hashlib.sha256(self.mu.tobytes()).hexdigest()
        self.device = device
        self.base = dataclasses.replace(
            self.base, **{f: getattr(self.base, f).to(device)
                          for f in ("mu", "R", "v", "omega", "colors", "opacity", "scales")})
        self.state_mode = CENTER_ONLY

    def state(self, frame: int, mu=None, R=None):
        mu_t = torch.as_tensor(self.mu[frame] if mu is None else mu, dtype=torch.float32,
                               device=self.device).contiguous()
        R_t = self.base.R if R is None else torch.as_tensor(R, dtype=torch.float32,
                                                            device=self.device).contiguous()
        return dataclasses.replace(self.base, mu=mu_t, R=R_t)

    def describe(self) -> dict:
        return {"key": self.key, "endpoint": self.endpoint, "direction": self.direction,
                "ply": str(self.ply), "ply_sha256": self.ply_sha256, "rows": self.N,
                "anchor_frame": self.anchor, "anchor_centres_vs_ply_max_abs": self.anchor_error,
                "frames": [self.frames[0], self.frames[-1]], "row_mapping": self.row_mapping,
                "sh_degree": self.sh_degree,
                "appearance_used": "DC term of the SH (frozen radiance, as Phase 1 rendered); "
                                   "higher SH bands loaded, not rendered",
                "state_mode": self.state_mode,
                "processing_order": [self.manifest["processing_order_original_indices"][0],
                                     self.manifest["processing_order_original_indices"][-1]],
                "track_sha256": self.track_sha256, "load_route": "load_scene(inference_only=True)"}


# ---------------------------------------------------------------------------
# rendering one view
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class View:
    name: str              # F0, B0, FT, BT, FT_J1, B0_J1, ...
    frame: int
    cam: str
    lin: object            # CameraLinearization (rgb, depth, alpha, cache)
    contrib: object        # renderer_access.Contributions
    state: object

    @property
    def d(self):
        return {"rgb": self.lin.rgb, "depth": self.lin.depth, "alpha": self.lin.alpha}


def render(name, state, spec, frame, cam) -> View:
    from estimator.renderer_bridge import build_linearization
    v = build_linearization(state, [spec]).views[0]
    return View(name, frame, cam, v, pixel_contributions(v.cache), state)


def conservation(view: View) -> dict:
    c, v = view.contrib, view.lin
    HW = c.H * c.W
    q = c.q.double()
    A = torch.zeros(HW, dtype=torch.float64, device=q.device).index_add_(0, c.pixel, q)
    Z = torch.zeros_like(A).index_add_(0, c.pixel, q * v.cache.depth[c.gid.long()].double())
    cov = v.alpha.reshape(-1) > 0.05
    return {"max_abs_sum_q_minus_A": float((A - v.alpha.reshape(-1).double()).abs().max()),
            "max_abs_depth_recon_m": float((Z / (A + v.cache.eps_depth) - v.depth.reshape(-1).double())[cov].abs().max())
            if bool(cov.any()) else 0.0,
            "final_T_bitwise": bool(torch.equal(c.final_T, v.transmittance.reshape(-1))),
            "early_terminated_pixels": int(c.terminated.sum())}


def static_observation(g0: dict, gT: dict, tol: float) -> dict:
    """Pixels whose MEASURED depth is the same at frames 0 and T: candidate
    observed-static surface at this camera (from sensor data only)."""
    v = g0["valid"] & gT["valid"] & ((g0["depth"] - gT["depth"]).abs() <= tol)
    return {"rgb": gT["rgb"], "depth": gT["depth"], "valid": v}


# ---------------------------------------------------------------------------
# patches from witness mass
# ---------------------------------------------------------------------------

def group_rows(xyz: np.ndarray, rows: np.ndarray, scales: np.ndarray, min_r=0.01,
               max_rows=400, split_cell_m=0.10):
    """3-D proximity components at one endpoint (radius 2 x median splat scale).
    Chaining can join a whole moving part into one component, so a component
    with more than `max_rows` rows is split into `split_cell_m` grid cells:
    patches stay local enough to inspect."""
    groups, r = _components(xyz, rows, scales, min_r)
    out = []
    for g in groups:
        if g.size <= max_rows:
            out.append(g)
            continue
        cells = np.floor(xyz[g] / split_cell_m).astype(np.int64)
        _u, inv = np.unique(cells, axis=0, return_inverse=True)
        out += [np.sort(g[inv.reshape(-1) == k]) for k in range(inv.max() + 1)]
    return out, r


def _components(xyz, rows, scales, min_r):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    if rows.size == 0:
        return [], 0.0
    r = max(2.0 * float(np.median(scales[rows])), min_r)
    P = cKDTree(xyz[rows]).query_pairs(r, output_type="ndarray")
    n = rows.size
    m = coo_matrix((np.ones(len(P)), (P[:, 0], P[:, 1])), shape=(n, n)) if len(P) else coo_matrix((n, n))
    nc, lab = connected_components(m, directed=False)
    return [np.sort(rows[lab == c]) for c in range(nc)], r


def witness_pixels(contrib, rows_t: torch.Tensor, z_all, W: torch.Tensor, gt_depth, valid,
                   tol: float, H: int, W_: int, top_k=200) -> dict:
    """Pixels where the patch's own depth-compatible contributions land on witness
    pixels W: (u, v, mass), sorted by mass, plus totals."""
    g = contrib.gid.long()
    sel = torch.isin(g, rows_t)
    if not bool(sel.any()):
        return {"pixels": 0, "mass": 0.0, "top": [], "bbox": None}
    pix = contrib.pixel[sel]
    q = contrib.q[sel].double()
    z = z_all[g[sel]].double()
    D = gt_depth.reshape(-1)[pix].double()
    ok = valid.reshape(-1)[pix] & ((z - D).abs() <= tol) & W.reshape(-1)[pix]
    m = torch.zeros(H * W_, dtype=torch.float64, device=q.device).index_add_(0, pix[ok], q[ok])
    nz = torch.nonzero(m > 0).squeeze(1)
    if nz.numel() == 0:
        return {"pixels": 0, "mass": 0.0, "top": [], "bbox": None}
    vals = m[nz]
    order = torch.argsort(vals, descending=True)[:top_k]
    top = [(int(p % W_), int(p // W_), float(x)) for p, x in zip(nz[order].tolist(), vals[order].tolist())]
    ys, xs = (nz // W_).cpu().numpy(), (nz % W_).cpu().numpy()
    return {"pixels": int(nz.numel()), "mass": float(vals.sum()), "top": top,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "map": m}


def mask_fraction_explained(pixmap: torch.Tensor, S: torch.Tensor) -> float:
    """Of the patch's witness pixels (weighted by its mass), the share that a
    different render explains (S = that render's S_d^depth)."""
    w = pixmap.reshape(-1)
    tot = float(w.sum())
    return float((w * S.reshape(-1).double()).sum()) / tot if tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def project(spec, xyz):
    p = torch.as_tensor(xyz, dtype=torch.float32, device=spec.viewmatrix.device)
    ph = torch.cat([p, torch.ones_like(p[:, :1])], 1)
    hom = ph @ spec.projmatrix
    ndc = hom[:, :2] / (hom[:, 3:4] + 1e-7)
    z = (ph @ spec.viewmatrix)[:, 2]
    u = ((ndc[:, 0] + 1.0) * spec.image_width - 1.0) * 0.5
    v = ((ndc[:, 1] + 1.0) * spec.image_height - 1.0) * 0.5
    return u.cpu().numpy(), v.cpu().numpy(), z.cpu().numpy()


def visibility_by_depth(spec, xyz, depth_np, valid_np, tol):
    """Per member: OUT_OF_VIEW, GT_INVALID, OCCLUDED (observed surface in front),
    AT_OBSERVED_DEPTH, BEHIND_GAP (observed surface behind it)."""
    u, v, z = project(spec, xyz)
    H, W = depth_np.shape
    inside = (z > 0.01) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    out = np.full(len(z), "OUT_OF_VIEW", dtype=object)
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    d, ok = depth_np[vi, ui], valid_np[vi, ui]
    out[inside & ~ok] = "GT_INVALID"
    out[inside & ok & (d < z - tol)] = "OCCLUDED"
    out[inside & ok & (np.abs(d - z) <= tol)] = "AT_OBSERVED_DEPTH"
    out[inside & ok & (d > z + tol)] = "BEHIND_GAP"
    return out


def rotation_transport(R_rows: torch.Tensor, Rm: np.ndarray) -> torch.Tensor:
    """R^J = R_part(t) R_part(ref)^{-1} R_ref: left-multiply the Gaussian's
    world rotation (the renderer's R in GaussianMotionState) by the part's."""
    return torch.as_tensor(Rm, dtype=R_rows.dtype, device=R_rows.device) @ R_rows


def j1_state(src: EndpointSource, ref_frame: int, owned: np.ndarray, T: np.ndarray):
    """src at ref_frame with `owned` rows moved by the 4x4 T (positions and
    orientations); all other rows stay at ref_frame (static prediction)."""
    mu = src.mu[ref_frame].copy()
    if owned.size:
        mu[owned] = mu[owned] @ T[:3, :3].T + T[:3, 3]
    R = src.base.R.clone()
    if owned.size:
        ot = torch.as_tensor(owned, device=R.device, dtype=torch.long)
        R[ot] = rotation_transport(R[ot], T[:3, :3])
    return src.state(ref_frame, mu=mu, R=R)


def infer_ownership(other_xyz: np.ndarray, part_xyz: np.ndarray, radius: float) -> np.ndarray:
    """Rows of the other source within `radius` of the estimated part's geometry
    at the same frame: an INFERRED, low-confidence ownership."""
    from scipy.spatial import cKDTree
    if part_xyz.size == 0:
        return np.empty(0, np.int64)
    d, _ = cKDTree(part_xyz).query(other_xyz, k=1, distance_upper_bound=radius)
    return np.flatnonzero(np.isfinite(d))


# ---------------------------------------------------------------------------
# selective image-sensitivity probe: H_s^T W H_o on a shortlisted region
# ---------------------------------------------------------------------------

def sensitivity_alignment(view_s: View, rows_s, view_o: View, rows_o, region: torch.Tensor,
                          valid: torch.Tensor) -> dict:
    """For unit world translations (x, y, z) of a patch in its own render and of
    its candidate counterpart in the other render, the cosine of the two pixel
    tangents over `region` under the sensor precisions (1/sigma_rgb^2,
    1/sigma_d^2 on valid depth). Production iekf_apply_H; nothing global.
    A 3x3 matrix, rows = the patch's direction, cols = the counterpart's.
    This is image-sensitivity alignment, not a correspondence."""
    from diff_gaussian_rasterization import iekf_apply_H

    def tangents(view, rows):
        out = []
        n = view.lin.cache.num_gaussians
        rt = torch.as_tensor(rows, device="cuda", dtype=torch.long)
        for k in range(3):
            d = torch.zeros(n, 6, device="cuda")
            d[rt, k] = 1.0
            drgb, ddep = iekf_apply_H(view.lin.cache, d)
            out.append((drgb, ddep))
        return out
    if len(rows_s) == 0 or len(rows_o) == 0:
        return {"available": False, "reason": "no counterpart rows"}
    ts, to = tangents(view_s, rows_s), tangents(view_o, rows_o)
    m = region.reshape(view_s.lin.depth.shape)
    md = m & valid
    wr, wd = 1.0 / pe.RGB_SIGMA ** 2, 1.0 / pe.DEPTH_SIGMA_M ** 2

    def ip(a, b):
        return float(wr * (a[0] * b[0] * m).sum() + wd * (a[1] * b[1] * md).sum())
    C = np.zeros((3, 3))
    ns = [max(ip(a, a), 1e-30) for a in ts]
    no = [max(ip(b, b), 1e-30) for b in to]
    for i in range(3):
        for j in range(3):
            C[i, j] = ip(ts[i], to[j]) / np.sqrt(ns[i] * no[j])
    return {"available": True, "cosine_xyz": C.round(4).tolist(),
            "diag_mean": float(np.mean(np.diag(C))),
            "region_pixels": int(m.sum()),
            "units": "per metre of world translation; RGB and depth weighted by the "
                     "Phase-1 sensor precisions",
            "caveat": "CENTER_ONLY_FIXED_ORIENTATION footprints; alignment of image "
                      "sensitivities, not physical identity"}


def adjoint_check(view: View, n_rows=200, seed=0) -> dict:
    """<H v, z> = <v, H^T z> on a real view (frozen sort), and H v vs a finite
    difference of the re-render for a small translation of a few rows."""
    from diff_gaussian_rasterization import iekf_apply_H, iekf_apply_HT
    from estimator.renderer_bridge import build_linearization
    c = view.lin.cache
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = c.num_gaussians
    v = torch.randn(n, 6, device="cuda", generator=g)
    zc = torch.randn_like(view.lin.rgb)
    zd = torch.randn_like(view.lin.depth)
    drgb, ddep = iekf_apply_H(c, v)
    lhs = float((drgb.double() * zc.double()).sum() + (ddep.double() * zd.double()).sum())
    ht = iekf_apply_HT(c, zc, zd, view.lin.transmittance, view.lin.alpha, view.lin.depth,
                       view.lin.n_contrib)
    rhs = float((v.double() * ht.double()).sum())
    scale = float(torch.sqrt((drgb.double() ** 2).sum() + (ddep.double() ** 2).sum())
                  * torch.sqrt((zc.double() ** 2).sum() + (zd.double() ** 2).sum()))
    # central finite difference of re-renders, compared where the frozen
    # convention holds: pixels whose ordered contributor list is the same in
    # both perturbed renders as in the linearised one (no resort, no alpha or
    # T_STOP crossing). Elsewhere a re-render differs discontinuously by design.
    vis = torch.unique(view.contrib.gid.long())
    rows = vis[torch.randperm(vis.numel(), generator=g, device="cuda")[:n_rows]]
    d = torch.zeros(n, 6, device="cuda")
    d[rows, 0] = 1.0
    eps = 1e-4
    st = view.state

    def rerender(sign):
        mu2 = st.mu.clone()
        mu2[rows, 0] += sign * eps
        v2 = build_linearization(dataclasses.replace(st, mu=mu2), [view.lin.camera]).views[0]
        return v2, pixel_contributions(v2.cache)

    def signature(cn):
        HW = cn.H * cn.W
        k = torch.arange(cn.gid.numel(), device="cuda") - cn.offsets[:-1].repeat_interleave(cn.counts)
        s1 = torch.zeros(HW, dtype=torch.float64, device="cuda").index_add_(
            0, cn.pixel, (cn.gid.double() + 1) * (k.double() + 1))
        return cn.counts, s1
    vp, cp = rerender(+1)
    vm, cm = rerender(-1)
    c0, s0 = signature(view.contrib)
    frozen = torch.ones_like(c0, dtype=torch.bool)
    for cn in (cp, cm):
        c_, s_ = signature(cn)
        frozen &= (c_ == c0) & (s_ == s0)
    fd = (vp.rgb - vm.rgb) / (2 * eps)
    jv, _ = iekf_apply_H(c, d)
    m = frozen.reshape(view.lin.depth.shape)
    touched = (jv.abs().sum(0) > 0)
    sel = m & touched
    rel = float((fd - jv)[:, sel].norm() / max(float(jv[:, sel].norm()), 1e-12)) if bool(sel.any()) else float("nan")
    return {"adjoint_relative_gap": abs(lhs - rhs) / max(scale, 1e-30),
            "adjoint_gap_over_abs_lhs": abs(lhs - rhs) / max(abs(lhs), 1e-12),
            "adjoint_normalisation": "|<Hv,z> - <v,H^T z>| / (|Hv| |z|); float32 atomics in the VJP "
                                     "make the raw ratio run-to-run noisy",
            "jvp_vs_finite_difference_relative": rel, "fd_step_m": eps, "fd": "central",
            "rows_moved": int(rows.numel()),
            "pixels_touched_by_the_JVP": int(touched.sum()),
            "of_which_frozen_contributor_lists": int(sel.sum()),
            "note": "compared on pixels whose ordered contributor list is unchanged by the "
                    "perturbation (the frozen-sort convention the JVP linearises)"}


# ---------------------------------------------------------------------------
# declared decision rules (pure functions; the CLI and the tests share them)
# ---------------------------------------------------------------------------

def j0_label(R: dict, *, s, o, fail, fail_transported, t_end, t_oth, n_pix, n_cam, frac_same,
             rgb_ratio, frac_other, is_static, frac_static, excl, unseen, sup_o, world_near=0.0):
    """Joint-agnostic hypothesis for a witness patch of source s (explained by s,
    not by the other source's render `fail` at the same endpoint). Order matters
    and is part of the declaration. The estimated joint is NOT an input: a J1
    result can never delete, reject or force-match a patch."""
    if n_pix < R["MIN_WITNESS_PIXELS"] or n_cam < R["MIN_WITNESS_CAMERAS"]:
        return "INSUFFICIENT_EVIDENCE", f"{n_pix} witness pixels in {n_cam} cameras"
    static_world = is_static and (frac_static >= R["COUNTERPART_FRACTION"]
                                  or world_near >= R["WORLD_NEAR_FRACTION"])
    if fail_transported and (frac_other >= R["COUNTERPART_FRACTION"] or static_world):
        how = (f"at frame {t_oth} these rows share the observed surface with {o} ({frac_other:.2f})"
               if frac_other >= R["COUNTERPART_FRACTION"] else
               f"static by its own track, and the anchored {o} has rows at the same world place "
               f"(co-contribution {frac_static:.2f}, within-radius fraction {world_near:.2f})")
        return "TRANSPORT_MISMATCH", f"{how}; so {o} has this surface and its transport {fail} misplaces it"
    if excl < R["EXCLUSIVE_FRINGE"] and frac_same >= R["COMMON_FRACTION"]:
        return "COMMON_SURFACE", (f"only {excl:.2f} of its supported mass is exclusive and {frac_same:.2f} "
                                  f"of its depth-compatible mass shares the surface with {o}: a local fringe")
    if frac_same >= R["COUNTERPART_FRACTION"] and rgb_ratio < R["APPEARANCE_RGB_RATIO"]:
        return "APPEARANCE_SHAPE_DISAGREEMENT", (f"depth-compatible {o} counterpart ({frac_same:.2f}) but "
                                                 f"colour-compatible on only {rgb_ratio:.2f}")
    if not fail_transported:
        return "MISSING_IN_ANCHOR_PROPOSAL", (f"{o} at its own anchor frame does not explain {n_pix} "
                                              f"observed pixels in {n_cam} cameras that {s} explains")
    if unseen >= R["UNSEEN_FRACTION"] or sup_o < R["S_MIN"]:
        return f"{s}_UNIQUE_OBSERVED_PROPOSAL", (
            f"observed in {n_cam} cameras ({n_pix} px); no {o} counterpart at frame {t_end} ({frac_same:.2f}) "
            f"or at frame {t_oth} ({frac_other:.2f}); at frame {t_oth} {unseen:.0%} of its projections are "
            "occluded/out of view/invalid")
    return "CONFLICT_AMBIGUOUS", (f"visible at frame {t_oth} ({1 - unseen:.0%}) yet neither explained there "
                                  f"(own supported mass {sup_o:.1f}) nor matched ({frac_other:.2f}); "
                                  f"same-endpoint counterpart {frac_same:.2f}, exclusive {excl:.2f}")


def j1_label(R: dict, explained):
    """Joint-informed reading: the share of the patch's witness pixels the other
    source explains once moved by the ESTIMATED part motion."""
    if explained is None:
        return "J1_NOT_APPLICABLE_ANCHOR_FAILS"
    if explained != explained:
        return "J1_NO_WITNESS"
    if explained >= R["J1_RECOVERS"]:
        return "J1_TRANSPORT_EXPLAINS"
    if explained <= R["J1_FAILS"]:
        return "J1_DOES_NOT_RECOVER"
    return "J1_PARTIAL"


def static_candidates(keys, O_surf, dist3d_surf, q2F, q2B, NB, NF, supported_F, static_F, static_B,
                      R: dict, r_match: float) -> dict:
    """F0 vs BT in world coordinates (contrast C, observed-static pixels only).

    For each F row: its best BT partner by same-surface overlap, rho_surf
    (cosine of the two same-surface contribution maps) and the overlap-weighted
    world distance of their centres -- no transport involved.
      world match: supported on observed-static pixels, rho >= STATIC_RHO, distance <= r_match
      tier STATIC_COMMON_CANDIDATE: world match AND both free tracks static
      tier STATIC_WORLD_MATCH_TRACK_DISAGREES: world match but a free track moves the
           row (evidence of a transport error, not of motion -- marked uncertain)
    """
    i, j = keys // NB, keys % NB
    rho = O_surf / np.sqrt(np.maximum(q2F[i] * q2B[j], 1e-30))
    dist = np.divide(dist3d_surf, O_surf, out=np.full(rho.size, np.inf), where=O_surf > 0)
    o = np.lexsort((-O_surf, i))
    st = np.flatnonzero(np.r_[True, i[o][1:] != i[o][:-1]]) if i.size else np.empty(0, np.int64)
    best = np.full(NF, -1)
    best_rho, best_dist = np.zeros(NF), np.full(NF, np.inf)
    if st.size:
        best[i[o][st]] = j[o][st]
        best_rho[i[o][st]] = rho[o][st]
        best_dist[i[o][st]] = dist[o][st]
    has = best >= 0
    world = has & supported_F & (best_rho >= R["STATIC_RHO"]) & (best_dist <= r_match)
    both_static = np.zeros(NF, bool)
    both_static[has] = static_F[has] & static_B[best[has]]
    tier = np.full(NF, "", dtype=object)
    tier[world & both_static] = "STATIC_COMMON_CANDIDATE"
    tier[world & ~both_static] = "STATIC_WORLD_MATCH_TRACK_DISAGREES"
    return {"best_B": best, "rho": best_rho, "dist": best_dist, "world": world,
            "full": world & both_static, "tier": tier, "rho_all": rho}
