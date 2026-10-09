"""Pixel-mediated Gaussian matching at the final frame.

The final image plane is not only a render target; it is a *correspondence
domain* between the two independently trained 3DGS models.

    final-frame pixels  -> final Gaussian contributors            (cached once)
    same pixels         -> transported source Gaussian candidates (per hypothesis)

If a source Gaussian transported under hypothesis k lands on the same calibrated
viewing rays, at a compatible depth, as the final Gaussians that actually explain
those pixels, then k is the right hypothesis for it.

Everything here is DETACHED evidence: the frozen part trajectories and the fixed
final model produce E_pixel[i,k], which the outer E-step then weights by the
membership probabilities. No ground-truth labels are ever read (spec section 37).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# section 39: data structures
# ---------------------------------------------------------------------------
@dataclass
class PixelContributorMap:
    """Top-L final-Gaussian contributors for one camera."""

    ids: torch.Tensor        # [H, W, L] int, -1 = empty slot
    weights: torch.Tensor    # [H, W, L] float, rho = T_before * alpha
    depths: torch.Tensor     # [H, W, L] float, camera-space z
    camera_id: int = 0
    metadata: dict | None = None

    def to(self, *a, **kw) -> "PixelContributorMap":
        return PixelContributorMap(
            self.ids.to(*a, **kw), self.weights.to(*a, **kw), self.depths.to(*a, **kw),
            self.camera_id, self.metadata,
        )


@dataclass
class PixelCandidateEvidence:
    """Per-source-Gaussian, per-hypothesis pixel matching evidence."""

    energy: torch.Tensor        # [N, K+1]
    confidence: torch.Tensor    # [N, K+1]
    overlap: torch.Tensor       # [N, K+1]
    xyz_energy: torch.Tensor    # [N, K+1]
    depth_energy: torch.Tensor  # [N, K+1]
    num_views: torch.Tensor     # [N, K+1]
    # CODEX_render_response_contrastive §4 needs E_ik per OBSERVATION, not the
    # view-aggregated energy. Populated only when ``collect_per_view=True``; the
    # observation axis is the fixed set of final cameras.
    per_view_energy: torch.Tensor | None = None   # [N, K+1, V]
    per_view_valid: torch.Tensor | None = None    # [N, K+1, V] bool

    def as_dict(self) -> dict:
        out = {
            "energy": self.energy, "confidence": self.confidence, "overlap": self.overlap,
            "xyz_energy": self.xyz_energy, "depth_energy": self.depth_energy,
            "num_views": self.num_views,
        }
        if self.per_view_energy is not None:
            out["per_view_energy"] = self.per_view_energy
            out["per_view_valid"] = self.per_view_valid
        return out


# ---------------------------------------------------------------------------
# sections 3-7: render contributor maps and cache them
# ---------------------------------------------------------------------------
def _camera_hashes(camera) -> dict:
    def _h(t):
        if t is None:
            return ""
        return hashlib.sha1(torch.as_tensor(t).detach().cpu().numpy().tobytes()).hexdigest()[:16]

    return {
        "intrinsics_hash": _h(torch.tensor([float(camera.FoVx), float(camera.FoVy),
                                            int(camera.image_width), int(camera.image_height)])),
        "extrinsics_hash": _h(camera.world_view_transform),
    }


def render_contributors(
    camera,
    gaussians,
    *,
    top_l: int = 4,
    contributor_min_weight: float = 1.0e-4,
    scaling_modifier: float = 1.0,
    override_xyz: torch.Tensor | None = None,
    override_rotation: torch.Tensor | None = None,
    override_opacity: torch.Tensor | None = None,
) -> PixelContributorMap:
    """sections 3-6: per-pixel top-L contributors ranked by rho = T_before * alpha."""
    from diff_gaussian_rasterization import _C

    xyz = gaussians.get_xyz if override_xyz is None else override_xyz
    rotations = gaussians.get_rotation if override_rotation is None else override_rotation
    opacity = gaussians.get_opacity if override_opacity is None else override_opacity
    scales = gaussians.get_scaling
    device = xyz.device
    n = int(xyz.shape[0])

    bg = torch.zeros(3, dtype=torch.float32, device=device)
    colors = torch.ones((n, 3), dtype=torch.float32, device=device)
    with torch.no_grad():
        ids, weights, depths, _rgb, _radii = _C.rasterize_gaussians_contributors(
            bg,
            xyz.contiguous().float(),
            colors,
            opacity.contiguous().float(),
            scales.contiguous().float(),
            rotations.contiguous().float(),
            float(scaling_modifier),
            torch.empty(0, device=device),
            camera.world_view_transform.contiguous(),
            camera.full_proj_transform.contiguous(),
            float(math.tan(camera.FoVx * 0.5)),
            float(math.tan(camera.FoVy * 0.5)),
            int(camera.image_height),
            int(camera.image_width),
            torch.empty(0, device=device),
            0,
            camera.camera_center.contiguous(),
            False,
            False,
            int(top_l),
            float(contributor_min_weight),
            False,
        )
    return PixelContributorMap(ids.detach(), weights.detach(), depths.detach())


CACHE_SCHEMA = 2


def build_target_cache(
    cameras,
    final_gaussians,
    cache_dir: Path,
    *,
    top_l: int = 4,
    contributor_min_weight: float = 1.0e-4,
    final_model_id: str = "",
    refresh: bool = False,
) -> list[PixelContributorMap]:
    """section 7: render the FIXED final 3DGS once per final camera and cache it.

    Metadata carries the camera id, H, W, L, final-model identifier and
    intrinsics/extrinsics hashes so a stale cache can never be silently reused.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    maps: list[PixelContributorMap] = []
    for idx, camera in enumerate(cameras):
        path = cache_dir / f"cam_{idx:03d}.pt"
        meta = {
            "schema_version": CACHE_SCHEMA,
            "camera_id": idx,
            "H": int(camera.image_height), "W": int(camera.image_width), "L": int(top_l),
            "final_model_id": str(final_model_id),
            "contributor_min_weight": float(contributor_min_weight),
            **_camera_hashes(camera),
        }
        if path.is_file() and not refresh:
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False)
                if blob.get("metadata") == meta:
                    maps.append(PixelContributorMap(
                        blob["ids"], blob["weights"], blob["depths"], idx, meta))
                    continue
            except Exception:
                pass
        cmap = render_contributors(
            camera, final_gaussians, top_l=top_l,
            contributor_min_weight=contributor_min_weight,
        )
        cmap.camera_id = idx
        cmap.metadata = meta
        torch.save({"ids": cmap.ids.cpu(), "weights": cmap.weights.cpu(),
                    "depths": cmap.depths.cpu(), "metadata": meta}, path)
        maps.append(cmap)
    return maps


# ---------------------------------------------------------------------------
# section 15: project a candidate Gaussian into a final camera
# ---------------------------------------------------------------------------
def project_gaussians(camera, xyz: torch.Tensor, cov3d: torch.Tensor) -> dict:
    """EWA splatting projection, matching the CUDA ``computeCov2D`` exactly.

    Returns pixel centre, camera depth, the 2D conic and a splat radius.
    """
    device = xyz.device
    view = camera.world_view_transform.to(device)          # column-major, row-vector convention
    w = int(camera.image_width)
    h = int(camera.image_height)
    tan_x = math.tan(camera.FoVx * 0.5)
    tan_y = math.tan(camera.FoVy * 0.5)
    fx = w / (2.0 * tan_x)
    fy = h / (2.0 * tan_y)

    t = xyz @ view[:3, :3] + view[3, :3]                   # camera space
    tz = t[:, 2].clamp_min(1.0e-6)
    limx, limy = 1.3 * tan_x, 1.3 * tan_y
    tx = (t[:, 0] / tz).clamp(-limx, limx) * tz
    ty = (t[:, 1] / tz).clamp(-limy, limy) * tz

    zero = torch.zeros_like(tz)
    j = torch.stack([
        torch.stack([fx / tz, zero, -(fx * tx) / (tz * tz)], dim=-1),
        torch.stack([zero, fy / tz, -(fy * ty) / (tz * tz)], dim=-1),
        torch.stack([zero, zero, zero], dim=-1),
    ], dim=1)                                              # [N,3,3]

    wmat = view[:3, :3].transpose(0, 1)                    # W as built in CUDA
    tmat = wmat.transpose(0, 1) @ j                        # T = W * J (col-major)
    cov = tmat.transpose(-1, -2) @ cov3d @ tmat
    cxx = cov[:, 0, 0] + 0.3
    cxy = cov[:, 0, 1]
    cyy = cov[:, 1, 1] + 0.3

    det = (cxx * cyy - cxy * cxy).clamp_min(1.0e-9)
    conic = torch.stack([cyy / det, -cxy / det, cxx / det], dim=-1)

    mid = 0.5 * (cxx + cyy)
    lam = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
    radius = torch.ceil(3.0 * torch.sqrt(lam))

    # NDC -> pixel, matching the rasteriser's ndc2Pix
    proj = camera.full_proj_transform.to(device)
    hom = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=-1) @ proj
    pw = 1.0 / (hom[:, 3] + 1.0e-7)
    px = ((hom[:, 0] * pw) + 1.0) * w * 0.5 - 0.5
    py = ((hom[:, 1] * pw) + 1.0) * h * 0.5 - 0.5
    return {"uv": torch.stack((px, py), dim=-1), "depth": t[:, 2],
            "conic": conic, "radius": radius, "in_front": t[:, 2] > 0.2}


def covariance_from_scale_rotation(scaling: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    from utils.rigid_transform_utils import quaternion_to_matrix

    r = quaternion_to_matrix(torch.nn.functional.normalize(rotation, dim=-1))
    s = torch.diag_embed(scaling)
    m = r @ s
    return m @ m.transpose(-1, -2)


# ---------------------------------------------------------------------------
# sections 16-21: candidate pixel evidence
# ---------------------------------------------------------------------------
def candidate_pixel_evidence(
    *,
    candidate_xyz: torch.Tensor,            # [N, K+1, 3] detached endpoints
    candidate_cov3d: torch.Tensor,          # [N, K+1, 3, 3] detached
    opacity: torch.Tensor,                  # [N, 1]
    cameras,
    target_maps: list[PixelContributorMap],
    final_xyz: torch.Tensor,                # [M, 3]
    bbox_diag: float,
    window: int = 3,
    sigma_z_ratio: float = 0.02,
    delta_z_ratio: float = 0.15,
    lambda_overlap: float = 0.5,
    lambda_xyz: float = 1.0,
    lambda_depth: float = 0.5,
    beta_entropy: float = 1.0,
    target_spread_ratio: float = 0.1,
    gamma_min: float = 1.0e-4,
    collect_sparse: bool = False,
    collect_per_view: bool = False,
    eps: float = 1.0e-8,
) -> PixelCandidateEvidence | tuple:
    """sections 14-21: E_pixel[i,k] and its confidence, for EVERY hypothesis.

    Rather than rendering a counterfactual scene per Gaussian, each candidate's
    projected splat footprint queries the cached final contributor maps -- that
    is what makes evaluating all K+1 hypotheses affordable.
    """
    device = candidate_xyz.device
    n, k_plus_1 = int(candidate_xyz.shape[0]), int(candidate_xyz.shape[1])
    sigma_z = max(float(sigma_z_ratio) * float(bbox_diag), 1.0e-6)
    delta_z = float(delta_z_ratio) * float(bbox_diag)

    num = torch.zeros((n, k_plus_1), device=device)          # S_ik
    gamma_sum = torch.zeros((n, k_plus_1), device=device)
    views = torch.zeros((n, k_plus_1), device=device)
    bary = torch.zeros((n, k_plus_1, 3), device=device)      # sum A * y_j
    bary_sq = torch.zeros((n, k_plus_1), device=device)      # sum A * |y_j|^2
    depth_num = torch.zeros((n, k_plus_1), device=device)
    entropy_num = torch.zeros((n, k_plus_1), device=device)  # sum A log A
    sparse_keys: list[torch.Tensor] = []
    sparse_vals: list[torch.Tensor] = []
    # Per-observation accumulators for the render-response fingerprint (§4).
    pv_num, pv_gamma, pv_bary, pv_barysq, pv_depth, pv_seen = [], [], [], [], [], []

    offsets = torch.arange(-int(window), int(window) + 1, device=device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)
    p_window = int(dx.numel())

    for cam_idx, camera in enumerate(cameras):
        cmap = target_maps[cam_idx].to(device)
        h, w, l = cmap.ids.shape
        seen_any = torch.zeros((n, k_plus_1), dtype=torch.bool, device=device)
        for k in range(k_plus_1):
            proj = project_gaussians(camera, candidate_xyz[:, k], candidate_cov3d[:, k])
            uv, zc, conic = proj["uv"], proj["depth"], proj["conic"]
            valid_center = proj["in_front"]

            px = (uv[:, 0].round().long()[:, None] + dx[None, :])
            py = (uv[:, 1].round().long()[:, None] + dy[None, :])
            inside = (px >= 0) & (px < w) & (py >= 0) & (py < h) & valid_center[:, None]
            px = px.clamp(0, w - 1)
            py = py.clamp(0, h - 1)

            # section 16: 2D splat kernel weight, before scene transmittance.
            ddx = px.float() - uv[:, 0:1]
            ddy = py.float() - uv[:, 1:2]
            power = -0.5 * (conic[:, 0:1] * ddx * ddx + conic[:, 2:3] * ddy * ddy) - conic[:, 1:2] * ddx * ddy
            gamma = torch.exp(power.clamp(max=0.0)) * opacity.reshape(-1, 1)
            gamma = torch.where(inside & (gamma > gamma_min), gamma, torch.zeros_like(gamma))
            gamma_sum[:, k] += gamma.sum(dim=1)

            flat = (py * w + px).reshape(-1)                              # [N*P]
            ids = cmap.ids.reshape(-1, l).index_select(0, flat)           # [N*P, L]
            rho = cmap.weights.reshape(-1, l).index_select(0, flat)
            zt = cmap.depths.reshape(-1, l).index_select(0, flat)

            gam = gamma.reshape(-1, 1)                                    # [N*P,1]
            zs = zc.reshape(-1, 1).expand(n, p_window).reshape(-1, 1)     # candidate depth
            dz = zs - zt
            # section 9/17: depth compatibility, with a hard reject for gross mismatch.
            wdep = torch.exp(-(dz * dz) / (2.0 * sigma_z * sigma_z))
            wdep = torch.where(dz.abs() > delta_z, torch.zeros_like(wdep), wdep)
            a = gam * rho * wdep
            a = torch.where(ids >= 0, a, torch.zeros_like(a))             # [N*P, L]

            safe_ids = ids.clamp_min(0)
            yj = final_xyz.index_select(0, safe_ids.reshape(-1)).reshape(-1, l, 3)
            contrib = a.reshape(n, p_window, l)
            num[:, k] += contrib.sum(dim=(1, 2))
            bary[:, k] += (a[..., None] * yj).reshape(n, p_window, l, 3).sum(dim=(1, 2))
            bary_sq[:, k] += (a * yj.square().sum(-1)).reshape(n, p_window, l).sum(dim=(1, 2))
            depth_num[:, k] += (a * dz * dz).reshape(n, p_window, l).sum(dim=(1, 2))
            entropy_num[:, k] += (a * torch.log(a.clamp_min(eps))).reshape(n, p_window, l).sum(dim=(1, 2))
            if collect_sparse:
                flat_a = a.reshape(-1)
                keep = flat_a > 0
                if bool(keep.any()):
                    src = torch.arange(n, device=device)[:, None, None].expand(n, p_window, l).reshape(-1)
                    tgt = safe_ids.reshape(-1)
                    m_targets = int(final_xyz.shape[0])
                    # pack (i, k, j) into one key for a later sparse reduce
                    sparse_keys.append(
                        (src[keep].long() * k_plus_1 + k) * m_targets + tgt[keep].long())
                    sparse_vals.append(flat_a[keep])
            seen_any[:, k] |= contrib.sum(dim=(1, 2)) > 0
        views += seen_any.float()
        if collect_per_view:
            # Snapshot this camera's incremental contribution so the same energy
            # formula can be evaluated per observation below.
            pv_num.append(num.clone()); pv_gamma.append(gamma_sum.clone())
            pv_bary.append(bary.clone()); pv_barysq.append(bary_sq.clone())
            pv_depth.append(depth_num.clone()); pv_seen.append(seen_any.clone())

    # section 18: normalised overlap support.
    overlap = num / (gamma_sum + eps)
    # section 19: target barycentre and spread from Q = A / S.
    mass = num.clamp_min(eps)
    ybar = bary / mass[..., None]
    spread = (bary_sq / mass - ybar.square().sum(-1)).clamp_min(0.0)

    # section 20: the three energy components.
    e_overlap = -torch.log(overlap + eps)
    e_xyz = (candidate_xyz - ybar).square().sum(-1) / (float(bbox_diag) ** 2 + eps)
    e_depth = (depth_num / mass) / (float(bbox_diag) ** 2 + eps)
    # Missing evidence must not read as a perfect 3D/depth match.
    has = num > 0
    e_xyz = torch.where(has, e_xyz, torch.zeros_like(e_xyz))
    e_depth = torch.where(has, e_depth, torch.zeros_like(e_depth))

    # section 45: normalise each component by its own robust scale first.
    def _norm(x):
        valid = x[has]
        med = valid.median() if valid.numel() else x.new_tensor(1.0)
        return x / (med.abs() + eps)

    energy = (
        float(lambda_overlap) * _norm(e_overlap)
        + float(lambda_xyz) * _norm(e_xyz)
        + float(lambda_depth) * _norm(e_depth)
    )

    # section 21: confidence from views, overlap, match entropy and target spread.
    q_entropy = (torch.log(mass) - entropy_num / mass).clamp_min(0.0)
    c_views = (views / max(len(cameras), 1)).clamp(0.0, 1.0)
    conf = (
        c_views
        * overlap.clamp(0.0, 1.0)
        * torch.exp(-float(beta_entropy) * q_entropy)
        * torch.exp(-spread / ((float(target_spread_ratio) * float(bbox_diag)) ** 2 + eps))
    )
    conf = torch.where(has, conf, torch.zeros_like(conf)).clamp(0.0, 1.0)

    per_view_energy = per_view_valid = None
    if collect_per_view and pv_num:
        # Per-observation energies use the SAME robust normalisation scales as the
        # aggregate, so fingerprint components stay commensurate across views.
        def _scale(x):
            valid = x[has]
            return (valid.median() if valid.numel() else x.new_tensor(1.0)).abs() + eps
        s_ov, s_xyz, s_dep = _scale(e_overlap), _scale(e_xyz), _scale(e_depth)
        energies, valids = [], []
        prev = None
        for idx in range(len(pv_num)):
            base = (torch.zeros_like(pv_num[0]) if prev is None else prev)
            d_num = pv_num[idx] - base
            d_gamma = pv_gamma[idx] - (torch.zeros_like(pv_gamma[0]) if idx == 0 else pv_gamma[idx - 1])
            d_bary = pv_bary[idx] - (torch.zeros_like(pv_bary[0]) if idx == 0 else pv_bary[idx - 1])
            d_bsq = pv_barysq[idx] - (torch.zeros_like(pv_barysq[0]) if idx == 0 else pv_barysq[idx - 1])
            d_dep = pv_depth[idx] - (torch.zeros_like(pv_depth[0]) if idx == 0 else pv_depth[idx - 1])
            prev = pv_num[idx]
            v_has = d_num > 0
            v_mass = d_num.clamp_min(eps)
            v_overlap = d_num / (d_gamma + eps)
            v_ybar = d_bary / v_mass[..., None]
            v_e_ov = -torch.log(v_overlap + eps)
            v_e_xyz = (candidate_xyz - v_ybar).square().sum(-1) / (float(bbox_diag) ** 2 + eps)
            v_e_dep = (d_dep / v_mass) / (float(bbox_diag) ** 2 + eps)
            v_e_xyz = torch.where(v_has, v_e_xyz, torch.zeros_like(v_e_xyz))
            v_e_dep = torch.where(v_has, v_e_dep, torch.zeros_like(v_e_dep))
            v_energy = (float(lambda_overlap) * v_e_ov / s_ov
                        + float(lambda_xyz) * v_e_xyz / s_xyz
                        + float(lambda_depth) * v_e_dep / s_dep)
            energies.append(torch.where(v_has, v_energy, torch.zeros_like(v_energy)))
            valids.append(v_has)
        per_view_energy = torch.stack(energies, dim=-1).detach()
        per_view_valid = torch.stack(valids, dim=-1).detach()

    evidence = PixelCandidateEvidence(
        energy=energy.detach(), confidence=conf.detach(), overlap=overlap.detach(),
        xyz_energy=e_xyz.detach(), depth_energy=e_depth.detach(), num_views=views.detach(),
        per_view_energy=per_view_energy, per_view_valid=per_view_valid,
    )
    if not collect_sparse:
        return evidence
    m_targets = int(final_xyz.shape[0])
    if sparse_keys:
        key = torch.cat(sparse_keys)
        val = torch.cat(sparse_vals)
        uniq, inverse = torch.unique(key, return_inverse=True)
        acc = torch.zeros(uniq.numel(), dtype=val.dtype, device=val.device).scatter_add_(0, inverse, val)
        ik = uniq // m_targets
        sparse = {"source": (ik // k_plus_1).detach(), "klass": (ik % k_plus_1).detach(),
                  "target": (uniq % m_targets).detach(), "weight": acc.detach(),
                  "mass": mass.detach()}
    else:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        sparse = {"source": empty, "klass": empty, "target": empty,
                  "weight": torch.zeros(0, device=device), "mass": mass.detach()}
    return evidence, sparse


# ---------------------------------------------------------------------------
# sections 22-23: the outer label losses
# ---------------------------------------------------------------------------
def pixel_candidate_loss(p_soft: torch.Tensor, evidence: PixelCandidateEvidence) -> torch.Tensor:
    """section 22: L = (1/N) sum_i sum_k p_ik c_ik E_ik."""
    e = evidence.energy.to(p_soft)
    c = evidence.confidence.to(p_soft)
    return (p_soft * c * e).sum(dim=1).mean()


def pixel_margin_loss(
    p_soft: torch.Tensor,
    evidence: PixelCandidateEvidence,
    *,
    margin: float = 0.05,
    neutral: float | None = None,
) -> torch.Tensor:
    """section 23: literal l != k margin over confidence-blended energies.

        E~_ik = c_ik E_ik + (1 - c_ik) E_neutral

    so missing pixel evidence is neutral rather than strong evidence either way.
    """
    e = evidence.energy.to(p_soft)
    c = evidence.confidence.to(p_soft)
    ref = e.median().detach() if neutral is None else torch.as_tensor(float(neutral), device=e.device)
    blended = c * e + (1.0 - c) * ref
    diff = blended[:, :, None] - blended[:, None, :]
    k = int(e.shape[1])
    off = ~torch.eye(k, dtype=torch.bool, device=e.device)
    hinge = torch.relu(float(margin) + diff) * off[None]
    return (p_soft * hinge.sum(dim=2)).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# sections 9-13: current-hard sparse affinity and attraction
# ---------------------------------------------------------------------------
def accumulate_pair_affinity(
    source_maps: list[PixelContributorMap],
    target_maps: list[PixelContributorMap],
    *,
    num_target: int,
    sigma_z: float,
    delta_z: float | None = None,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """sections 9-11: sparse COO accumulation of A_ij over all final cameras.

    Never builds a dense N x M matrix; pairs are packed into one key and reduced.
    """
    keys, vals = [], []
    # Cached target maps come back from disk on CPU while source maps are freshly
    # rendered on the GPU; align them before any pairwise op.
    device = source_maps[0].ids.device if source_maps else torch.device("cpu")
    for smap, tmap in zip(source_maps, target_maps):
        smap = smap.to(device)
        tmap = tmap.to(device)
        sid, srho, sz = smap.ids.reshape(-1, smap.ids.shape[-1]), smap.weights.reshape(-1, smap.ids.shape[-1]), smap.depths.reshape(-1, smap.ids.shape[-1])
        tid, trho, tz = tmap.ids.reshape(-1, tmap.ids.shape[-1]), tmap.weights.reshape(-1, tmap.ids.shape[-1]), tmap.depths.reshape(-1, tmap.ids.shape[-1])
        # all (source slot, target slot) pairs at the same pixel
        si = sid[:, :, None].expand(-1, -1, tid.shape[1]).reshape(-1)
        tj = tid[:, None, :].expand(-1, sid.shape[1], -1).reshape(-1)
        a = (srho[:, :, None] * trho[:, None, :]).reshape(-1)
        dz = (sz[:, :, None] - tz[:, None, :]).reshape(-1)
        wdep = torch.exp(-(dz * dz) / (2.0 * sigma_z * sigma_z))
        if delta_z is not None:
            wdep = torch.where(dz.abs() > float(delta_z), torch.zeros_like(wdep), wdep)
        a = a * wdep
        keep = (si >= 0) & (tj >= 0) & (a > 0)
        if not bool(keep.any()):
            continue
        keys.append(si[keep].long() * int(num_target) + tj[keep].long())
        vals.append(a[keep])
    if not keys:
        # Stay on the render device: when a diverged transport puts every source
        # splat off the target footprint, no pair survives and the caller still
        # index_selects CUDA geometry with these ids.
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, torch.zeros(0, device=device)
    key = torch.cat(keys)
    val = torch.cat(vals)
    uniq, inverse = torch.unique(key, return_inverse=True)
    acc = torch.zeros(uniq.numel(), dtype=val.dtype, device=val.device).scatter_add_(0, inverse, val)
    return (uniq // int(num_target)), (uniq % int(num_target)), acc


def normalized_match_confidence(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_source: int,
    num_target: int,
    eps: float = 1.0e-8,
) -> dict:
    """section 12: source-, target- and symmetrically normalised confidences."""
    src_sum = torch.zeros(num_source, dtype=weights.dtype, device=weights.device).scatter_add_(0, source_ids, weights)
    tgt_sum = torch.zeros(num_target, dtype=weights.dtype, device=weights.device).scatter_add_(0, target_ids, weights)
    p_j_given_i = weights / (src_sum.index_select(0, source_ids) + eps)
    p_i_given_j = weights / (tgt_sum.index_select(0, target_ids) + eps)
    symmetric = weights / torch.sqrt(
        (src_sum.index_select(0, source_ids) + eps) * (tgt_sum.index_select(0, target_ids) + eps)
    )
    return {"p_j_given_i": p_j_given_i, "p_i_given_j": p_i_given_j,
            "symmetric": symmetric, "source_mass": src_sum, "target_mass": tgt_sum}


def current_pixel_attraction_loss(
    current_xyz: torch.Tensor,
    final_xyz: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    bbox_diag: float,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """section 13: pull the current hard endpoint toward its target barycentre."""
    n = int(current_xyz.shape[0])
    mass = torch.zeros(n, dtype=weights.dtype, device=weights.device).scatter_add_(0, source_ids, weights)
    bary = torch.zeros((n, 3), dtype=weights.dtype, device=weights.device)
    bary.index_add_(0, source_ids, weights[:, None] * final_xyz.index_select(0, target_ids))
    ybar = bary / (mass[:, None] + eps)
    resid = (current_xyz - ybar).square().sum(-1)
    loss = (mass * resid).sum() / ((float(bbox_diag) ** 2) * mass.sum().clamp_min(eps))
    return loss, mass


# ---------------------------------------------------------------------------
# section 28: symmetric target coverage
# ---------------------------------------------------------------------------
def target_coverage_loss(
    p_soft: torch.Tensor,
    q_source: torch.Tensor,
    q_target: torch.Tensor,
    q_weights: torch.Tensor,
    target_support: torch.Tensor,
    *,
    num_target: int,
    q_class: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """section 28: weak L1 between predicted source mass on targets and their support.

    Deliberately weak and normalised: the two 3DGS models have different Gaussian
    counts and densities, so exact one-to-one correspondence is not forced.
    """
    if q_class is not None and q_source.numel():
        # r_j = sum_i sum_k p_ik Q_ikj, with Q from the real sparse affinity.
        flat = q_source.long() * int(p_soft.shape[1]) + q_class.long()
        contrib = p_soft.reshape(-1).index_select(0, flat)
    elif q_source.numel():
        contrib = p_soft.reshape(-1).index_select(0, q_source)
    else:
        contrib = q_weights
    r = torch.zeros(num_target, dtype=p_soft.dtype, device=p_soft.device)
    r = r.index_add_(0, q_target, contrib * q_weights)
    t = target_support.to(p_soft)
    return ((r / (r.sum() + eps)) - (t / (t.sum() + eps))).abs().sum()
