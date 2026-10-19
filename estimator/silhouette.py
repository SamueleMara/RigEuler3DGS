"""Object masks in the RGB-D measurement: where the object is absent, it must not render.

Why this is needed on real captures. The synthetic exports have a black
background without depth, which is exactly what the renderer produces where no
Gaussian is, so the full-frame RGB term already acts as a silhouette
constraint. Real captures have a textured background (mean RGB ~0.46) with
depth on ~50% of the frame and foreground occluders (the robot/hand, ~11-14% of
the frame): the RGB/depth terms there either attract object Gaussians to the
background or say nothing about a Gaussian that has left the object.

Why an RGB "background" target is not enough inside an EKF. Appearance and
opacity are frozen, so a Gaussian outside the object cannot fade the way static
carving lets it; it can only move.
  * Colour camouflage: against a fixed background colour, a Gaussian of that
    colour leaves no residual and its linearization sees nothing.
  * Flat landscape: even with a colour-free alpha residual A_p (target 0), a
    footprint lying wholly in the "absent" region has zero gradient -- sliding
    it around inside that region does not change sum A_p^2 -- so the IEKF has
    no direction and the Gaussian stays stuck outside the mask.

The measurement used here.
  * Object views: RGB and depth only inside the mask (the recorded RGB-D there
    is the object), zero precision outside.
  * Silhouette views: the SAME Gaussians rendered all black over a white
    background, so the colour is C = 1 - A exactly and the existing CUDA
    JVP/VJP give dA. Target C = 1 (A = 0) on "absent" pixels with precision
    (w / sigma)^2 on one channel, w = 1 + (min(D, D_max) - band) / D_0 beyond
    the band (0 inside it), D the distance in pixels to the nearest "allowed"
    pixel: full strength at the silhouette edge, and a slope with distance so
    the weighted residual w*A has a gradient everywhere outside the mask,
    pointing back to it.
  * Allowed = mask OR occluder, occluder = outside the mask where the recorded
    surface is in front of the rendered object (the robot in front of the door):
    the object may be hidden there, so nothing is claimed.

Inactive unless activated (`set_active`), so every existing run is unchanged.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

_ACTIVE = None


def set_active(sil) -> None:
    global _ACTIVE
    _ACTIVE = sil


def get_active():
    return _ACTIVE


def load_object_masks(source: Path, cam_names, original_frames, height: int, width: int):
    """mask/<cam>/frame_NNNNNN.png per (processing-order) frame and camera, as
    bool [H,W] CPU tensors. Missing masks are an error, not a silent "all in"."""
    from PIL import Image
    out = []
    for f in original_frames:
        row = []
        for c in cam_names:
            cands = [Path(source) / "mask" / c / f"frame_{int(f):06d}{e}" for e in (".png", ".jpg", ".jpeg")]
            p = next((q for q in cands if q.is_file()), None)
            if p is None:
                raise SystemExit(f"--object-masks: no mask for {c} frame {int(f)} under {Path(source) / 'mask'}")
            im = Image.open(p).convert("L")
            if im.size != (width, height):
                im = im.resize((width, height), Image.NEAREST)
            row.append(torch.from_numpy(np.asarray(im) > 127))
        out.append(row)
    return out


class ObjectSilhouette:
    def __init__(self, masks, *, d0_px: float = 10.0, dmax_px: float = 100.0, band_px: float = 2.0,
                 sigma: float = 0.1, occluder_tol_m: float = 0.05, min_alpha: float = 0.05):
        self.masks = masks
        self.d0, self.dmax, self.band = float(d0_px), float(dmax_px), float(band_px)
        self.sigma, self.tol, self.min_alpha = float(sigma), float(occluder_tol_m), float(min_alpha)
        self.frame = 0
        self.check = {}          # frame -> silhouette violation (last linearization of that frame)

    def config(self) -> dict:
        return {"d0_px": self.d0, "dmax_px": self.dmax, "band_px": self.band, "sigma": self.sigma,
                "occluder_tol_m": self.tol, "min_alpha": self.min_alpha, "frames": len(self.masks)}

    def set_frame(self, k: int) -> None:
        self.frame = int(k)

    # -- linearization: the silhouette views ----------------------------------
    def augment(self, lin, state, cameras, eps_depth: float = 1e-6, **render_kw):
        """lin with one silhouette view per camera appended (same geometry,
        black Gaussians over white -> C = 1 - A)."""
        from .renderer_bridge import FrameLinearization, build_linearization
        black = dataclasses.replace(state, colors=torch.zeros_like(state.colors))
        white = torch.ones(3, device=state.mu.device, dtype=torch.float32)
        sil = build_linearization(black, cameras, background=white, eps_depth=eps_depth, **render_kw)
        out = FrameLinearization(list(lin.views) + list(sil.views), lin.num_gaussians)
        out.num_object_views = len(lin.views)
        return out

    @staticmethod
    def targets(ncam: int, like_rgb: torch.Tensor, like_depth: torch.Tensor):
        """The observation for the silhouette views: C = 1 (A = 0)."""
        one = torch.ones_like(like_rgb)
        zero = torch.zeros_like(like_depth)
        return [(one, zero)] * ncam

    # -- precisions -----------------------------------------------------------
    def object_gate(self, j: int, device) -> torch.Tensor:
        return self.masks[self.frame][j].to(device)

    def terms(self, j: int, view_obj, view_sil, obs_depth, obs_valid):
        """(p_rgb [3,H,W], p_depth [H,W], r_rgb, r_depth) for silhouette view j."""
        from scipy.ndimage import distance_transform_edt
        dev = view_sil.rgb.device
        M = self.masks[self.frame][j].to(dev)
        d_obs = obs_depth.to(dev)
        valid = obs_valid.to(dev).bool() if obs_valid is not None else d_obs > 0
        occ = (~M) & valid & (view_obj.alpha > self.min_alpha) & (d_obs < view_obj.depth - self.tol)
        allowed = (M | occ).cpu().numpy()
        D = torch.from_numpy(distance_transform_edt(~allowed)).to(dev, torch.float32)
        # full strength from the band outward (as the RGB residual against a black
        # background gives on the synthetic exports, which is what pins the
        # silhouette edge), plus a slope with distance so a footprint wholly
        # outside still has a direction back to the mask
        excess = (D.clamp(max=self.dmax) - self.band)
        w = torch.where(excess > 0, 1.0 + excess / self.d0, torch.zeros_like(excess))
        p_rgb = torch.zeros_like(view_sil.rgb)
        p_rgb[0] = (w / self.sigma) ** 2
        r_rgb = torch.ones_like(view_sil.rgb) - view_sil.rgb          # = A
        A = view_obj.alpha
        absent = w > 0
        tot = float(A.sum())
        self.check.setdefault(self.frame, {})[j] = {
            "alpha_outside_mask_fraction": float(A[absent].sum()) / tot if tot > 0 else 0.0,
            "alpha_outside_weighted_px": float((A * D)[absent].sum()) / tot if tot > 0 else 0.0,
            "occluder_pixels": int(occ.sum())}
        return p_rgb, torch.zeros_like(view_sil.depth), r_rgb, torch.zeros_like(view_sil.depth)

    def summary(self) -> dict:
        rows = {k: {"alpha_outside_mask_fraction": float(np.mean([c["alpha_outside_mask_fraction"] for c in v.values()])),
                    "alpha_outside_weighted_px": float(np.mean([c["alpha_outside_weighted_px"] for c in v.values()])),
                    "occluder_pixels": int(sum(c["occluder_pixels"] for c in v.values()))}
                for k, v in sorted(self.check.items())}
        f = [r["alpha_outside_mask_fraction"] for r in rows.values()]
        return {"config": self.config(), "per_frame": rows,
                "max_alpha_outside_mask_fraction": max(f) if f else None,
                "mean_alpha_outside_mask_fraction": float(np.mean(f)) if f else None,
                "definition": ("share of the rendered object alpha that falls on 'absent' pixels "
                               "(outside mask and occluders by more than band_px), at the last "
                               "linearization of each frame; GT-free")}


def add_cli(ap) -> None:
    """The object-mask options, shared by the Phase 1/2/3 runners."""
    ap.add_argument("--object-masks", action="store_true",
                    help="use <source>/mask/<cam>/frame_NNNNNN.png in the RGB-D measurement: object "
                         "RGB-D inside the mask, a distance-weighted silhouette residual outside it, "
                         "occluders (recorded surface in front of the render) allowed")
    ap.add_argument("--silhouette-d0-px", type=float, default=10.0,
                    help="distance at which one silhouette pixel weighs like one RGB pixel")
    ap.add_argument("--silhouette-dmax-px", type=float, default=100.0)
    ap.add_argument("--silhouette-band-px", type=float, default=2.0,
                    help="mask-boundary band with no silhouette penalty (mask uncertainty)")
    ap.add_argument("--silhouette-occluder-tol-m", type=float, default=0.05)


def activate_from_args(args, source, frame_ids, frames_raw):
    """Build and activate the silhouette model for a run, or return None."""
    if not getattr(args, "object_masks", False):
        set_active(None)
        return None
    cams = list(frame_ids.get("cameras") or [])
    frames = [int(f["original_frame_index"]) if isinstance(f, dict) and "original_frame_index" in f
              else int(f["dataset_frame_index"]) if isinstance(f, dict) else int(f)
              for f in frame_ids["frames"]]
    rgb0 = frames_raw[0][0][0][0]
    H, W = int(rgb0.shape[-2]), int(rgb0.shape[-1])
    ncam = len(frames_raw[0][0])
    masks = load_object_masks(Path(source), cams[:ncam], frames, H, W)
    sil = ObjectSilhouette(masks, d0_px=args.silhouette_d0_px, dmax_px=args.silhouette_dmax_px,
                           band_px=args.silhouette_band_px, occluder_tol_m=args.silhouette_occluder_tol_m)
    set_active(sil)
    cover = float(np.mean([m.float().mean() for row in masks for m in row]))
    print(f"  object masks: {len(masks)} frames x {ncam} cameras, mean object coverage {cover:.1%}; "
          f"silhouette d0 {sil.d0:g} px, band {sil.band:g} px")
    return sil


def silhouette_closures(sil, truth, mu0, R0, cameras, cfg, render_state, build_linearization,
                        base_prec, obs_all, cam_ids):
    """(linearize, precision_fn wrapper, obs extender) for the joint/rigid runners.

    linearize(st, frame) renders the state and, with a silhouette active, appends
    its silhouette views; precision(lin, frame, base) gates the object precisions
    by the mask and adds the silhouette terms; extend(obs_pairs) appends the
    A = 0 targets to every frame's observation list."""
    def linearize(st, frame):
        mu, Rg = st.gaussians(frame, mu0, R0)
        lin = render_state(truth, mu, Rg, cameras, cfg, build_linearization)
        if sil is None:
            return lin
        sil.set_frame(frame)
        kw = {"mu": mu.to(truth.mu.dtype)}
        if Rg is not None:
            kw["R"] = Rg.to(truth.R.dtype)
        return sil.augment(lin, dataclasses.replace(truth, **kw), cameras, eps_depth=cfg.eps_depth)

    def precision(lin, frame):
        base = base_prec(lin, frame)
        if sil is None:
            return base
        n = len(cam_ids)
        out = []
        for j, (pc, pd) in enumerate(base):
            M = sil.object_gate(j, pc.device).to(pc.dtype)
            out.append((pc * M[None], pd * M))
        for j, c in enumerate(cam_ids):
            ob = obs_all[frame][c]
            p_rgb, p_d, _r, _rd = sil.terms(j, lin.views[j], lin.views[n + j], ob[1],
                                             ob[2] if len(ob) > 2 else None)
            out.append((p_rgb, p_d))
        return out

    def extend(obs_pairs):
        if sil is None:
            return obs_pairs
        out = []
        for ob in obs_pairs:
            ref = next(o for o in ob if o is not None)
            out.append(list(ob) + sil.targets(len(cam_ids), ref[0], ref[1]))
        return out
    return linearize, precision, extend
