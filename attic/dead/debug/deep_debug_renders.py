"""Sections 32, 33, 47, 48, 49: the fixed-camera render sequence on disk.

CODEX_DEEP_STAGEWISE_DEBUG_AND_THEORY_AUDIT.md.

Images, as files, under

    renders/camera_<id>/frame_XXXX/<stage>_<rgb|depth>.png
                                   <stage>_<rgb|depth>_residual.png
    renders/camera_<id>/sequence_manifest.json
    renders/strips/

Section 49 is the rule everything here obeys: every stage is rendered from the
SAME camera, the same intrinsics and extrinsics, the same frame, the same crop
and the same RGB and depth normalisation. The only thing allowed to differ
between two images is the motion model. Depth in particular is normalised once
per camera over the whole sequence from the GT range, so a depth image at
frame 40 is comparable with one at frame 0 -- per-image autoscaling would make
every stage look equally good.

Stages that do not exist for a run are written into the manifest as
unavailable with a reason (section 47); nothing is fabricated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

STAGES = ("gt", "canonical", "phase1_free", "phase2_rigid", "P", "R",
          "refined")


def _png(path: Path, arr: np.ndarray):
    """Write an [H,W] or [H,W,3] float array in [0,1] as an 8-bit PNG."""
    from PIL import Image
    a = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0)
    a = (a * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path)


def _colormap(x: np.ndarray, name: str = "magma") -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    try:
        cmap = matplotlib.colormaps[name]          # matplotlib >= 3.7
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(name)
    return cmap(np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0))[..., :3]


class DepthNorm:
    """One depth normalisation per camera, fixed over the whole sequence.

    Section 49 forbids renormalising per image: the difference between two
    stages must be the motion model, not the colour map's gain.
    """

    def __init__(self, lo: float, hi: float):
        self.lo, self.hi = float(lo), float(max(hi, lo + 1e-6))

    def __call__(self, d: np.ndarray) -> np.ndarray:
        return (d - self.lo) / (self.hi - self.lo)


def render_state(objective, mu, R, obs_dev, camera_index: int):
    """One stage, one frame, one camera -> (rgb, depth, alpha) on the CPU."""
    lin, _p, _r = objective.linearize_frame(mu, R, obs_dev)
    v = lin.views[camera_index]
    out = (v.rgb.detach().float().cpu(), v.depth.detach().float().cpu(),
           v.alpha.detach().float().cpu())
    del lin
    return out


def write_frame_images(folder: Path, stage: str, rgb, depth, alpha,
                       gt_rgb, gt_depth, gt_valid, dnorm: DepthNorm,
                       min_alpha: float, residual_gain: float) -> dict:
    """All images for one stage at one frame, plus their scalar residuals."""
    folder.mkdir(parents=True, exist_ok=True)
    rec: dict = {}
    rgb_np = rgb.permute(1, 2, 0).numpy()
    _png(folder / f"{stage}_rgb.png", rgb_np)
    rec[f"{stage}_rgb"] = f"{stage}_rgb.png"
    _png(folder / f"{stage}_depth.png",
         _colormap(dnorm(depth.numpy()), "viridis"))
    rec[f"{stage}_depth"] = f"{stage}_depth.png"

    if stage == "gt" or gt_rgb is None:
        return rec
    # Residuals share one fixed gain across stages and frames, so a brighter
    # residual image always means a worse fit.
    res = (gt_rgb - rgb).abs().mean(0).numpy()
    _png(folder / f"{stage}_rgb_residual.png",
         _colormap(res / max(residual_gain, 1e-9), "magma"))
    rec[f"{stage}_rgb_residual"] = f"{stage}_rgb_residual.png"
    rec[f"{stage}_rgb_l1"] = float(res.mean())

    m = gt_valid & (alpha > min_alpha)
    dres = (gt_depth - depth).abs() * m.to(gt_depth.dtype)
    span = max(dnorm.hi - dnorm.lo, 1e-6)
    _png(folder / f"{stage}_depth_residual.png",
         _colormap(dres.numpy() / (0.1 * span), "magma"))
    rec[f"{stage}_depth_residual"] = f"{stage}_depth_residual.png"
    denom = float(m.sum())
    rec[f"{stage}_depth_l1"] = (float(dres.sum()) / denom) if denom else None
    rec[f"{stage}_valid_depth_px"] = int(denom)
    return rec


def keyframe_strip(renders_root: Path, camera_id: int, frames: list,
                   stages: list, out: Path, residual: bool = False):
    """Section 48: GT | static | Phase1 | Phase2 | P | R | refined."""
    from PIL import Image, ImageDraw
    rows = []
    for t in frames:
        folder = renders_root / f"camera_{camera_id:03d}" / f"frame_{t:04d}"
        cells = []
        for st in stages:
            name = f"{st}_rgb_residual.png" if (residual and st != "gt") \
                else f"{st}_rgb.png"
            p = folder / name
            cells.append(Image.open(p).convert("RGB") if p.is_file() else None)
        rows.append((t, cells))
    live = [c for _t, cs in rows for c in cs if c is not None]
    if not live:
        return None
    w, h = live[0].size
    sc = min(1.0, 320.0 / w)
    w, h = int(w * sc), int(h * sc)
    pad, label = 2, 16
    W = len(stages) * (w + pad) + pad
    H = len(rows) * (h + pad + label) + pad
    sheet = Image.new("RGB", (W, H), (17, 19, 26))
    d = ImageDraw.Draw(sheet)
    for ri, (t, cells) in enumerate(rows):
        y = pad + ri * (h + pad + label)
        d.text((pad, y), f"frame {t}", fill=(190, 195, 210))
        for ci, c in enumerate(cells):
            x = pad + ci * (w + pad)
            if c is not None:
                sheet.paste(c.resize((w, h)), (x, y + label))
            else:
                d.rectangle([x, y + label, x + w, y + label + h],
                            outline=(60, 65, 80))
                d.text((x + 6, y + label + 6), "n/a", fill=(110, 118, 135))
            if ri == 0:
                d.text((x, y - 0), stages[ci], fill=(140, 150, 170))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
