"""Phase 4B, step 1: unexplained-foreground maps of the Phase-4A scene at EVERY (camera, frame) (spec 5.1).

Pixel codes (uint8):
  0 EXPLAINED          the model accounts for the observed foreground
  1 COVERAGE_GAP       mask = 1, valid depth, rendered alpha < alpha_cov
  2 FRONT_SURFACE      observed depth nearer than the prediction by > tol (new surface OR pose / calibration error)
  3 BEHIND_PREDICTION  observed surface behind the prediction (a wrong occluder competes: never a birth source)
  4 DEPTH_INVALID      mask = 1 without valid depth (RGB / mask still constrain; no 3D proposal)
  5 MIXED_EDGE         within edge_px of the silhouette or of an observed depth jump (no naive fusion)
  6 BACKGROUND         mask = 0 (kept as negative evidence for contradiction checks)
OUT_OF_VIEW / OCCLUDED are not pixel states: a missing observation is not negative evidence; 4C only scores
pixels a camera actually saw.

Thresholds are declared config, not calibrated: tol = max(depth_tol, k_sigma * sigma_D(d)) with sigma_D from
the sensor model when given (a + b d^2), else depth_tol alone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

CODES = {"EXPLAINED": 0, "COVERAGE_GAP": 1, "FRONT_SURFACE": 2, "BEHIND_PREDICTION": 3, "DEPTH_INVALID": 4,
         "MIXED_EDGE": 5, "BACKGROUND": 6}
BIRTH_SOURCES = (CODES["COVERAGE_GAP"], CODES["FRONT_SURFACE"])


@dataclass
class DiscoveryConfig:
    alpha_cov: float = 0.5
    depth_tol: float = 0.01          # m
    k_sigma: float = 3.0
    sigma_a: float = 0.0             # sensor sd model sigma_D(d) = sigma_a + sigma_b d^2 (0, 0 = not given)
    sigma_b: float = 0.0
    edge_px: int = 2
    depth_jump: float = 0.03         # m, |second difference| of observed depth = discontinuity (a planar slope
                                     # has zero second difference, so oblique surfaces are not edges)

    def to_dict(self):
        return asdict(self)


def _dilate(m, px):
    if px <= 0:
        return m
    return F.max_pool2d(m.float()[None, None], 2 * px + 1, 1, px)[0, 0] > 0.5


def edge_band(mask: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor, cfg: DiscoveryConfig):
    sil = _dilate(mask, cfg.edge_px) & _dilate(~mask, cfg.edge_px)
    d = torch.where(valid, depth, torch.zeros_like(depth))
    jump = torch.zeros_like(mask)
    sx = (d[:, 2:] - 2 * d[:, 1:-1] + d[:, :-2]).abs() > cfg.depth_jump
    sy = (d[2:, :] - 2 * d[1:-1, :] + d[:-2, :]).abs() > cfg.depth_jump
    vx = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    vy = valid[2:, :] & valid[1:-1, :] & valid[:-2, :]
    jump[:, 1:-1] |= sx & vx
    jump[1:-1, :] |= sy & vy
    # a valid / invalid depth transition inside the mask is a depth edge too
    hole = mask & ~valid
    return sil | _dilate(jump | (_dilate(hole, 1) & valid), cfg.edge_px)


def classify_view(out: dict, view, cfg: DiscoveryConfig) -> torch.Tensor:
    A, Dh = out["alpha"][0], out["depth"][0]
    M, V, D = view.mask, view.depth_valid, view.depth
    tol = torch.full_like(D, cfg.depth_tol)
    if cfg.sigma_a > 0 or cfg.sigma_b > 0:
        tol = torch.maximum(tol, cfg.k_sigma * (cfg.sigma_a + cfg.sigma_b * D * D))
    code = torch.full(M.shape, CODES["EXPLAINED"], dtype=torch.uint8, device=M.device)
    code[~M] = CODES["BACKGROUND"]
    code[M & ~V] = CODES["DEPTH_INVALID"]
    cov = A >= cfg.alpha_cov
    code[M & V & ~cov] = CODES["COVERAGE_GAP"]
    code[M & V & cov & (D < Dh - tol)] = CODES["FRONT_SURFACE"]
    code[M & V & cov & (D > Dh + tol)] = CODES["BEHIND_PREDICTION"]
    code[M & V & edge_band(M, D, V, cfg)] = CODES["MIXED_EDGE"]
    return code


def discovery_maps(scene, moving: torch.Tensor, cfg: DiscoveryConfig):
    """Code map per view (CPU uint8) and per-view counts."""
    maps, stats = [], []
    for vr in scene.views:
        v = scene.view(vr)
        code = classify_view(scene.render(vr, moving), v, cfg)
        maps.append(code.cpu())
        cnt = torch.bincount(code.reshape(-1).long(), minlength=len(CODES)).cpu().numpy()
        stats.append({"camera": vr.cam.name, "frame": int(vr.frame), **{k: int(cnt[i]) for k, i in CODES.items()}})
    return maps, stats
