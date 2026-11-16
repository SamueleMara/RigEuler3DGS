"""Conservative per-Gaussian tile footprints for exact ROI scoring (directive 2026-11-11, 4.2).

The production kernel (forward.cu preprocessCUDA + duplicateWithKeys + renderCUDA) lets Gaussian i touch a
pixel only if the pixel's 16 x 16 tile lies in

    rect_i = [ trunc((p_x - r_i) / 16), trunc((p_x + r_i + 15) / 16) ) x [ same for y ], clamped to the grid

with r_i = ceil(3 sqrt(lambda_max(Sigma'_i))) the kernel's own radius (0 when frustum-culled: view z <= 0.2,
or a degenerate conic) and p the projected mean ((ndc + 1) W - 1) / 2. Tiles outside rect_i never list
Gaussian i, so their per-tile sorted contributor list, front-to-back compositing and early termination are
untouched by i: every pixel outside rect_i is BITWISE identical with or without i, in any pose of i.
(Sorting is a stable radix sort on (tile, depth) keys; removing/moving i changes neither the keys nor the
relative order of the other entries of any other tile.)

r_i is taken from the CUDA kernel itself (a render's `radii`). p is recomputed in torch with the kernel's
formula; its float rounding can differ from the device's by ~1e-4 px, which can move a rect edge by one
tile, so every rect is PADDED by one tile on each side (conservative superset). Alpha-threshold (1/255)
pixels inside the rect that the Gaussian does not reach only make the region larger, never smaller.
"""
from __future__ import annotations

import numpy as np
import torch

BLOCK = 16
EMPTY = np.array([0, 0, 0, 0], np.int16)


def tile_grid(W, H):
    return (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK


def project_means(cam, mu: torch.Tensor):
    """Kernel projection (row-vector full_proj_transform), float32."""
    P = cam.P.to(mu.device, torch.float32)
    ph = mu.float() @ P[:3] + P[3]
    w = 1.0 / (ph[:, 3] + 1e-7)
    px = ((ph[:, 0] * w + 1.0) * cam.W - 1.0) * 0.5
    py = ((ph[:, 1] * w + 1.0) * cam.H - 1.0) * 0.5
    return px, py


def tile_rects(cam, mu: torch.Tensor, radii: torch.Tensor, pad: int = 1) -> torch.Tensor:
    """[N,4] int16 (tx0, ty0, tx1, ty1), half-open tile ranges, padded by `pad` tiles; all-zero when empty."""
    gx, gy = tile_grid(cam.W, cam.H)
    px, py = project_means(cam, mu)
    r = radii.float()
    x0 = torch.clamp(torch.trunc((px - r) / BLOCK), 0, gx) - pad
    x1 = torch.clamp(torch.trunc((px + r + BLOCK - 1) / BLOCK), 0, gx) + pad
    y0 = torch.clamp(torch.trunc((py - r) / BLOCK), 0, gy) - pad
    y1 = torch.clamp(torch.trunc((py + r + BLOCK - 1) / BLOCK), 0, gy) + pad
    x0, x1 = x0.clamp(0, gx), x1.clamp(0, gx)
    y0, y1 = y0.clamp(0, gy), y1.clamp(0, gy)
    out = torch.stack([x0, y0, x1, y1], 1)
    # conservative emptiness: only a culled Gaussian (kernel radius 0) or a non-finite projection touches no
    # tile. A clamped-empty rect is NOT treated as empty here: a one-tile projection error could otherwise turn
    # a just-on-screen Gaussian into an off-screen one; the padded rect then covers the border tile.
    empty = (radii <= 0) | ~torch.isfinite(px) | ~torch.isfinite(py)
    out[empty] = 0
    return out.to(torch.int16)


def aggregate(rects: torch.Tensor, member_idx: torch.Tensor, group_of: torch.Tensor, G: int) -> torch.Tensor:
    """Union bounding rect per group. rects [K,N,4] (K views or cameras), member_idx / group_of [m] -> [K,G,4]
    int16; an all-empty group gets (0,0,0,0)."""
    K = rects.shape[0]
    r = rects[:, member_idx].long()                                     # [K,m,4]
    nonempty = (r[..., 2] > r[..., 0]) & (r[..., 3] > r[..., 1])
    big = 1 << 20
    lo = torch.where(nonempty[..., None], r[..., :2], torch.full_like(r[..., :2], big))
    hi = torch.where(nonempty[..., None], r[..., 2:], torch.full_like(r[..., 2:], -1))
    idx = group_of.long()[None, :, None].expand(K, -1, 2)
    out_lo = torch.full((K, G, 2), big, dtype=torch.long, device=rects.device).scatter_reduce(1, idx, lo, "amin")
    out_hi = torch.full((K, G, 2), -1, dtype=torch.long, device=rects.device).scatter_reduce(1, idx, hi, "amax")
    out = torch.cat([out_lo, out_hi], -1)
    empty = (out[..., 2] <= out[..., 0]) | (out[..., 3] <= out[..., 1])
    out[empty] = 0
    return out.to(torch.int16)


def rect_nonempty(r) -> np.ndarray:
    r = np.asarray(r)
    return (r[..., 2] > r[..., 0]) & (r[..., 3] > r[..., 1])


def pixel_box(rect, W, H, halo: int):
    """Tile rect -> pixel box (y0, y1, x0, x1), dilated by `halo` pixels, clipped to the image."""
    x0, y0, x1, y1 = (int(v) for v in rect)
    return (max(0, y0 * BLOCK - halo), min(H, y1 * BLOCK + halo), max(0, x0 * BLOCK - halo), min(W, x1 * BLOCK + halo))


def boxes_for(rect_a, rect_b, W, H, halo: int):
    """Disjoint output boxes covering the dilated union of two rects (merged into one box if they meet)."""
    bs = [pixel_box(r, W, H, halo) for r in (rect_a, rect_b) if rect_nonempty(r)]
    if len(bs) == 2:
        (a0, a1, a2, a3), (b0, b1, b2, b3) = bs
        if a0 < b1 and b0 < a1 and a2 < b3 and b2 < a3:      # overlap -> bounding box
            bs = [(min(a0, b0), max(a1, b1), min(a2, b2), max(a3, b3))]
    return bs


def rects_overlap(a: np.ndarray, b: np.ndarray, dil: int = 0) -> np.ndarray:
    """Elementwise overlap of tile rects a [...,4] and b [...,4] (b dilated by `dil` tiles); empty never overlaps."""
    a = np.asarray(a, np.int32); b = np.asarray(b, np.int32)
    ne = rect_nonempty(a) & rect_nonempty(b)
    return ne & (a[..., 0] < b[..., 2] + dil) & (b[..., 0] - dil < a[..., 2]) & \
        (a[..., 1] < b[..., 3] + dil) & (b[..., 1] - dil < a[..., 3])
