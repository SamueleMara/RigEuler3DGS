"""Phase-4 physical image objective with an EXACT additive per-pixel decomposition (directive 2026-11-11, 4.1-4.2).

For a view v = (camera, frame) of size H x W, observed RGB I, binary object mask M, metric axial depth D and
the fixed observed-valid depth support V = erode(M & finite(D) & D > 0) (independent of any hypothesis),

    L_v = sum_p  e_l1(p) + e_ssim(p) + e_mask(p) + e_depth(p)

    e_l1(p)    = lam_rgb (1 - lam_ssim) sum_c |M (I_hat_c - I_c)| / (3 H W)
    e_ssim(p)  = lam_rgb lam_ssim (1 - mean_c SSIM_c(p)) / (H W)          SSIM of M*I_hat vs M*I, 11x11
                                                                         Gaussian window (sigma 1.5), zero
                                                                         padding, C1 = 0.01^2, C2 = 0.03^2
    e_mask(p)  = lam_mask |A_hat(p) - M(p)| / (H W)                      every pixel, background included
    e_depth(p) = lam_depth [p in V] pen(p) / max(1, |V|)
    pen(p)     = rho(D_hat(p) - D(p))   if A_hat(p) >= depth_alpha_min,  D_hat = Z_hat / (A_hat + 1e-6)
               = depth_miss_penalty     otherwise (explicit missing-coverage cost, counted)

This is the FreeArt-parity weighting of recovery/rgbd_loss.py (lam_ssim 0.2 as a convex split of the RGB
weight, mask L1 over the whole image) with two declared corrections required for exact model selection:
  * every denominator is a fixed full-view count (3HW, HW, |V|); none depends on the hypothesis, so
    L_v(G_a) - L_v(G_b) is a plain sum of per-pixel differences;
  * a valid observed-depth pixel the render leaves (almost) uncovered costs `depth_miss_penalty` instead of
    being dropped -- a hypothesis cannot avoid depth error by leaving the pixel empty. The penalty value
    (default 0.05 m-equivalent) is DECLARED, not calibrated.
recovery/rgbd_loss.py (Phase 5) is unchanged; its support-normalised depth term is not used here.

SSIM: the window sums are computed as fixed-order shifted sums in float64 (`_box_sums`), so the SSIM value
at a pixel is bitwise identical whether it is evaluated on the full image or on a crop that contains its
11 x 11 input neighbourhood (zeros only beyond the true image border). A change at pixel q alters SSIM
outputs only within the 11 x 11 neighbourhood of q, i.e. within SSIM_R = 5 pixels.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

CHANNELS = ("rgb_l1", "rgb_ssim", "mask", "depth")
SSIM_R = 5
SSIM_C1, SSIM_C2 = 0.01 ** 2, 0.03 ** 2
DEPTH_EPS = 1e-6


@dataclass
class ExactLossConfig:
    lambda_rgb: float = 1.0
    lambda_ssim: float = 0.2
    lambda_depth: float = 1.0
    lambda_mask: float = 1.0
    depth_loss: str = "l1"                # l1 | student_t
    student_t_nu: float = 4.0
    student_t_scale: float = 0.01
    depth_alpha_min: float = 0.01
    depth_miss_penalty: float = 0.05      # declared, uncalibrated (m-equivalent L1 cost of an uncovered valid pixel)
    depth_edge_px: int = 0                # erodes V only (never the silhouette)

    def to_dict(self):
        return asdict(self)


def _gauss1d(dtype, device):
    g = torch.tensor([math.exp(-(x - SSIM_R) ** 2 / (2 * 1.5 ** 2)) for x in range(2 * SSIM_R + 1)], dtype=torch.float64)
    return (g / g.sum()).to(device=device, dtype=dtype)


_G1 = {}


def gauss1d(device):
    k = str(device)
    if k not in _G1:
        _G1[k] = _gauss1d(torch.float64, device).tolist()
    return _G1[k]


def _box_sums(x: torch.Tensor, g: list) -> torch.Tensor:
    """Separable Gaussian window sum by fixed-order shifted adds. x: [C, h + 2R, w + 2R] -> [C, h, w]."""
    R = SSIM_R
    w = x.shape[-1] - 2 * R
    t = x[..., 0:w] * g[0]
    for k in range(1, 2 * R + 1):
        t = t + x[..., k:k + w] * g[k]
    h = t.shape[-2] - 2 * R
    o = t[..., 0:h, :] * g[0]
    for k in range(1, 2 * R + 1):
        o = o + t[..., k:k + h, :] * g[k]
    return o


def _crop_with_halo(img: torch.Tensor, box, H, W):
    """img [C,H,W] -> float64 [C, (y1-y0)+2R, (x1-x0)+2R]; zeros outside the true image (= conv zero padding)."""
    y0, y1, x0, x1 = box
    R = SSIM_R
    ya, yb, xa, xb = max(0, y0 - R), min(H, y1 + R), max(0, x0 - R), min(W, x1 + R)
    c = img[:, ya:yb, xa:xb].double()
    pad = (xa - (x0 - R), (x1 + R) - xb, ya - (y0 - R), (y1 + R) - yb)
    if any(pad):
        c = F.pad(c, pad)
    return c


class ViewTargets:
    """The observation of one view on the GPU plus everything about it that no hypothesis can change."""

    def __init__(self, view, cfg: ExactLossConfig):
        self.cfg = cfg
        self.I = view.rgb.float()
        self.M = view.mask.bool()
        self.Mf = self.M.float()
        self.D = view.depth.float()
        V = view.depth_valid.bool()
        if cfg.depth_edge_px > 0:
            k = 2 * cfg.depth_edge_px + 1
            V = (-F.max_pool2d(-V.float()[None, None], k, 1, cfg.depth_edge_px))[0, 0] > 0.5
        self.V = V
        self.nV = int(V.sum())
        self.H, self.W = self.M.shape
        self.HW = self.H * self.W
        self.MI = self.I * self.Mf
        self.g = gauss1d(self.I.device)
        # target window statistics over the full image (identical crops for every hypothesis)
        full = (0, self.H, 0, self.W)
        y = _crop_with_halo(self.MI, full, self.H, self.W)
        self.mu_y = _box_sums(y, self.g)
        self.yy = _box_sums(y * y, self.g)
        self.w_l1 = cfg.lambda_rgb * (1 - cfg.lambda_ssim) / (3 * self.HW)
        self.w_ssim = cfg.lambda_rgb * cfg.lambda_ssim / self.HW
        self.w_mask = cfg.lambda_mask / self.HW
        self.w_depth = cfg.lambda_depth / max(1, self.nV)

    # ------------------------------------------------------------------ per-pixel maps (float64)
    def pixel_maps(self, out: dict, box=None) -> dict:
        """e_l1, e_mask, e_depth on `box` (y0, y1, x0, x1) of the render `out`; also `uncovered` (bool)."""
        y0, y1, x0, x1 = box if box is not None else (0, self.H, 0, self.W)
        sl = (slice(y0, y1), slice(x0, x1))
        I_hat = out["color"][:, y0:y1, x0:x1].double()
        A = out["alpha"][0][sl].double()
        Z = out["zdepth"][0][sl].double()
        Mf = self.Mf[sl].double()
        e_l1 = self.w_l1 * ((I_hat - self.I[:, y0:y1, x0:x1].double()) * Mf).abs().sum(0)
        e_mask = self.w_mask * (A - Mf).abs()
        V = self.V[sl]
        cov = A >= self.cfg.depth_alpha_min
        Dh = Z / (A + DEPTH_EPS)
        r = Dh - self.D[sl].double()
        if self.cfg.depth_loss == "l1":
            rho = r.abs()
        else:
            nu, s = self.cfg.student_t_nu, self.cfg.student_t_scale
            rho = 0.5 * (nu + 1) * torch.log1p((r / s) ** 2 / nu)
        pen = torch.where(cov, rho, torch.full_like(rho, self.cfg.depth_miss_penalty))
        e_depth = self.w_depth * torch.where(V, pen, torch.zeros_like(pen))
        return {"rgb_l1": e_l1, "mask": e_mask, "depth": e_depth, "uncovered": V & ~cov}

    def ssim_map(self, out: dict, box=None) -> torch.Tensor:
        """e_ssim on `box`; bitwise identical to the same pixels of the full-image map."""
        box = box if box is not None else (0, self.H, 0, self.W)
        y0, y1, x0, x1 = box
        x = _crop_with_halo(out["color"] * self.Mf, box, self.H, self.W)
        y = _crop_with_halo(self.MI, box, self.H, self.W)
        S = _box_sums(torch.cat([x, x * x, x * y], 0), self.g)        # one fixed-order pass, channel-separable
        mu_x, xx, xy = S[0:3], S[3:6], S[6:9]
        mu_y = self.mu_y[:, y0:y1, x0:x1]
        yy = self.yy[:, y0:y1, x0:x1]
        sx, sy, sxy = xx - mu_x * mu_x, yy - mu_y * mu_y, xy - mu_x * mu_y
        s = ((2 * mu_x * mu_y + SSIM_C1) * (2 * sxy + SSIM_C2)) / \
            ((mu_x * mu_x + mu_y * mu_y + SSIM_C1) * (sx + sy + SSIM_C2))
        return self.w_ssim * (1.0 - s.mean(0))

    def full_maps(self, out: dict) -> dict:
        m = self.pixel_maps(out)
        m["rgb_ssim"] = self.ssim_map(out)
        return m

    def totals(self, maps: dict) -> torch.Tensor:
        """[4] float64 per-channel view loss (CHANNELS order)."""
        return torch.stack([maps[c].sum() for c in CHANNELS])


def view_objective(out: dict, tgt: ViewTargets) -> torch.Tensor:
    """Full-image reference: [4] float64 per-channel L_v (sums to the view loss)."""
    return tgt.totals(tgt.full_maps(out))


def box_delta(alt: dict, base: dict, base_maps: dict, tgt: ViewTargets, boxes) -> tuple:
    """Exact L_v(alt) - L_v(base) per channel, given that alt and base differ only on pixels inside the
    union of `boxes` shrunk by SSIM_R (the caller's footprint contract), and the boxes are pairwise disjoint.

    Returns ([4] float64 delta, int changed-pixel count tensor, int newly-uncovered-valid count tensor)."""
    d = torch.zeros(4, dtype=torch.float64, device=tgt.I.device)
    changed = torch.zeros((), dtype=torch.int64, device=tgt.I.device)
    unc = torch.zeros((), dtype=torch.int64, device=tgt.I.device)
    for box in boxes:
        y0, y1, x0, x1 = box
        sl = (slice(y0, y1), slice(x0, x1))
        a = tgt.pixel_maps(alt, box)
        d[0] += (a["rgb_l1"] - base_maps["rgb_l1"][sl]).sum()
        d[2] += (a["mask"] - base_maps["mask"][sl]).sum()
        d[3] += (a["depth"] - base_maps["depth"][sl]).sum()
        d[1] += (tgt.ssim_map(alt, box) - base_maps["rgb_ssim"][sl]).sum()
        ch = (alt["color"][:, y0:y1, x0:x1] != base["color"][:, y0:y1, x0:x1]).any(0) | \
             (alt["alpha"][0][sl] != base["alpha"][0][sl]) | (alt["zdepth"][0][sl] != base["zdepth"][0][sl])
        changed += ch.sum()
        unc += (a["uncovered"] & ~base_maps["uncovered"][sl]).sum()
    return d, changed, unc


def box_delta_weighted(alt: dict, base: dict, base_maps: dict, tgt: ViewTargets, boxes, weight: torch.Tensor) -> torch.Tensor:
    """Like box_delta, restricted to pixels with weight[p] = True (e.g. one data role): per-channel
    sum_p w(p) (e_alt(p) - e_base(p)). SSIM is scored at the weighted CENTRES; with role blocks guarded by
    >= SSIM_R + 1 pixels, every scored window reads only its own role's observations."""
    d = torch.zeros(4, dtype=torch.float64, device=tgt.I.device)
    for box in boxes:
        y0, y1, x0, x1 = box
        sl = (slice(y0, y1), slice(x0, x1))
        w = weight[sl].double()
        a = tgt.pixel_maps(alt, box)
        d[0] += ((a["rgb_l1"] - base_maps["rgb_l1"][sl]) * w).sum()
        d[2] += ((a["mask"] - base_maps["mask"][sl]) * w).sum()
        d[3] += ((a["depth"] - base_maps["depth"][sl]) * w).sum()
        d[1] += ((tgt.ssim_map(alt, box) - base_maps["rgb_ssim"][sl]) * w).sum()
    return d
