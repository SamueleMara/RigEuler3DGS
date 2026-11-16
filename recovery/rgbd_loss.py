"""FreeArtGS-style RGB + metric-depth + alpha objective, per view (spec 2.4, 7.3).

    L = lam_rgb [(1 - lam_s) L1(M*I_hat, M*I) + lam_s (1 - SSIM(M*I_hat, M*I))] + lam_D L_D + lam_M L_M
    L_D = sum_{p in V_depth} rho(D_hat_p - D_p) / max(1, |V_depth|)       D_hat = Z / (A + eps)
    L_M = sum_{p in V_mask} |A_p - M_p| / max(1, |V_mask|)

Parity with the FreeArtGS run the user compared against (results_09_11_26/freeart_real/.../config.yml and
nerfstudio/models/splatfacto.py get_loss_dict): ssim_lambda 0.2, fg_mask_lambda 1.0, sensor_depth_lambda 1.0,
white background, masked-image global-mean L1 + SSIM, alpha L1 over the whole image.

Declared differences from that code (the specification's rules, not tuning):
  * depth: FreeArt takes the L1 of (pred*mask - depth*mask) over ALL pixels, so a masked pixel whose sensor
    depth is missing (0) pulls the predicted depth to 0. Here only V_depth = mask & finite & depth > 0 counts,
    normalised by |V_depth|; with no valid depth L_D = 0 and its support is reported as 0.
  * depth where the render has (almost) no coverage: D_hat = Z/A is ill-conditioned as A -> 0 (dD/dw ~ 1/A).
    Pixels with A < depth_alpha_min are left out of L_D and counted as `depth_uncovered`; the mask term
    still pulls their coverage up.
  * optional depth_edge_px > 0 erodes V_depth only (mixed depth at silhouettes); the silhouette used by the
    mask and RGB terms is never eroded.
RGB terms: L1 = utils/loss_utils.l1_loss and SSIM = fused_ssim (CUDA) or utils/loss_utils.ssim, the functions of the
original 3DGS code, combined as its train.py does: (1 - lambda_dssim) L1 + lambda_dssim (1 - SSIM), lambda_dssim 0.2.
This objective is an explicitly weighted engineering loss, not the IEKF likelihood.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from utils.loss_utils import l1_loss


@dataclass
class LossConfig:
    lambda_rgb: float = 1.0
    lambda_ssim: float = 0.2
    lambda_depth: float = 1.0
    lambda_mask: float = 1.0
    depth_loss: str = "l1"            # l1 (FreeArt parity) | student_t (separate option, not the baseline)
    student_t_nu: float = 4.0
    student_t_scale: float = 0.01     # metres
    depth_alpha_min: float = 0.01
    depth_edge_px: int = 0
    use_rgb: bool = True
    use_depth: bool = True
    use_mask: bool = True
    ssim_impl: str = "fused_ssim"     # fused_ssim: the CUDA SSIM that 3DGS train.py uses | 3dgs_python: utils/loss_utils.ssim

    def to_dict(self):
        return asdict(self)


def _ssim(a, b, impl: str = "fused_ssim"):
    """The 3DGS (graphdeco-inria/gaussian-splatting) SSIM, exactly as its train.py calls it: the CUDA fused_ssim
    package, or the reference python utils/loss_utils.ssim (11x11 Gaussian window, sigma 1.5, C1 0.01^2, C2 0.03^2).
    No silent fallback: the implementation is a declared, recorded setting."""
    if impl == "fused_ssim":
        from fused_ssim import fused_ssim
        return fused_ssim(a[None], b[None])
    if impl == "3dgs_python":
        from utils.loss_utils import ssim
        return ssim(a[None], b[None])
    raise ValueError(f"ssim_impl {impl!r}")


def erode(mask: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return mask
    m = mask.float()[None, None]
    k = 2 * px + 1
    return (-F.max_pool2d(-m, k, stride=1, padding=px))[0, 0] > 0.5


def pixel_loss_map(out: dict, view, cfg: LossConfig) -> torch.Tensor:
    """Per-pixel [H,W] version of the same objective without the (non-local) SSIM term: the affected-pixel
    statistic of Phase 4 candidate tests. Weights match view_loss; the SSIM share of the RGB weight is
    dropped, not redistributed."""
    A, D_hat = out["alpha"][0], out["depth"][0]
    M = view.mask
    Mf = M.float()
    m = torch.zeros_like(A)
    if cfg.use_rgb:
        m = m + cfg.lambda_rgb * (1 - cfg.lambda_ssim) * ((out["color"] - view.rgb) * Mf).abs().mean(0)
    if cfg.use_depth:
        sup = erode(view.depth_valid, cfg.depth_edge_px) & (A >= cfg.depth_alpha_min)
        r = torch.where(sup, D_hat - view.depth, torch.zeros_like(D_hat))
        if cfg.depth_loss == "l1":
            rho = r.abs()
        else:
            rho = 0.5 * (cfg.student_t_nu + 1) * torch.log1p((r / cfg.student_t_scale) ** 2 / cfg.student_t_nu)
        m = m + cfg.lambda_depth * rho * sup.float()
    if cfg.use_mask:
        m = m + cfg.lambda_mask * (A - Mf).abs()
    return m


def view_loss(out: dict, view, cfg: LossConfig) -> dict:
    """out: render_articulated output; view: views.View on the same device. Returns the total and every term
    (detached floats for logging, `total` with grad)."""
    I_hat, A, D_hat = out["color"], out["alpha"][0], out["depth"][0]
    I, M = view.rgb, view.mask
    Mf = M.float()
    zero = I_hat.sum() * 0.0
    terms = {}
    # RGB on the foreground, FreeArt-style global mean over all pixels
    pm, gm = I_hat * Mf, I * Mf
    l1 = l1_loss(pm, gm)                                       # 3DGS utils/loss_utils.l1_loss
    ssim_loss = 1.0 - _ssim(pm, gm, cfg.ssim_impl)
    rgb = (1 - cfg.lambda_ssim) * l1 + cfg.lambda_ssim * ssim_loss
    n_fg = int(M.sum())
    terms["rgb_l1"] = float(l1)
    terms["rgb_ssim_loss"] = float(ssim_loss)
    terms["rgb_l1_foreground"] = float((pm - gm).abs().sum() / max(1, 3 * n_fg))     # coverage-normalised
    # depth on valid observed object depth, where the render covers the pixel
    vd = erode(view.depth_valid, cfg.depth_edge_px)
    covered = A.detach() >= cfg.depth_alpha_min
    sup = vd & covered
    n_d = int(sup.sum())
    if n_d > 0:
        r = D_hat[sup] - view.depth[sup]
        if cfg.depth_loss == "l1":
            rho = r.abs()
        elif cfg.depth_loss == "student_t":
            rho = 0.5 * (cfg.student_t_nu + 1) * torch.log1p((r / cfg.student_t_scale) ** 2 / cfg.student_t_nu)
        else:
            raise ValueError(cfg.depth_loss)
        dloss = rho.sum() / n_d
        terms["depth_l1"] = float(r.abs().mean())
    else:
        dloss = zero
        terms["depth_l1"] = 0.0
    terms["depth_support"] = n_d
    terms["depth_uncovered"] = int((vd & ~covered).sum())
    # alpha vs the whole binary mask (negatives included)
    mloss = (A - Mf).abs().mean()
    terms["mask_l1"] = float(mloss)
    terms["alpha_gap_fg"] = float((1 - A[M]).mean()) if n_fg else 0.0
    terms["false_alpha_bg"] = float(A[~M].mean()) if int((~M).sum()) else 0.0
    total = zero
    if cfg.use_rgb:
        total = total + cfg.lambda_rgb * rgb
    if cfg.use_depth:
        total = total + cfg.lambda_depth * dloss
    if cfg.use_mask:
        total = total + cfg.lambda_mask * mloss
    terms["total"] = total
    with torch.no_grad():
        mse = ((I_hat - I) ** 2 * Mf).sum() / max(1, 3 * n_fg)
        terms["psnr_fg"] = float(-10 * torch.log10(mse.clamp_min(1e-12)))
    return terms
