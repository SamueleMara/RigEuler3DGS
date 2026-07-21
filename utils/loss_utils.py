#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


# -----------------------------------------------------------------------------
# Static semantic / DINO feature-field losses.
# -----------------------------------------------------------------------------
def _dynamic_root_for_train(dataset, dyn):
    explicit = getattr(dyn, "root_scene", "") or getattr(dyn, "dynamic_sequence_root", "")
    if explicit:
        return Path(explicit).expanduser().resolve()
    src = Path(dataset.source_path).expanduser().resolve()
    if src.name == "colmap":
        return src.parent
    if (src / "colmap").exists():
        return src
    return src


def _dino_root_for_train(dataset, dyn):
    root = _dynamic_root_for_train(dataset, dyn)
    return root / getattr(dyn, "dinov3_root", "dinov3")


def _read_dino_feature_tensor(path):
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        for key in ("features", "dino", "dino_feat", "feature", "feat"):
            if key in obj:
                obj = obj[key]
                break
    if not torch.is_tensor(obj):
        raise RuntimeError(f"DINO feature file has unsupported object type: {type(obj)!r}")
    feat = obj.float()
    if feat.ndim == 4:
        feat = feat[0]
    if feat.ndim != 3:
        raise RuntimeError(f"DINO feature map must be [C,H,W] or [1,C,H,W], got {tuple(feat.shape)}")
    # If stored as H,W,C, move channels first.
    if feat.shape[0] > feat.shape[-1] and feat.shape[-1] <= 4096:
        # Ambiguous; DINO maps are usually C,H,W with C large and H/W small.
        pass
    elif feat.shape[-1] > feat.shape[0] and feat.shape[0] <= 256 and feat.shape[1] <= 256:
        feat = feat.permute(2, 0, 1).contiguous()
    return feat


def _mask_root_for_train(dataset, dyn):
    root = _dynamic_root_for_train(dataset, dyn)
    return root / getattr(dyn, "semantic_mask_dir", "mask")


def _mask_path_for_camera(viewpoint_cam, dataset, dyn):
    mask_root = _mask_root_for_train(dataset, dyn)
    cam_id = getattr(viewpoint_cam, "dynamic_camera_id", None)
    frame_id = getattr(viewpoint_cam, "frame_id", None)
    if cam_id is not None and frame_id is not None:
        for ext in (".png", ".jpg", ".jpeg"):
            p = mask_root / str(cam_id) / f"frame_{int(frame_id):06d}{ext}"
            if p.exists():
                return p
    img = getattr(viewpoint_cam, "dynamic_image_path", "")
    if img:
        imgp = Path(img)
        for ext in (".png", ".jpg", ".jpeg"):
            p = mask_root / imgp.parent.name / (imgp.stem + ext)
            if p.exists():
                return p
    # Static COLMAP fallback: mask directory mirrors image name.
    image_name = getattr(viewpoint_cam, "image_name", "")
    if image_name:
        stem = Path(image_name).stem
        for pat in (f"**/{stem}.png", f"**/{stem}.jpg", f"**/{stem}.jpeg"):
            found = list(mask_root.glob(pat))
            if found:
                return found[0]
    return None


def _load_mask_for_view(viewpoint_cam, dataset, dyn, hw, device="cuda"):
    path = _mask_path_for_camera(viewpoint_cam, dataset, dyn)
    if path is None:
        # Use camera alpha mask if available; many datasets already load masks there.
        alpha = getattr(viewpoint_cam, "alpha_mask", None)
        if alpha is not None:
            mask = alpha.float()
            if mask.ndim == 3:
                mask = mask[0]
            return F.interpolate(mask[None, None].to(device), size=hw, mode="nearest")[0, 0].clamp(0, 1)
        return None
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    mask = torch.from_numpy(arr).to(device=device)
    mask = F.interpolate(mask[None, None], size=hw, mode="nearest")[0, 0]
    return mask.clamp(0, 1)


def dice_loss(pred, target, eps=1e-6):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    inter = (pred * target).sum()
    return 1.0 - (2.0 * inter + eps) / (pred.sum() + target.sum() + eps)


def _gaussian_rgb_for_affinity(gaussians):
    """Return a detached color descriptor [N,3] for color-aware local smoothing."""
    try:
        dc = gaussians.get_features_dc.detach()
    except Exception:
        return None
    if dc.ndim == 3:
        # GraphDeCo stores DC as [N,1,3]. Be robust to [N,3,1].
        if dc.shape[1] == 1 and dc.shape[2] == 3:
            dc = dc[:, 0, :]
        elif dc.shape[1] == 3 and dc.shape[2] == 1:
            dc = dc[:, :, 0]
        else:
            dc = dc.reshape(dc.shape[0], -1)[:, :3]
    elif dc.ndim == 2:
        dc = dc[:, :3]
    else:
        return None
    # Only an affinity cue is needed. Sigmoid bounds SH-DC values and keeps the
    # color distance numerically stable without backpropagating through color.
    return torch.sigmoid(dc.float()).detach()


def gaussian_aux_3d_smoothness(
    xyz,
    field,
    colors=None,
    mode="cosine",
    k=8,
    max_points=2048,
    sigma_x=-1.0,
    sigma_c=0.25,
):
    """Weak local 3D smoothness for auxiliary Gaussian fields.

    Affinity:
      a_ij = exp(-||mu_i-mu_j||^2/sigma_x^2) exp(-||c_i-c_j||^2/sigma_c^2)

    Geometry/color only define the detached neighbourhood weights. Gradients
    flow only into `field`, so semantic smoothness trains semantic logits only
    and DINO smoothness trains DINO features only.
    """
    if field is None or field.numel() == 0:
        return None
    if xyz is None or xyz.numel() == 0:
        return None
    n = int(field.shape[0])
    if n < 2:
        return None
    k = max(1, min(int(k), n - 1))
    m = min(int(max_points), n) if int(max_points) > 0 else n
    device = field.device
    if m < n:
        idx = torch.randperm(n, device=device)[:m]
    else:
        idx = torch.arange(n, device=device)

    xyz_s = xyz.detach()[idx].float()
    field_s = field[idx].float()
    with torch.no_grad():
        dist = torch.cdist(xyz_s, xyz_s, p=2)
        dist.fill_diagonal_(float("inf"))
        knn_dist, knn = torch.topk(dist, k=k, largest=False, dim=1)
        if float(sigma_x) > 0:
            sx = torch.tensor(float(sigma_x), device=device, dtype=knn_dist.dtype)
        else:
            finite = knn_dist[torch.isfinite(knn_dist)]
            sx = finite.median() if finite.numel() else torch.tensor(1.0, device=device, dtype=knn_dist.dtype)
            sx = sx.clamp_min(1e-6)
        weights = torch.exp(-(knn_dist ** 2) / (sx ** 2 + 1e-12))

        if colors is not None and float(sigma_c) > 0:
            col_s = colors.detach()[idx].float()
            col_j = col_s[knn]
            col_i = col_s[:, None, :]
            cd2 = ((col_i - col_j) ** 2).sum(dim=-1)
            weights = weights * torch.exp(-cd2 / (float(sigma_c) ** 2 + 1e-12))
        weights = weights.detach()

    if mode == "semantic":
        if field_s.ndim == 2 and field_s.shape[1] == 1:
            values = torch.sigmoid(field_s[:, 0])
        else:
            values = torch.sigmoid(field_s.reshape(field_s.shape[0], -1)[:, 0])
        vi = values[:, None]
        vj = values[knn]
        pair_loss = F.smooth_l1_loss(vi.expand_as(vj), vj, reduction="none")
    else:
        z = F.normalize(field_s, dim=-1)
        zi = z[:, None, :]
        zj = z[knn]
        pair_loss = 1.0 - (zi * zj).sum(dim=-1).clamp(-1.0, 1.0)

    denom = weights.sum().clamp_min(1e-6)
    return (weights * pair_loss).sum() / denom


def semantic_feature_field_loss(viewpoint_cam, gaussians, pipe, dataset, dyn):
    """Alpha-weighted binary semantic loss on the rendered Gaussian field.

    Implements the image-domain loss:

        L_sem = - sum_p A_k(p) * BCE_terms(M_k(p), M_hat_k(p)) / sum_p A_k(p)

    where A_k is the accumulated Gaussian alpha/visibility.  The semantic render
    uses detached geometry/opacities, so gradients flow only to
    gaussians._semantic_logits.
    """
    from gaussian_renderer import render_semantic_field, render_feature_field

    if not getattr(dyn, "use_semantic_feature_field", False):
        return None, {}
    if not hasattr(gaussians, "get_semantic_probability") or gaussians.get_semantic_probability is None:
        return None, {}

    H = int(viewpoint_cam.original_image.shape[-2])
    W = int(viewpoint_cam.original_image.shape[-1])
    target = _load_mask_for_view(viewpoint_cam, dataset, dyn, (H, W), device=gaussians.get_xyz.device)
    if target is None:
        return None, {"semantic_mask_missing": 1.0}

    # Detach RGB/depth geometry for this auxiliary render: the semantic loss
    # must train only the per-Gaussian semantic logits.
    pkg = render_semantic_field(viewpoint_cam, gaussians, pipe, detach_geometry=True)
    pred = pkg["render"][0].clamp(1e-5, 1.0 - 1e-5)

    # A_k(p): accumulated alpha/visibility.  The feature rasterizer alpha-
    # composites an all-ones scalar field, giving sum_i w_i(p).
    with torch.no_grad():
        ones_pkg = render_feature_field(
            viewpoint_camera=viewpoint_cam,
            pc=gaussians,
            pipe=pipe,
            features=torch.ones((gaussians.get_xyz.shape[0], 1), device=gaussians.get_xyz.device, dtype=pred.dtype),
            bg_feature=torch.zeros(1, device=gaussians.get_xyz.device, dtype=pred.dtype),
            detach_geometry=True,
        )
        alpha = ones_pkg["render"][0].detach().clamp(0.0, 1.0)

    valid = torch.isfinite(pred) & torch.isfinite(target) & torch.isfinite(alpha) & (alpha > 1e-8)
    if valid.sum() < 16:
        return None, {"semantic_valid_pixels": float(valid.sum().item())}

    loss_type = str(getattr(dyn, "semantic_loss_type", "bce"))
    bce_map = F.binary_cross_entropy(pred, target, reduction="none")
    weight = alpha * valid.float()
    denom = weight.sum().clamp_min(1e-6)
    bce = (weight * bce_map).sum() / denom

    # Optional dice branch is also alpha-weighted if selected.
    pred_w = pred * weight
    target_w = target * weight
    dice = 1.0 - (2.0 * (pred_w * target).sum() + 1e-6) / (pred_w.sum() + target_w.sum() + 1e-6)

    if loss_type == "dice":
        pixel_loss = dice
    elif loss_type == "bce_dice":
        pixel_loss = bce + dice
    else:
        pixel_loss = bce

    smooth_loss = None
    lambda_smooth = float(getattr(dyn, "lambda_semantic_3d_smooth", 0.0))
    if lambda_smooth > 0:
        smooth_loss = gaussian_aux_3d_smoothness(
            xyz=gaussians.get_canonical_xyz,
            field=gaussians.get_semantic_logits,
            colors=_gaussian_rgb_for_affinity(gaussians),
            mode="semantic",
            k=int(getattr(dyn, "smooth_3d_knn", 8)),
            max_points=int(getattr(dyn, "smooth_3d_max_points", 2048)),
            sigma_x=float(getattr(dyn, "smooth_3d_sigma_x", -1.0)),
            sigma_c=float(getattr(dyn, "smooth_3d_sigma_c", 0.25)),
        )

    loss = pixel_loss if smooth_loss is None else pixel_loss + lambda_smooth * smooth_loss
    return loss, {
        "L_semantic_feature_field": float(loss.detach().item()),
        "semantic_bce": float(bce.detach().item()),
        "semantic_dice": float(dice.detach().item()),
        "semantic_valid_pixels": float(valid.sum().item()),
        "semantic_mean_alpha": float(alpha[valid].mean().detach().item()) if valid.any() else 0.0,
        "L_semantic_3d_smooth": float(smooth_loss.detach().item()) if smooth_loss is not None else 0.0,
    }


def _dino_feature_path_for_camera(viewpoint_cam, dataset, dyn):
    dino_root = _dino_root_for_train(dataset, dyn)
    cam_id = getattr(viewpoint_cam, "dynamic_camera_id", None)
    frame_id = getattr(viewpoint_cam, "frame_id", None)
    if cam_id is not None and frame_id is not None:
        p = dino_root / str(cam_id) / f"frame_{int(frame_id):06d}.pt"
        if p.exists():
            return p
    # Fallback from dynamic_image_path: ROOT/rgb/cam_XXX/frame_XXXXXX.png -> ROOT/dinov3/cam_XXX/frame_XXXXXX.pt
    img = getattr(viewpoint_cam, "dynamic_image_path", "")
    if img:
        imgp = Path(img)
        p = dino_root / imgp.parent.name / (imgp.stem + ".pt")
        if p.exists():
            return p
    return None


def _load_dino_target_for_view(viewpoint_cam, dataset, dyn, hw, device="cuda"):
    path = _dino_feature_path_for_camera(viewpoint_cam, dataset, dyn)
    if path is None:
        return None
    cache = getattr(dyn, "_dino_feature_cache", None)
    key = str(path)
    if cache is not None and key in cache:
        feat = cache[key]
    else:
        try:
            feat = _read_dino_feature_tensor(path)
        except Exception as exc:
            print(f"[DINO feature field] Failed reading {path}: {exc}")
            return None
        max_cache = int(getattr(dyn, "dino_feature_cache_size", 16))
        if cache is not None and max_cache > 0:
            if len(cache) >= max_cache:
                # Simple FIFO eviction.  Good enough and avoids caching all 480 maps.
                old = next(iter(cache.keys()))
                cache.pop(old, None)
            cache[key] = feat
    dim = int(getattr(dyn, "dino_feature_field_dim", feat.shape[0]))
    feat = feat[:dim].unsqueeze(0).to(device=device, dtype=torch.float32)
    feat = F.interpolate(feat, size=hw, mode="bilinear", align_corners=False)[0]
    feat = F.normalize(feat, dim=0)
    return feat


def dino_feature_field_loss(viewpoint_cam, gaussians, pipe, background, dataset, dyn, separate_sh=False, stage="static"):
    """DINO feature-field supervision.

    Static stage:
        train per-Gaussian DINO descriptors z_i while geometry/opacities are
        detached.

    Dynamic stage:
        freeze z_i and use it as a canonical descriptor attached to each
        Gaussian. Render the transported Gaussians' frozen descriptors and
        compare them with the target DINO map. Gradients flow through the
        transported render to the Eulerian motion field/grid, not to z_i.
    """
    from gaussian_renderer import render_feature_field
    if not getattr(dyn, "use_dino_feature_field", False):
        return None, {}
    feat_param = getattr(gaussians, "get_dino_feature_field", None)
    if feat_param is None:
        feat_param = getattr(gaussians, "_dino_feature_field", None)
    if feat_param is None or feat_param.numel() == 0:
        return None, {}

    dynamic_stage = bool(getattr(dyn, "dynamic_motion", False) and str(stage) != "static")
    freeze_dynamic = bool(getattr(dyn, "freeze_dino_feature_dynamic", True))
    train_features = not (dynamic_stage and freeze_dynamic)

    H = int(viewpoint_cam.original_image.shape[-2])
    W = int(viewpoint_cam.original_image.shape[-1])
    target = _load_dino_target_for_view(viewpoint_cam, dataset, dyn, (H, W), device=feat_param.device)
    if target is None:
        return None, {"dino_feature_missing": 1.0}
    K = int(feat_param.shape[1])

    feat_for_render = feat_param if train_features else feat_param.detach()
    feat_for_render = F.normalize(feat_for_render, dim=-1)

    # Static: detach geometry so DINO trains only z_i.
    # Dynamic: keep transported geometry live so DINO trains the transport.
    detach_geometry = not dynamic_stage

    pred_pkg = render_feature_field(
        viewpoint_camera=viewpoint_cam,
        pc=gaussians,
        pipe=pipe,
        features=feat_for_render,
        bg_feature=torch.zeros(K, device=feat_param.device, dtype=feat_param.dtype),
        detach_geometry=detach_geometry,
    )
    pred_raw = pred_pkg["render"]

    # Match the normalized alpha-composited expression:
    #   z_tilde(p) = sum_i w_i(p) z_i / (sum_i w_i(p) + eps)
    # In dynamic stage the denominator remains differentiable w.r.t.
    # transported geometry, so feature consistency can move Gaussians.
    weight_pkg = render_feature_field(
        viewpoint_camera=viewpoint_cam,
        pc=gaussians,
        pipe=pipe,
        features=torch.ones((feat_param.shape[0], 1), device=feat_param.device, dtype=feat_param.dtype),
        bg_feature=torch.zeros(1, device=feat_param.device, dtype=feat_param.dtype),
        detach_geometry=detach_geometry,
    )
    opacity = weight_pkg["render"][0]
    pred = pred_raw / opacity.clamp_min(1e-4)[None]

    pred_n = F.normalize(pred, dim=0)
    target_n = F.normalize(target, dim=0)
    valid = (
        (opacity.detach() > float(getattr(dyn, "dino_feature_opacity_threshold", 0.05)))
        & torch.isfinite(pred_n).all(dim=0)
        & torch.isfinite(target_n).all(dim=0)
    )
    if valid.sum() < 16:
        return None, {"dino_feature_valid_pixels": float(valid.sum().item())}

    cos = 1.0 - (pred_n[:, valid] * target_n[:, valid]).sum(dim=0).clamp(-1.0, 1.0)
    if str(getattr(dyn, "dino_feature_loss_type", "cosine")) == "cosine_l1":
        dino_pixel_loss = cos.mean() + 0.1 * torch.abs(pred[:, valid] - target[:, valid]).mean()
    else:
        dino_pixel_loss = cos.mean()

    smooth_loss = None
    # DINO 3D smoothness is only a static descriptor-field regularizer.
    lambda_smooth = float(getattr(dyn, "lambda_dino_3d_smooth", 0.0)) if not dynamic_stage else 0.0
    if lambda_smooth > 0 and train_features:
        smooth_loss = gaussian_aux_3d_smoothness(
            xyz=gaussians.get_canonical_xyz,
            field=feat_param,
            colors=_gaussian_rgb_for_affinity(gaussians),
            mode="cosine",
            k=int(getattr(dyn, "smooth_3d_knn", 8)),
            max_points=int(getattr(dyn, "smooth_3d_max_points", 2048)),
            sigma_x=float(getattr(dyn, "smooth_3d_sigma_x", -1.0)),
            sigma_c=float(getattr(dyn, "smooth_3d_sigma_c", 0.25)),
        )

    loss = dino_pixel_loss if smooth_loss is None else dino_pixel_loss + lambda_smooth * smooth_loss
    return loss, {
        "L_dino_feature_field": float(loss.detach().item()),
        "dino_cosine": float(cos.mean().detach().item()),
        "L_dino_3d_smooth": float(smooth_loss.detach().item()) if smooth_loss is not None else 0.0,
        "dino_feature_valid_pixels": float(valid.sum().item()),
        "dino_feature_mean_opacity": float(opacity.detach()[valid].mean().item()) if valid.any() else 0.0,
        "dino_dynamic_transport_supervision": float(dynamic_stage),
        "dino_feature_frozen": float(dynamic_stage and freeze_dynamic),
    }

# -----------------------------------------------------------------------------
# Dynamic Eulerian-motion extension helpers.
# Kept here as re-exports so existing code can continue importing losses from
# utils.loss_utils, while the implementation lives in utils.dynamic_loss_utils.
# -----------------------------------------------------------------------------
try:
    from utils.dynamic_loss_utils import (
        speed_loss,
        static_velocity_loss,
        dynamic_sparsity_loss,
        dynamic_binarization_loss,
        cycle_consistency_loss,
        sample_pairs_by_cell,
        dino_weighted_rigidity_loss,
        se3_local_coherence_loss,
        compute_cell_incoherence_scores,
    )
except Exception:
    # The static 3DGS path must remain importable even if dynamic utilities are
    # not available in a partial checkout.
    pass
