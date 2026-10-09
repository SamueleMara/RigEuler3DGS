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
import re
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


def _read_feature_tensor(path, *, family="feature"):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(str(path))
        for key in ("features", family, "feature", "feat", "embedding"):
            if key in data:
                obj = torch.from_numpy(data[key])
                break
        else:
            first = data.files[0] if data.files else None
            if first is None:
                raise RuntimeError(f"{family} npz file is empty: {path}")
            obj = torch.from_numpy(data[first])
    elif path.suffix.lower() == ".npy":
        obj = torch.from_numpy(np.load(str(path)))
    else:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("features", family, f"{family}_feat", "feature", "feat", "embedding"):
            if key in obj:
                obj = obj[key]
                break
    if not torch.is_tensor(obj):
        raise RuntimeError(f"{family} feature file has unsupported object type: {type(obj)!r}")
    feat = obj.float()
    if feat.ndim == 4:
        feat = feat[0]
    if feat.ndim != 3:
        raise RuntimeError(f"{family} feature map must be [C,H,W] or [1,C,H,W], got {tuple(feat.shape)}")
    # If stored as H,W,C, move channels first.
    if feat.shape[0] > feat.shape[-1] and feat.shape[-1] <= 4096:
        # Ambiguous; DINO maps are usually C,H,W with C large and H/W small.
        pass
    elif feat.shape[-1] > feat.shape[0] and feat.shape[0] <= 256 and feat.shape[1] <= 256:
        feat = feat.permute(2, 0, 1).contiguous()
    return feat


def _read_dino_feature_tensor(path):
    return _read_feature_tensor(path, family="dino")


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
    # Canonical COLMAP images normally represent frame zero.  Accept both
    # nested names (cam_000/frame_000000) and flattened names
    # (cam_000_frame_000000) so Phase 0 supervises the same semantic foreground
    # used by soft discovery and later hard joint fitting.
    name = str(getattr(viewpoint_cam, "image_name", ""))
    cam_match = re.search(r"cam_\d+", name)
    frame_match = re.search(r"frame[_-]?(\d+)", name)
    if cam_match is not None:
        cam_id = cam_match.group(0)
        frame_id = int(frame_match.group(1)) if frame_match is not None else 0
        stem = Path(name).stem
        for ext in (".png", ".jpg", ".jpeg"):
            for p in (
                mask_root / cam_id / f"frame_{frame_id:06d}{ext}",
                mask_root / cam_id / f"{stem}{ext}",
                mask_root / f"{stem}{ext}",
            ):
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
    composited = pkg["render"][0]

    # A_k(p): accumulated alpha/visibility.  The feature rasterizer alpha-
    # composites an all-ones scalar field, giving sum_i w_i(p).
    with torch.no_grad():
        ones_pkg = render_feature_field(
            viewpoint_camera=viewpoint_cam,
            pc=gaussians,
            pipe=pipe,
            features=torch.ones((gaussians.get_xyz.shape[0], 1), device=gaussians.get_xyz.device, dtype=composited.dtype),
            bg_feature=torch.zeros(1, device=gaussians.get_xyz.device, dtype=composited.dtype),
            detach_geometry=True,
        )
        alpha = ones_pkg["render"][0].detach().clamp(0.0, 1.0)

    # NORMALISE BY COVERAGE BEFORE COMPARING WITH THE MASK.
    #
    #     S(p) = sum_i w_i(p) s_i        is a weighted SUM, not a mean
    #     A(p) = sum_i w_i(p)            the same weights, on a field of ones
    #     s_hat(p) = S(p) / (A(p) + eps) the coverage-normalised mean
    #
    # The mask is a per-pixel statement about WHAT is there, not about how
    # much of the pixel is covered. Comparing the raw sum against it conflates
    # the two: wherever A < 1 -- a silhouette edge, a thin part, anything
    # partially occluded -- S is attenuated even when every contributing
    # Gaussian already has s_i = 1, so the target of 1 is unreachable and the
    # only way to close the gap is to push the logits up. Measured on the
    # trained models, that is exactly what happened: the field saturated at
    # p = 1.0 on 9 of 13 scenes and separated the two classes by less than
    # 0.005 on most of them.
    #
    # s_hat is the expected semantic value of whatever is at that pixel,
    # independent of coverage, which is the quantity the mask is a label for.
    # `alpha` is detached, so the division changes the gradient the logits
    # receive but does not open a second route into the geometry -- this loss
    # trains `_semantic_logits` and nothing else.
    eps = 1e-6
    pred = (composited / (alpha + eps)).clamp(1e-5, 1.0 - 1e-5)

    valid = torch.isfinite(pred) & torch.isfinite(target) & torch.isfinite(alpha) & (alpha > 1e-8)
    if valid.sum() < 16:
        return None, {"semantic_valid_pixels": float(valid.sum().item())}

    bce_map = F.binary_cross_entropy(pred, target, reduction="none")
    weight = alpha * valid.float()
    denom = weight.sum() + eps
    bce = (weight * bce_map).sum() / denom

    return bce, {
        "L_semantic_feature_field": float(bce.detach().item()),
        "semantic_bce": float(bce.detach().item()),
        "semantic_valid_pixels": float(valid.sum().item()),
        "semantic_mean_alpha": float(alpha[valid].mean().detach().item()) if valid.any() else 0.0,
        "semantic_mean_pred": float(pred[valid].mean().detach().item()) if valid.any() else 0.0,
        "semantic_mean_target": float(target[valid].mean().detach().item()) if valid.any() else 0.0,
    }


def _dino_feature_path_for_camera(viewpoint_cam, dataset, dyn):
    dino_root = _dino_root_for_train(dataset, dyn)
    extension = str(getattr(dyn, "dinov3_feature_ext", ".pt"))
    if not extension.startswith("."):
        extension = "." + extension
    cam_id = getattr(viewpoint_cam, "dynamic_camera_id", None)
    frame_id = getattr(viewpoint_cam, "frame_id", None)
    if cam_id is not None and frame_id is not None:
        p = dino_root / str(cam_id) / f"frame_{int(frame_id):06d}{extension}"
        if p.exists():
            return p
    # Fallback from dynamic_image_path: ROOT/rgb/cam_XXX/frame_XXXXXX.png -> ROOT/dinov3/cam_XXX/frame_XXXXXX.pt
    img = getattr(viewpoint_cam, "dynamic_image_path", "")
    if img:
        imgp = Path(img)
        p = dino_root / imgp.parent.name / (imgp.stem + extension)
        if p.exists():
            return p
    name = str(getattr(viewpoint_cam, "image_name", ""))
    cam_match = re.search(r"cam_\d+", name)
    frame_match = re.search(r"frame[_-]?(\d+)", name)
    if cam_match is not None:
        cam_id = cam_match.group(0)
        frame_id = int(frame_match.group(1)) if frame_match is not None else 0
        stem = Path(name).stem
        for p in (
            dino_root / cam_id / f"frame_{frame_id:06d}{extension}",
            dino_root / cam_id / f"{stem}{extension}",
            dino_root / f"{stem}{extension}",
        ):
            if p.exists():
                return p
    return None


def _sam3_root_for_train(dataset, dyn):
    requested = str(getattr(dyn, "sam3_feature_root", "") or "").strip()
    if requested:
        path = Path(requested).expanduser()
        return path if path.is_absolute() else (_dynamic_root_for_train(dataset, dyn) / path)
    return _dynamic_root_for_train(dataset, dyn) / "sam3"


def _sam3_feature_path_for_camera(viewpoint_cam, dataset, dyn):
    sam3_root = _sam3_root_for_train(dataset, dyn)
    extensions = (".pt", ".pth", ".npz", ".npy")
    cam_id = getattr(viewpoint_cam, "dynamic_camera_id", None)
    frame_id = getattr(viewpoint_cam, "frame_id", None)
    if cam_id is not None and frame_id is not None:
        for ext in extensions:
            p = sam3_root / str(cam_id) / f"frame_{int(frame_id):06d}{ext}"
            if p.exists():
                return p
    img = getattr(viewpoint_cam, "dynamic_image_path", "")
    if img:
        imgp = Path(img)
        for ext in extensions:
            p = sam3_root / imgp.parent.name / (imgp.stem + ext)
            if p.exists():
                return p
    name = str(getattr(viewpoint_cam, "image_name", ""))
    cam_match = re.search(r"cam_\d+", name)
    frame_match = re.search(r"frame[_-]?(\d+)", name)
    if cam_match is not None:
        cam_id = cam_match.group(0)
        frame_id = int(frame_match.group(1)) if frame_match is not None else 0
        stem = Path(name).stem
        for ext in extensions:
            for p in (
                sam3_root / cam_id / f"frame_{frame_id:06d}{ext}",
                sam3_root / cam_id / f"{stem}{ext}",
                sam3_root / f"{stem}{ext}",
            ):
                if p.exists():
                    return p
    return None


_DINO_TARGET_CACHE = {}
_SAM3_TARGET_CACHE = {}


def _load_dino_target_for_view(viewpoint_cam, dataset, dyn, hw, device="cuda"):
    path = _dino_feature_path_for_camera(viewpoint_cam, dataset, dyn)
    if path is None:
        return None
    key = str(path)
    if key in _DINO_TARGET_CACHE:
        feat = _DINO_TARGET_CACHE[key]
    else:
        try:
            feat = _read_dino_feature_tensor(path)
        except Exception as exc:
            print(f"[DINO feature field] Failed reading {path}: {exc}")
            return None
        if len(_DINO_TARGET_CACHE) >= 16:
            _DINO_TARGET_CACHE.pop(next(iter(_DINO_TARGET_CACHE)))
        _DINO_TARGET_CACHE[key] = feat
    dim = int(getattr(dyn, "dino_feature_field_dim", feat.shape[0]))
    feat = feat[:dim].unsqueeze(0).to(device=device, dtype=torch.float32)
    feat = F.interpolate(feat, size=hw, mode="bilinear", align_corners=False)[0]
    feat = F.normalize(feat, dim=0)
    return feat


def _load_sam3_target_for_view(viewpoint_cam, dataset, dyn, hw, device="cuda"):
    path = _sam3_feature_path_for_camera(viewpoint_cam, dataset, dyn)
    if path is None:
        return None
    key = str(path)
    if key in _SAM3_TARGET_CACHE:
        feat = _SAM3_TARGET_CACHE[key]
    else:
        try:
            feat = _read_feature_tensor(path, family="sam3")
        except Exception as exc:
            print(f"[SAM3 feature field] Failed reading {path}: {exc}")
            return None
        if len(_SAM3_TARGET_CACHE) >= 16:
            _SAM3_TARGET_CACHE.pop(next(iter(_SAM3_TARGET_CACHE)))
        _SAM3_TARGET_CACHE[key] = feat
    dim = int(getattr(dyn, "sam3_feature_dim", feat.shape[0]))
    feat = feat[:dim].unsqueeze(0).to(device=device, dtype=torch.float32)
    feat = F.interpolate(feat, size=hw, mode="bilinear", align_corners=False)[0]
    feat = F.normalize(feat, dim=0)
    return feat


def _local_feature_cluster_loss(features, points, *, max_points=2048, margin=0.75):
    """Cheap optional SAM3 field sharpening. Disabled by default."""
    feature = F.normalize(torch.as_tensor(features).float(), dim=-1)
    xyz = torch.as_tensor(points, device=feature.device, dtype=feature.dtype)
    count = int(feature.shape[0])
    if count < 3:
        return feature.sum() * 0.0
    if count > int(max_points):
        ids = torch.linspace(0, count - 1, int(max_points), device=feature.device).round().long().unique()
        feature = feature.index_select(0, ids)
        xyz = xyz.index_select(0, ids)
    distance = torch.cdist(xyz, xyz)
    distance.fill_diagonal_(float("inf"))
    nn = distance.argmin(dim=1)
    cosine = (feature * feature[nn]).sum(-1).clamp(-1.0, 1.0)
    positive = (1.0 - cosine).square()
    negative = F.relu(float(margin) - torch.linalg.norm(feature - feature[nn], dim=-1)).square()
    weight_pos = (cosine.detach() > 0.5).float()
    weight_neg = 1.0 - weight_pos
    return (weight_pos * positive + weight_neg * negative).mean()


def sam3_feature_field_loss(viewpoint_cam, gaussians, pipe, background, dataset, dyn, separate_sh=False, stage="static"):
    """SAM3 feature-field supervision from cached dense tensors.

    The SAM3 extractor runs outside the RigEuler env. This loss only renders
    cached descriptors and optimizes the per-Gaussian SAM3 field unless called
    in a dynamic stage, where feature rows are detached and geometry receives
    the transport gradient.
    """
    from gaussian_renderer import render_feature_field, render_sam3_feature_field

    del background, separate_sh
    if not getattr(dyn, "use_sam3_feature_field", False):
        return None, {}
    feat_param = getattr(gaussians, "get_sam3_feature_field", None)
    if feat_param is None or not torch.is_tensor(feat_param) or feat_param.numel() == 0:
        return None, {}

    dynamic_stage = bool(getattr(dyn, "dynamic_motion", False) and str(stage) != "static")
    freeze_dynamic = bool(getattr(dyn, "freeze_sam3_feature_dynamic", True))
    train_features = not (dynamic_stage and freeze_dynamic)

    H = int(viewpoint_cam.original_image.shape[-2])
    W = int(viewpoint_cam.original_image.shape[-1])
    target = _load_sam3_target_for_view(viewpoint_cam, dataset, dyn, (H, W), device=feat_param.device)
    if target is None:
        return None, {"sam3_feature_missing": 1.0}
    K = min(int(feat_param.shape[1]), int(target.shape[0]))
    if K <= 0:
        return None, {"sam3_feature_missing": 1.0}

    feat_for_render = F.normalize(feat_param[:, :K] if train_features else feat_param[:, :K].detach(), dim=-1)
    detach_geometry = not dynamic_stage

    pred_pkg = render_sam3_feature_field(
        viewpoint_camera=viewpoint_cam,
        pc=gaussians,
        pipe=pipe,
        features=feat_for_render,
        detach_geometry=detach_geometry,
    )
    weight_pkg = render_feature_field(
        viewpoint_camera=viewpoint_cam,
        pc=gaussians,
        pipe=pipe,
        features=torch.ones((feat_param.shape[0], 1), device=feat_param.device, dtype=feat_param.dtype),
        bg_feature=torch.zeros(1, device=feat_param.device, dtype=feat_param.dtype),
        detach_geometry=detach_geometry,
    )
    opacity = weight_pkg["render"][0]
    pred = pred_pkg["render"] / opacity.clamp_min(1.0e-4)[None]
    pred_n = F.normalize(pred, dim=0)
    target_n = F.normalize(target[:K], dim=0)
    valid = (
        (opacity.detach() > float(getattr(dyn, "sam3_feature_opacity_threshold", 0.05)))
        & torch.isfinite(pred_n).all(dim=0)
        & torch.isfinite(target_n).all(dim=0)
    )
    if dynamic_stage:
        foreground_mask = _load_mask_for_view(
            viewpoint_cam, dataset, dyn, (H, W), device=feat_param.device
        )
        if foreground_mask is None:
            raise RuntimeError("Foreground-only Phase 1 requires a mask for every dynamic view")
        valid = valid & (foreground_mask > 0.5)
    if valid.sum() < 16:
        return None, {"sam3_feature_valid_pixels": float(valid.sum().item())}

    cosine = 1.0 - (pred_n[:, valid] * target_n[:, valid]).sum(dim=0).clamp(-1.0, 1.0)
    alpha_weight = opacity.detach()[valid].clamp_min(0.0)
    loss = (alpha_weight * cosine).sum() / alpha_weight.sum().clamp_min(1.0e-6)
    if str(stage) == "static" and float(getattr(dyn, "sam3_cluster_weight", 0.0)) > 0.0:
        loss = loss + float(getattr(dyn, "sam3_cluster_weight", 0.0)) * _local_feature_cluster_loss(
            feat_for_render,
            gaussians.get_xyz.detach(),
        )
    return loss, {
        "L_sam3_feature_field": float(loss.detach().item()),
        "sam3_cosine": float(cosine.mean().detach().item()),
        "sam3_feature_valid_pixels": float(valid.sum().item()),
        "sam3_feature_mean_opacity": float(opacity.detach()[valid].mean().item()) if valid.any() else 0.0,
        "sam3_dynamic_transport_supervision": float(dynamic_stage),
        "sam3_feature_frozen": float(dynamic_stage and freeze_dynamic),
    }


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
    if dynamic_stage:
        foreground_mask = _load_mask_for_view(
            viewpoint_cam, dataset, dyn, (H, W), device=feat_param.device
        )
        if foreground_mask is None:
            raise RuntimeError(
                "Foreground-only Phase 1 requires a mask for every dynamic view"
            )
        valid = valid & (foreground_mask > 0.5)
    if valid.sum() < 16:
        return None, {"dino_feature_valid_pixels": float(valid.sum().item())}

    cos = 1.0 - (pred_n[:, valid] * target_n[:, valid]).sum(dim=0).clamp(-1.0, 1.0)
    loss = cos.mean()
    return loss, {
        "L_dino_feature_field": float(loss.detach().item()),
        "dino_cosine": float(cos.mean().detach().item()),
        "dino_feature_valid_pixels": float(valid.sum().item()),
        "dino_feature_mean_opacity": float(opacity.detach()[valid].mean().item()) if valid.any() else 0.0,
        "dino_dynamic_transport_supervision": float(dynamic_stage),
        "dino_feature_frozen": float(dynamic_stage and freeze_dynamic),
    }
