#!/usr/bin/env python3
"""Train the original successful v1 Gaussian denoiser from canonical frame 0 to an arbitrary target frame.

This is the restored minimal trainer for the checkpoint consumed by
`tools/frame01_motion_classifier.py`.

Important:
- canonical 3DGS geometry stays frozen;
- fresh positional noise is sampled around the canonical frame-0 Gaussians;
- target-frame RGB/depth render evidence supervises the recurrent corrections;
- no probability head;
- no GT in training;
- no classifier training;
- no temporal aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "scene").is_dir():
        raise RuntimeError(
            "Keep this script in RigEuler3DGS/tools and run it from the RigEuler3DGS repository."
        )
    return root


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import accumulate_pixel_features_to_gaussians, render  # noqa: E402
from scene.gaussian_motion_denoiser import GaussianMotionDenoiser, build_canonical_knn  # noqa: E402
from tools.test_render_denoise_discovery import build_scene, _ordered_complete_frame_groups  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


def _camera_id(camera) -> str:
    return str(getattr(camera, "dynamic_camera_id", getattr(camera, "image_name", "camera")))


def _select_random(cameras: list, count: int, generator: torch.Generator):
    if count <= 0 or count >= len(cameras):
        return list(cameras)
    order = torch.randperm(len(cameras), generator=generator).tolist()
    return [cameras[i] for i in order[: int(count)]]


def _image_mask(camera, device, dtype):
    mask = torch.ones((1, int(camera.image_height), int(camera.image_width)), device=device, dtype=dtype)
    if getattr(camera, "alpha_mask", None) is not None:
        mask = camera.alpha_mask.to(device=device, dtype=dtype)
        if mask.ndim == 2:
            mask = mask[None]
        mask = mask[:1].clamp(0.0, 1.0)
    return mask


def _robust_unit(value: torch.Tensor, valid: torch.Tensor | None = None):
    x = torch.as_tensor(value).detach().float()
    mask = torch.isfinite(x)
    if valid is not None:
        mask &= torch.as_tensor(valid, device=x.device).bool()
    if not bool(mask.any()):
        return torch.zeros_like(x)
    vals = x[mask]
    lo, hi = torch.quantile(vals, torch.tensor([0.10, 0.90], device=x.device))
    return torch.where(
        mask,
        ((x - lo) / (hi - lo).clamp_min(1.0e-8)).clamp(0.0, 1.0),
        torch.zeros_like(x),
    )


def _pseudo_huber(value: torch.Tensor, delta: float):
    d = max(float(delta), 1.0e-8)
    return d * d * (torch.sqrt(1.0 + (value / d).square()) - 1.0)


def _frame_maps(scene):
    groups = _ordered_complete_frame_groups(scene.getTrainCameras())
    if len(groups) < 1:
        raise RuntimeError("No complete dynamic multiview frame groups were built")
    return {int(frame): list(cameras) for frame, cameras in groups}


def _frame_time(cameras, frame_id: int, fps: float):
    if cameras:
        return float(getattr(cameras[0], "timestamp", float(frame_id) / float(fps)))
    return float(frame_id) / float(fps)


def _render_loss_and_evidence(
    cameras,
    gaussians,
    pipe,
    background,
    xyz,
    rotations,
    args,
    *,
    collect_evidence: bool,
):
    n = int(xyz.shape[0])
    device, dtype = xyz.device, xyz.dtype

    loss = xyz.sum() * 0.0
    rgb_loss = xyz.new_zeros(())
    ssim_loss = xyz.new_zeros(())
    depth_loss = xyz.new_zeros(())

    rgb_sum = torch.zeros(n, device=device)
    rgb_weight = torch.zeros(n, device=device)
    depth_sum = torch.zeros(n, device=device)
    depth_weight = torch.zeros(n, device=device)
    raw_support = torch.zeros(n, device=device)
    visible_mass = torch.zeros(n, device=device)
    valid_visible_mass = torch.zeros(n, device=device)

    for camera in cameras:
        package = render(
            camera,
            gaussians,
            pipe,
            background,
            use_trained_exp=False,
            means3D_override=xyz,
            rotations_override=rotations,
        )
        image = package["render"]
        target = camera.original_image.to(device=device, dtype=dtype)
        mask = _image_mask(camera, device, dtype)

        abs_rgb = torch.abs(image - target)
        rgb = (abs_rgb * mask).sum() / mask.expand_as(abs_rgb).sum().clamp_min(1.0)
        dssim = 1.0 - ssim(image * mask, target * mask)

        depth = image.new_zeros(())
        depth_residual = None
        depth_valid = None
        if bool(args.use_depth) and bool(getattr(camera, "depth_reliable", False)):
            target_depth = camera.invdepthmap.to(device=device, dtype=dtype)
            depth_mask = camera.depth_mask.to(device=device, dtype=dtype)
            if depth_mask.ndim == 2:
                depth_mask = depth_mask[None]
            depth_valid = (depth_mask[:1] * mask).clamp(0.0, 1.0)
            depth_residual = torch.abs(package["depth"] - target_depth)
            depth = (depth_residual * depth_valid).sum() / depth_valid.sum().clamp_min(1.0)

        view_loss = (
            float(args.lambda_rgb) * rgb
            + float(args.lambda_ssim) * dssim
            + float(args.lambda_depth) * depth
        )
        denom = max(1, len(cameras))
        loss = loss + view_loss / denom
        rgb_loss = rgb_loss + rgb.detach() / denom
        ssim_loss = ssim_loss + dssim.detach() / denom
        depth_loss = depth_loss + depth.detach() / denom

        if collect_evidence:
            rgb_residual = abs_rgb.detach().mean(0, keepdim=True)
            stats = accumulate_pixel_features_to_gaussians(
                camera,
                gaussians,
                pipe,
                rgb_residual,
                mask.detach(),
                query_positions=xyz.detach(),
                query_rotations=rotations.detach(),
            )
            rgb_sum += stats["feature_sum"][:, 0].float()
            rgb_weight += stats["valid_visible_mass"].float()
            raw_support += stats["raw_support"].float()
            visible_mass += stats["visible_mass"].float()
            valid_visible_mass += stats["valid_visible_mass"].float()

            if depth_residual is not None and depth_valid is not None:
                depth_stats = accumulate_pixel_features_to_gaussians(
                    camera,
                    gaussians,
                    pipe,
                    depth_residual.detach(),
                    depth_valid.detach(),
                    query_positions=xyz.detach(),
                    query_rotations=rotations.detach(),
                )
                depth_sum += depth_stats["feature_sum"][:, 0].float()
                depth_weight += depth_stats["valid_visible_mass"].float()

    eps = 1.0e-8
    rgb_evidence = rgb_sum / rgb_weight.clamp_min(eps)
    depth_evidence = depth_sum / depth_weight.clamp_min(eps)
    evidence = {
        "rgb_evidence": rgb_evidence,
        "depth_evidence": depth_evidence,
        "combined_residual": rgb_evidence + float(args.lambda_depth) * depth_evidence,
        "raw_support": raw_support,
        "visible_mass": visible_mass,
        "valid_visible_mass": valid_visible_mass,
        "visibility_ratio": (visible_mass / raw_support.clamp_min(eps)).clamp(0.0, 1.0),
        "valid_fraction": (valid_visible_mass / visible_mass.clamp_min(eps)).clamp(0.0, 1.0),
    }
    return {
        "loss": loss,
        "rgb": rgb_loss,
        "ssim": ssim_loss,
        "depth": depth_loss,
        "evidence": evidence,
    }



def _render_loss_evidence_and_xyz_grad_memory_safe(
    cameras,
    gaussians,
    pipe,
    background,
    xyz,
    rotations,
    args,
    *,
    collect_evidence: bool,
):
    """Memory-safe multiview render supervision.

    Important ordering for each camera:
      1) build ONE differentiable render/SSIM/depth graph;
      2) compute dL_view/dxyz with autograd.grad();
      3) release that graph immediately;
      4) only then compute detached reverse-raster evidence;
      5) move to the next camera.

    This avoids retaining all camera graphs and all recurrent-step renderer graphs
    simultaneously. The detached xyz gradient is later used in a linear surrogate
    objective. Since the canonical 3DGS is frozen and the renderer loss reaches the
    denoiser only through xyz, the surrogate has the same first-order gradient with
    respect to denoiser parameters as the original render loss at the current point.
    """
    if not xyz.requires_grad:
        raise RuntimeError("xyz must require gradients in memory-safe training")

    n = int(xyz.shape[0])
    device, dtype = xyz.device, xyz.dtype
    denom = max(1, len(cameras))

    # Detached numerical loss values for logging.
    loss_value = xyz.new_zeros(())
    rgb_loss = xyz.new_zeros(())
    ssim_loss = xyz.new_zeros(())
    depth_loss = xyz.new_zeros(())

    # Exact average d(render loss)/dxyz, detached from the heavy renderer graph.
    xyz_grad = torch.zeros_like(xyz)

    # Detached evidence accumulators.
    rgb_sum = torch.zeros(n, device=device, dtype=torch.float32)
    rgb_weight = torch.zeros(n, device=device, dtype=torch.float32)
    depth_sum = torch.zeros(n, device=device, dtype=torch.float32)
    depth_weight = torch.zeros(n, device=device, dtype=torch.float32)
    raw_support = torch.zeros(n, device=device, dtype=torch.float32)
    visible_mass = torch.zeros(n, device=device, dtype=torch.float32)
    valid_visible_mass = torch.zeros(n, device=device, dtype=torch.float32)

    for camera in cameras:
        package = render(
            camera,
            gaussians,
            pipe,
            background,
            use_trained_exp=False,
            means3D_override=xyz,
            rotations_override=rotations,
        )
        image = package["render"]
        target = camera.original_image.to(device=device, dtype=dtype)
        mask = _image_mask(camera, device, dtype)

        abs_rgb = torch.abs(image - target)
        rgb = (abs_rgb * mask).sum() / mask.expand_as(abs_rgb).sum().clamp_min(1.0)
        dssim = 1.0 - ssim(image * mask, target * mask)

        depth = image.new_zeros(())
        depth_residual = None
        depth_valid = None
        if bool(args.use_depth) and bool(getattr(camera, "depth_reliable", False)):
            target_depth = camera.invdepthmap.to(device=device, dtype=dtype)
            depth_mask = camera.depth_mask.to(device=device, dtype=dtype)
            if depth_mask.ndim == 2:
                depth_mask = depth_mask[None]
            depth_valid = (depth_mask[:1] * mask).clamp(0.0, 1.0)
            depth_residual = torch.abs(package["depth"] - target_depth)
            depth = (depth_residual * depth_valid).sum() / depth_valid.sum().clamp_min(1.0)

        # Keep only detached residual images needed later for reverse accumulation.
        rgb_residual_detached = abs_rgb.detach().mean(0, keepdim=True)
        mask_detached = mask.detach()
        if depth_residual is not None and depth_valid is not None:
            depth_residual_detached = depth_residual.detach()
            depth_valid_detached = depth_valid.detach()
        else:
            depth_residual_detached = None
            depth_valid_detached = None

        view_loss = (
            float(args.lambda_rgb) * rgb
            + float(args.lambda_ssim) * dssim
            + float(args.lambda_depth) * depth
        ) / denom

        loss_value = loss_value + view_loss.detach()
        rgb_loss = rgb_loss + rgb.detach() / denom
        ssim_loss = ssim_loss + dssim.detach() / denom
        depth_loss = depth_loss + depth.detach() / denom

        # CRITICAL: consume this camera's differentiable graph BEFORE doing the
        # reverse-raster evidence pass. retain_graph=False releases render+SSIM.
        view_grad = torch.autograd.grad(
            view_loss,
            xyz,
            retain_graph=False,
            create_graph=False,
        )[0]
        xyz_grad.add_(view_grad.detach())

        # Drop every reference into the differentiable camera graph now.
        del view_grad, view_loss, rgb, dssim, depth, abs_rgb, image, package
        if depth_residual is not None:
            del depth_residual
        if depth_valid is not None:
            del depth_valid

        # Detached evidence is computed only after the heavy graph is gone.
        if collect_evidence:
            with torch.no_grad():
                stats = accumulate_pixel_features_to_gaussians(
                    camera,
                    gaussians,
                    pipe,
                    rgb_residual_detached,
                    mask_detached,
                    query_positions=xyz.detach(),
                    query_rotations=rotations.detach(),
                )
                rgb_sum += stats["feature_sum"][:, 0].float()
                rgb_weight += stats["valid_visible_mass"].float()
                raw_support += stats["raw_support"].float()
                visible_mass += stats["visible_mass"].float()
                valid_visible_mass += stats["valid_visible_mass"].float()
                del stats

                if depth_residual_detached is not None and depth_valid_detached is not None:
                    depth_stats = accumulate_pixel_features_to_gaussians(
                        camera,
                        gaussians,
                        pipe,
                        depth_residual_detached,
                        depth_valid_detached,
                        query_positions=xyz.detach(),
                        query_rotations=rotations.detach(),
                    )
                    depth_sum += depth_stats["feature_sum"][:, 0].float()
                    depth_weight += depth_stats["valid_visible_mass"].float()
                    del depth_stats

        del rgb_residual_detached, mask_detached, target, mask
        if depth_residual_detached is not None:
            del depth_residual_detached
        if depth_valid_detached is not None:
            del depth_valid_detached

    eps = 1.0e-8
    rgb_evidence = rgb_sum / rgb_weight.clamp_min(eps)
    depth_evidence = depth_sum / depth_weight.clamp_min(eps)
    evidence = {
        "rgb_evidence": rgb_evidence,
        "depth_evidence": depth_evidence,
        "combined_residual": rgb_evidence + float(args.lambda_depth) * depth_evidence,
        "raw_support": raw_support,
        "visible_mass": visible_mass,
        "valid_visible_mass": valid_visible_mass,
        "visibility_ratio": (visible_mass / raw_support.clamp_min(eps)).clamp(0.0, 1.0),
        "valid_fraction": (valid_visible_mass / visible_mass.clamp_min(eps)).clamp(0.0, 1.0),
    }
    return {
        "loss": loss_value,
        "rgb": rgb_loss,
        "ssim": ssim_loss,
        "depth": depth_loss,
        "evidence": evidence,
        "xyz_grad": xyz_grad,
    }

def _feature_tensor(canonical, current, sigma, evidence, grad, median_nn):
    valid = evidence["valid_visible_mass"] > 0
    if bool(valid.any()):
        grad_scale = torch.quantile(
            torch.linalg.norm(grad.detach(), dim=-1)[valid], 0.90
        ).clamp_min(1.0e-8)
    else:
        grad_scale = grad.new_tensor(1.0)

    nn_scale = median_nn.clamp_min(1.0e-8)
    sigma_ratio = torch.as_tensor(sigma, device=canonical.device, dtype=canonical.dtype) / nn_scale

    # EXACT v1 14-D feature vector.
    return torch.cat(
        (
            (current - canonical) / nn_scale,                                    # 3
            torch.full(
                (canonical.shape[0], 1),
                float(sigma_ratio),
                device=canonical.device,
                dtype=canonical.dtype,
            ),                                                                   # 1
            _robust_unit(evidence["rgb_evidence"], valid)[:, None].to(canonical),      # 1
            _robust_unit(evidence["depth_evidence"], valid)[:, None].to(canonical),    # 1
            (grad / grad_scale).clamp(-10.0, 10.0),                              # 3
            _robust_unit(evidence["raw_support"])[:, None].to(canonical),              # 1
            _robust_unit(evidence["visible_mass"])[:, None].to(canonical),             # 1
            _robust_unit(evidence["valid_visible_mass"])[:, None].to(canonical),       # 1
            evidence["visibility_ratio"][:, None].to(canonical),                       # 1
            evidence["valid_fraction"][:, None].to(canonical),                         # 1
        ),
        -1,
    )


def _spatial_loss(displacement, knn_idx):
    neighbors = knn_idx.to(displacement.device)
    return torch.abs(displacement[:, None, :] - displacement[neighbors]).sum(-1).mean()


def _write_csv(path: Path, rows: list[dict], columns: list[str]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _write_curve_png(path: Path, rows: list[dict], keys: list[str], title: str):
    if not rows:
        return
    width, height = 1100, 520
    pad_l, pad_r, pad_t, pad_b = 80, 30, 55, 55
    image = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(image, title, (pad_l, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    colors = [(220, 70, 40), (40, 120, 220), (45, 170, 75), (180, 75, 200), (20, 170, 190), (80, 80, 80)]
    x = np.arange(len(rows), dtype=np.float32)
    x_pix = pad_l + (width - pad_l - pad_r) * (x / max(float(len(rows) - 1), 1.0))
    for idx, key in enumerate(keys):
        y = np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=np.float32)
        finite = np.isfinite(y)
        if not finite.any():
            continue
        vals = y[finite]
        lo, hi = np.percentile(vals, [2.0, 98.0])
        if abs(hi - lo) < 1.0e-12:
            lo, hi = float(vals.min()), float(vals.max() + 1.0e-8)
        y_clip = np.clip(y, lo, hi)
        y_pix = pad_t + (height - pad_t - pad_b) * (1.0 - (y_clip - lo) / max(hi - lo, 1.0e-12))
        pts = np.stack([x_pix, y_pix], axis=1).round().astype(np.int32)
        color = colors[idx % len(colors)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(image, tuple(a), tuple(b), color, 2, cv2.LINE_AA)
        cv2.putText(image, key, (pad_l + 180 * (idx % 5), height - 18 - 22 * (idx // 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.rectangle(image, (pad_l, pad_t), (width - pad_r, height - pad_b), (0, 0, 0), 1)
    cv2.imwrite(str(path), image)


def _run_training_denoising_pass(
    denoiser,
    cameras,
    gaussians,
    pipe,
    background,
    canonical,
    rotations,
    knn_idx,
    noise,
    schedule,
    median_nn,
    args,
):
    # Canonical frame 0 + fresh artificial noise.
    current = (canonical + noise).detach().requires_grad_(True)
    iter_loss_value = current.new_zeros(())
    iter_surrogate = current.sum() * 0.0
    correction_norms = []
    num_steps = max(1, len(schedule))

    for step, sigma in enumerate(schedule):
        out = _render_loss_evidence_and_xyz_grad_memory_safe(
            cameras,
            gaussians,
            pipe,
            background,
            current,
            rotations,
            args,
            collect_evidence=True,
        )

        # Renderer guidance remains detached exactly as in successful v1.
        grad = out["xyz_grad"]
        features = _feature_tensor(
            canonical,
            current,
            sigma,
            out["evidence"],
            grad,
            median_nn,
        )

        correction = denoiser(features, knn_idx, step_scale=sigma)
        correction_norms.append(torch.linalg.norm(correction.detach(), dim=-1))

        # Numerical value of the original recurrent render objective.
        weight = float(step + 1) / num_steps
        iter_loss_value = iter_loss_value + weight * out["loss"]

        # Memory-safe surrogate. Its gradient wrt denoiser parameters is the
        # same first-order chain-rule gradient as the renderer loss because
        # grad=dL/dcurrent is evaluated at this current and then detached.
        iter_surrogate = iter_surrogate + weight * (grad * current).sum()

        current = current + correction

    final = _render_loss_evidence_and_xyz_grad_memory_safe(
        cameras,
        gaussians,
        pipe,
        background,
        current,
        rotations,
        args,
        collect_evidence=True,
    )
    final_surrogate = (final["xyz_grad"] * current).sum()

    return {
        "xyz": current,
        "final": final,
        "final_surrogate": final_surrogate,
        "iter_loss": iter_loss_value / num_steps,
        "iter_surrogate": iter_surrogate / num_steps,
        "correction_norm": torch.stack(correction_norms).mean(0),
    }

def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene, gaussians, pipe, white_background = build_scene(args)
    args.white_background = white_background
    frame_map = _frame_maps(scene)

    source_frame = int(args.source_frame)
    target_frame = int(args.target_frame)
    if source_frame not in frame_map:
        raise RuntimeError(f"source frame {source_frame} not available")
    if target_frame not in frame_map:
        raise RuntimeError(f"target frame {target_frame} not available")
    if source_frame != 0:
        print(
            f"[WARNING] This restored v1 trainer assumes the loaded static 3DGS is canonical frame 0; "
            f"requested source_frame={source_frame}."
        )

    source_time = _frame_time(frame_map[source_frame], source_frame, args.fps)
    target_time = _frame_time(frame_map[target_frame], target_frame, args.fps)
    delta_t = target_time - source_time
    if delta_t <= 0:
        delta_t = 1.0 / max(float(args.fps), 1.0e-8)

    target_cameras = sorted(frame_map[target_frame], key=_camera_id)

    device = gaussians.get_canonical_xyz.device
    dtype = gaussians.get_canonical_xyz.dtype
    canonical = gaussians.get_canonical_xyz.detach().clone()
    canonical_guard = canonical.clone()
    rotations = gaussians.get_rotation.detach().clone()
    rotations_guard = rotations.clone()

    # Freeze every canonical 3DGS parameter. Only denoiser weights are trained.
    for tensor in (
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._opacity,
        gaussians._scaling,
        gaussians._rotation,
    ):
        tensor.requires_grad_(False)

    background = torch.tensor(
        [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0],
        device=device,
        dtype=dtype,
    )

    knn_idx, median_nn = build_canonical_knn(canonical, int(args.knn))
    knn_idx = knn_idx.to(device)
    median_nn = median_nn.to(device=device, dtype=dtype)

    sigma_max = float(args.noise_max_ratio) * float(median_nn)
    sigma_min = float(args.noise_min_ratio) * float(median_nn)
    schedule = torch.logspace(
        math.log10(max(sigma_max, 1.0e-12)),
        math.log10(max(sigma_min, 1.0e-12)),
        steps=int(args.denoise_iterations),
        device=device,
        dtype=dtype,
    )
    if int(args.denoise_iterations) == 6:
        expected_sigma_max = 0.15 * float(median_nn)
        if abs(float(schedule[0]) - expected_sigma_max) > max(1.0e-10, 1.0e-5 * expected_sigma_max):
            raise RuntimeError("sigma_max does not match 0.15 * median_nn")

    denoiser = GaussianMotionDenoiser(
        input_dim=14,
        hidden_dim=int(args.hidden_dim),
        message_layers=int(args.message_passing_layers),
    ).to(device)
    if not all(p.requires_grad for p in denoiser.parameters()):
        raise RuntimeError("all denoiser parameters must require gradients during training")
    if any(
        bool(t.requires_grad)
        for t in (
            gaussians._xyz,
            gaussians._features_dc,
            gaussians._features_rest,
            gaussians._opacity,
            gaussians._scaling,
            gaussians._rotation,
        )
    ):
        raise RuntimeError("canonical Gaussian tensors must be frozen")

    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    print(f"[Train v1 denoiser] source={source_frame} target={target_frame}")
    print(f"  Gaussians={canonical.shape[0]} target_cameras={len(target_cameras)}")
    print(f"  delta_t={delta_t:.9f}s median_nn={float(median_nn):.12f}")
    print(f"  schedule={[float(x) for x in schedule.detach().cpu()]}")
    print(f"  train_cameras_per_iter={int(args.train_cameras_per_iter)} (0=all)")

    # Baseline render is diagnostic only. Do not leave a renderer/SSIM graph
    # alive before training. This was a major memory leak on large scenes.
    with torch.no_grad():
        static_eval = _render_loss_and_evidence(
            target_cameras,
            gaussians,
            pipe,
            background,
            canonical,
            rotations,
            args,
            collect_evidence=False,
        )
    static_target_render_loss = float(static_eval["loss"])
    static_target_rgb_loss = float(static_eval["rgb"])
    print(
        f"  static target render: total={static_target_render_loss:.9f} "
        f"rgb={static_target_rgb_loss:.9f}"
    )
    del static_eval
    torch.cuda.empty_cache()
    print("  memory-safe per-camera render gradients: ON")

    train_rng = torch.Generator(device="cpu").manual_seed(int(args.seed))
    last = {}
    train_rows: list[dict] = []
    csv_columns = [
        "iteration",
        "train_total",
        "train_final_render",
        "train_iter_render",
        "train_rgb",
        "train_ssim",
        "train_depth",
        "train_spatial",
        "train_motion_prior",
        "train_displacement_mean",
        "train_displacement_median",
        "train_displacement_p90",
        "train_correction_norm_mean",
        "train_correction_norm_p90",
        "train_valid_visible_fraction",
        "learning_rate",
    ]
    progress = tqdm(
        range(int(args.training_iterations)),
        desc=f"v1 denoiser 0->{target_frame}",
        unit="iter",
    )

    for iteration in progress:
        cameras = _select_random(
            target_cameras,
            int(args.train_cameras_per_iter),
            train_rng,
        )

        # EXACT original v1 behavior: fresh max-scale positional noise each training iteration.
        noise = schedule[0] * torch.randn(
            canonical.shape,
            device=device,
            dtype=dtype,
        )

        pass_out = _run_training_denoising_pass(
            denoiser,
            cameras,
            gaussians,
            pipe,
            background,
            canonical,
            rotations,
            knn_idx,
            noise,
            schedule,
            median_nn,
            args,
        )

        displacement = pass_out["xyz"] - canonical
        spatial = _spatial_loss(displacement, knn_idx)
        motion = _pseudo_huber(
            torch.linalg.norm(displacement, dim=-1),
            0.10 * float(median_nn),
        ).mean()

        # True original objective VALUE for logging/debugging. Renderer terms
        # are detached because their heavy camera graphs were already consumed.
        total = (
            pass_out["final"]["loss"]
            + float(args.lambda_iter) * pass_out["iter_loss"]
            + float(args.lambda_spatial) * spatial.detach()
            + float(args.lambda_motion) * motion.detach()
        )

        # First-order-equivalent training objective for the renderer terms.
        # Spatial and motion priors remain fully differentiable as before.
        backward_objective = (
            pass_out["final_surrogate"]
            + float(args.lambda_iter) * pass_out["iter_surrogate"]
            + float(args.lambda_spatial) * spatial
            + float(args.lambda_motion) * motion
        )

        if not bool(torch.isfinite(total)) or not bool(torch.isfinite(backward_objective)):
            raise RuntimeError("non-finite denoiser training loss")

        optimizer.zero_grad(set_to_none=True)
        backward_objective.backward()
        optimizer.step()

        disp_norm = torch.linalg.norm(displacement.detach(), dim=-1)
        corr_norm = pass_out["correction_norm"]
        valid_visible_fraction = float(
            (pass_out["final"]["evidence"]["valid_visible_mass"] > 0).float().mean()
        )

        last = {
            "iteration": int(iteration + 1),
            "train_total": float(total.detach()),
            "train_final_render": float(pass_out["final"]["loss"].detach()),
            "train_iter_render": float(pass_out["iter_loss"].detach()),
            "train_rgb": float(pass_out["final"]["rgb"]),
            "train_ssim": float(pass_out["final"]["ssim"]),
            "train_depth": float(pass_out["final"]["depth"]),
            "train_spatial": float(spatial.detach()),
            "train_motion_prior": float(motion.detach()),
            "train_displacement_mean": float(disp_norm.mean()),
            "train_displacement_median": float(disp_norm.median()),
            "train_displacement_p90": float(torch.quantile(disp_norm, 0.90)),
            "train_correction_norm_mean": float(corr_norm.mean()),
            "train_correction_norm_p90": float(torch.quantile(corr_norm, 0.90)),
            "train_valid_visible_fraction": valid_visible_fraction,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        train_rows.append(last)

        progress.set_postfix(
            loss=f"{last['train_total']:.6f}",
            rgb=f"{last['train_rgb']:.6f}",
            disp=f"{last['train_displacement_mean']:.5f}",
        )

    canonical_change = float(
        (gaussians.get_canonical_xyz.detach() - canonical_guard).abs().max()
    )
    if canonical_change >= 1.0e-10:
        raise RuntimeError(f"canonical Gaussian geometry changed: {canonical_change}")
    rotation_change = float(
        (gaussians.get_rotation.detach() - rotations_guard).abs().max()
    )
    if rotation_change >= 1.0e-10:
        raise RuntimeError(f"canonical Gaussian rotations changed: {rotation_change}")

    _write_csv(output_dir / "train_log.csv", train_rows, csv_columns)
    _write_curve_png(
        output_dir / "train_loss_curves.png",
        train_rows,
        ["train_total", "train_final_render", "train_iter_render", "train_rgb", "train_depth"],
        f"Train loss curves 0->{target_frame}",
    )
    _write_curve_png(
        output_dir / "train_curves.png",
        train_rows,
        ["train_total", "train_final_render", "train_rgb", "train_depth", "train_displacement_mean", "train_valid_visible_fraction"],
        f"Train curves 0->{target_frame}",
    )
    _write_curve_png(
        output_dir / "train_displacement_curves.png",
        train_rows,
        ["train_displacement_mean", "train_displacement_median", "train_displacement_p90", "train_correction_norm_mean", "train_correction_norm_p90"],
        f"Train displacement/correction curves 0->{target_frame}",
    )
    _write_curve_png(
        output_dir / "train_visibility_curves.png",
        train_rows,
        ["train_valid_visible_fraction"],
        f"Train visibility curves 0->{target_frame}",
    )

    checkpoint_path = output_dir / f"denoiser_frame0_to_{target_frame}.pt"
    payload = {
        "state_dict": denoiser.state_dict(),
        "network_config": {
            "input_dim": 14,
            "hidden_dim": int(args.hidden_dim),
            "message_passing_layers": int(args.message_passing_layers),
        },
        "knn": int(args.knn),
        "noise_schedule": schedule.detach().cpu(),
        "median_nn_distance": float(median_nn),
        "delta_t": float(delta_t),
        "source_frame": source_frame,
        "target_frame": target_frame,
        "training_config": vars(args),
    }
    torch.save(payload, checkpoint_path)

    summary = {
        "source_frame": source_frame,
        "target_frame": target_frame,
        "num_gaussians": int(canonical.shape[0]),
        "num_target_cameras": int(len(target_cameras)),
        "delta_t": float(delta_t),
        "median_nn_distance": float(median_nn),
        "sigma_max": float(schedule[0]),
        "noise_schedule": [float(x) for x in schedule.detach().cpu()],
        "training_iterations": int(args.training_iterations),
        "static_target_render_loss": static_target_render_loss,
        "static_target_rgb_loss": static_target_rgb_loss,
        "canonical_geometry_max_change": canonical_change,
        "canonical_rotation_max_change": rotation_change,
        "checkpoint": str(checkpoint_path),
        "final_training": last,
    }
    summary_path = output_dir / f"train_frame0_to_{target_frame}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[COMPLETE]")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  summary:    {summary_path}")
    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-scene", required=True)
    p.add_argument("--static-model-dir", required=True)
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--source-frame", type=int, default=0)
    p.add_argument("--target-frame", type=int, default=1)

    # Exact successful v1 defaults.
    p.add_argument("--training-iterations", type=int, default=1000)
    p.add_argument("--denoise-iterations", type=int, default=6)
    p.add_argument("--knn", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--message-passing-layers", type=int, default=2)
    p.add_argument("--noise-max-ratio", type=float, default=0.15)
    p.add_argument("--noise-min-ratio", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-5)
    p.add_argument("--lambda-rgb", type=float, default=1.0)
    p.add_argument("--lambda-ssim", type=float, default=0.2)
    p.add_argument("--lambda-depth", type=float, default=0.1)
    p.add_argument("--lambda-iter", type=float, default=0.25)
    p.add_argument("--lambda-spatial", type=float, default=0.005)
    p.add_argument("--lambda-motion", type=float, default=0.002)
    p.add_argument("--train-cameras-per-iter", type=int, default=0)

    p.add_argument("--use-depth", action=BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--sh-degree", type=int, default=None)
    p.add_argument("--cameras", default="")
    p.add_argument(
        "--camera-data-device",
        default="cpu",
        help="Device for cached camera RGB/depth tensors; cpu keeps only the selected render batch on CUDA.",
    )

    bg = p.add_mutually_exclusive_group()
    bg.add_argument("--white-background", dest="white_background", action="store_true")
    bg.add_argument("--black-background", dest="white_background", action="store_false")
    p.set_defaults(white_background=None)

    args = p.parse_args()
    args.training_iterations = max(1, int(args.training_iterations))
    args.denoise_iterations = max(1, int(args.denoise_iterations))
    args.knn = max(1, int(args.knn))
    args.hidden_dim = max(1, int(args.hidden_dim))
    args.message_passing_layers = max(1, int(args.message_passing_layers))
    args.train_cameras_per_iter = max(0, int(args.train_cameras_per_iter))
    args.lambda_rgb = max(0.0, float(args.lambda_rgb))
    args.lambda_ssim = max(0.0, float(args.lambda_ssim))
    args.lambda_depth = max(0.0, float(args.lambda_depth))
    args.lambda_iter = max(0.0, float(args.lambda_iter))
    args.lambda_spatial = max(0.0, float(args.lambda_spatial))
    args.lambda_motion = max(0.0, float(args.lambda_motion))
    args.noise_max_ratio = max(1.0e-8, float(args.noise_max_ratio))
    args.noise_min_ratio = max(
        1.0e-8,
        min(float(args.noise_min_ratio), args.noise_max_ratio),
    )
    return args


if __name__ == "__main__":
    train(parse_args())
