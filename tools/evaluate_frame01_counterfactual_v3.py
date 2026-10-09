#!/usr/bin/env python3
"""Minimal, interpretable frame-0 -> frame-1 motion diagnostic.

Purpose
-------
Do NOT train a new classifier and do NOT fine-tune the denoiser.

1. Load the already-working v1 frame01 denoiser checkpoint.
2. Freeze it completely.
3. Run multiple independent positional-noise trials on frame 0.
4. For each Gaussian compare two hypotheses against the real multiview frame-1 RGB-D:
       STATIC: canonical frame-0 position
       MOVED:  v1 denoiser proposed position
5. Save each evidence signal separately.
6. If SAPIEN GT `points3D_parts.ply` is available, evaluate every signal separately.

There is no weighted combination of classification losses in this script.
The only learned model is the frozen v1 denoiser already trained in the previous experiment.

Interpretation:
    counterfactual RGB score > 0   -> moved hypothesis explains RGB better
    counterfactual RGB score < 0   -> static hypothesis explains RGB better

The same is reported independently for depth, total residual, motion magnitude,
and noise-seed consistency, so we can measure which signal actually predicts
GT static/dynamic labels before adding another learned head.
"""

from __future__ import annotations

import argparse
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
            "tools/evaluate_frame01_counterfactual_v4.py must live in RigEuler3DGS/tools"
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
    return str(
        getattr(
            camera,
            "dynamic_camera_id",
            getattr(camera, "image_name", "camera"),
        )
    )


def _image_mask(camera, device, dtype):
    mask = torch.ones(
        (1, int(camera.image_height), int(camera.image_width)),
        device=device,
        dtype=dtype,
    )
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
    lo, hi = torch.quantile(
        vals,
        torch.tensor([0.10, 0.90], device=x.device),
    )
    denom = (hi - lo).clamp_min(1.0e-8)
    return torch.where(
        mask,
        ((x - lo) / denom).clamp(0.0, 1.0),
        torch.zeros_like(x),
    )


def _quantiles(value: torch.Tensor, probs=(0.10, 0.50, 0.90, 0.99)):
    x = torch.as_tensor(value).detach().float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {f"p{int(p * 100):02d}": None for p in probs}
    q = torch.quantile(x, torch.tensor(probs, device=x.device))
    return {
        f"p{int(p * 100):02d}": float(v)
        for p, v in zip(probs, q)
    }


def _to_png(tensor):
    array = torch.as_tensor(tensor).detach().cpu().float()
    if array.ndim == 2:
        array = array[None]
    if array.shape[0] == 1:
        array = array.repeat(3, 1, 1)
    array = (
        array[:3]
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .numpy()
    )
    return (array * 255.0 + 0.5).astype(np.uint8)


def _frame_maps(scene):
    groups = _ordered_complete_frame_groups(scene.getTrainCameras())
    if len(groups) < 1:
        raise RuntimeError("No complete dynamic multiview frame groups were built")
    return {int(frame): list(cameras) for frame, cameras in groups}


def _frame_time(cameras, frame_id: int, fps: float):
    if cameras:
        return float(
            getattr(
                cameras[0],
                "timestamp",
                float(frame_id) / float(fps),
            )
        )
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
    """Exact frame01 RGB-D rendering/evidence logic, local to this evaluator."""
    n = int(xyz.shape[0])
    device, dtype = xyz.device, xyz.dtype

    loss = xyz.sum() * 0.0
    rgb_sum = torch.zeros(n, device=device)
    rgb_weight = torch.zeros(n, device=device)
    depth_sum = torch.zeros(n, device=device)
    depth_weight = torch.zeros(n, device=device)
    raw_support = torch.zeros(n, device=device)
    visible_mass = torch.zeros(n, device=device)
    valid_visible_mass = torch.zeros(n, device=device)

    rgb_loss = xyz.new_zeros(())
    ssim_loss = xyz.new_zeros(())
    depth_loss = xyz.new_zeros(())

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
        rgb = (
            (abs_rgb * mask).sum()
            / mask.expand_as(abs_rgb).sum().clamp_min(1.0)
        )
        dssim = 1.0 - ssim(image * mask, target * mask)

        depth = image.new_zeros(())
        depth_residual = None
        depth_valid = None

        if bool(args.use_depth) and bool(
            getattr(camera, "depth_reliable", False)
        ):
            target_depth = camera.invdepthmap.to(
                device=device,
                dtype=dtype,
            )
            depth_mask = camera.depth_mask.to(
                device=device,
                dtype=dtype,
            )
            if depth_mask.ndim == 2:
                depth_mask = depth_mask[None]

            depth_valid = (depth_mask[:1] * mask).clamp(0.0, 1.0)
            depth_residual = torch.abs(package["depth"] - target_depth)
            depth = (
                (depth_residual * depth_valid).sum()
                / depth_valid.sum().clamp_min(1.0)
            )

        view_loss = (
            float(args.lambda_rgb) * rgb
            + float(args.lambda_ssim) * dssim
            + float(args.lambda_depth) * depth
        )
        denom_views = max(1, len(cameras))
        loss = loss + view_loss / denom_views
        rgb_loss = rgb_loss + rgb.detach() / denom_views
        ssim_loss = ssim_loss + dssim.detach() / denom_views
        depth_loss = depth_loss + depth.detach() / denom_views

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
        "combined_residual": (
            rgb_evidence
            + float(args.lambda_depth) * depth_evidence
        ),
        "raw_support": raw_support,
        "visible_mass": visible_mass,
        "valid_visible_mass": valid_visible_mass,
        "visibility_ratio": (
            visible_mass / raw_support.clamp_min(eps)
        ).clamp(0.0, 1.0),
        "valid_fraction": (
            valid_visible_mass / visible_mass.clamp_min(eps)
        ).clamp(0.0, 1.0),
    }

    return {
        "loss": loss,
        "rgb": rgb_loss,
        "ssim": ssim_loss,
        "depth": depth_loss,
        "evidence": evidence,
    }


def _feature_tensor(
    canonical,
    current,
    sigma,
    evidence,
    grad,
    median_nn,
):
    valid = evidence["valid_visible_mass"] > 0
    if bool(valid.any()):
        grad_scale = torch.quantile(
            torch.linalg.norm(grad.detach(), dim=-1)[valid],
            0.90,
        ).clamp_min(1.0e-8)
    else:
        grad_scale = grad.new_tensor(1.0)

    nn_scale = median_nn.clamp_min(1.0e-8)
    sigma_ratio = torch.as_tensor(
        sigma,
        device=canonical.device,
        dtype=canonical.dtype,
    ) / nn_scale

    return torch.cat(
        (
            (current - canonical) / nn_scale,
            torch.full(
                (canonical.shape[0], 1),
                float(sigma_ratio),
                device=canonical.device,
                dtype=canonical.dtype,
            ),
            _robust_unit(
                evidence["rgb_evidence"],
                valid,
            )[:, None].to(canonical),
            _robust_unit(
                evidence["depth_evidence"],
                valid,
            )[:, None].to(canonical),
            (grad / grad_scale).clamp(-10.0, 10.0),
            _robust_unit(
                evidence["raw_support"],
            )[:, None].to(canonical),
            _robust_unit(
                evidence["visible_mass"],
            )[:, None].to(canonical),
            _robust_unit(
                evidence["valid_visible_mass"],
            )[:, None].to(canonical),
            evidence["visibility_ratio"][:, None].to(canonical),
            evidence["valid_fraction"][:, None].to(canonical),
        ),
        -1,
    )


def _run_v1_denoising_pass(
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
    *,
    keep_trajectory: bool = False,
):
    """Exact inference semantics of the original successful v1 experiment.

    Important: the original GaussianMotionDenoiser returns ONE tensor:
        correction = denoiser(features, knn_idx, step_scale=sigma)

    Do not unpack it as a tuple.
    """
    current = (canonical + noise).detach().requires_grad_(True)
    trajectory = [current.detach().cpu()] if keep_trajectory else None
    correction_norms = []

    for sigma in schedule:
        out = _render_loss_and_evidence(
            cameras,
            gaussians,
            pipe,
            background,
            current,
            rotations,
            args,
            collect_evidence=True,
        )

        # Same render-gradient feature as v1. We only need the gradient w.r.t.
        # current xyz; no training graph is retained.
        grad = torch.autograd.grad(
            out["loss"],
            current,
            retain_graph=False,
            create_graph=False,
        )[0].detach()

        features = _feature_tensor(
            canonical,
            current,
            sigma,
            out["evidence"],
            grad,
            median_nn,
        )

        with torch.no_grad():
            correction = denoiser(
                features,
                knn_idx,
                step_scale=sigma,
            )

        if not torch.is_tensor(correction):
            raise RuntimeError(
                "Expected original v1 GaussianMotionDenoiser to return one "
                f"Tensor correction, got {type(correction)}"
            )
        if correction.shape != canonical.shape:
            raise RuntimeError(
                f"Bad v1 correction shape {tuple(correction.shape)}; "
                f"expected {tuple(canonical.shape)}"
            )

        correction_norms.append(
            torch.linalg.norm(correction.detach(), dim=-1)
        )

        # Exact original inference update:
        # detach old state, add correction, re-enable xyz gradient for the next render.
        current = (
            current.detach() + correction
        ).requires_grad_(True)

        if keep_trajectory:
            trajectory.append(current.detach().cpu())

    final = _render_loss_and_evidence(
        cameras,
        gaussians,
        pipe,
        background,
        current,
        rotations,
        args,
        collect_evidence=True,
    )

    return {
        "xyz": current,
        "final": final,
        "correction_norm": torch.stack(correction_norms).mean(0),
        "trajectory": trajectory,
    }


def _load_v1_checkpoint(path: str, device):
    ckpt_path = Path(path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"v1 checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError("Expected v1 checkpoint dict with a state_dict")

    cfg = payload.get("network_config", {})
    input_dim = int(cfg.get("input_dim", 14))
    hidden_dim = int(cfg.get("hidden_dim", 96))
    message_layers = int(cfg.get("message_passing_layers", cfg.get("message_layers", 2)))

    model = GaussianMotionDenoiser(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        message_layers=message_layers,
    ).to(device)

    # Strict on purpose. This is the known-good v1 model; never partially load it.
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    import inspect
    print(f"[Checkpoint] Loaded v1 STRICTLY: {ckpt_path}")
    print(
        f"[Checkpoint] input_dim={input_dim} hidden_dim={hidden_dim} "
        f"message_layers={message_layers}"
    )
    print(f"[Checkpoint] class source: {inspect.getsourcefile(GaussianMotionDenoiser)}")
    print(f"[Checkpoint] forward API: {inspect.signature(GaussianMotionDenoiser.forward)}")
    return model, payload


def _resolve_schedule(payload, median_nn, device, dtype, args):
    # Default: reproduce the exact schedule used by the working v1 checkpoint.
    if not args.override_noise_schedule and "noise_schedule" in payload:
        schedule = torch.as_tensor(
            payload["noise_schedule"], device=device, dtype=dtype
        ).flatten()
        if schedule.numel() == 0:
            raise RuntimeError("checkpoint noise_schedule is empty")
        print(
            "[Schedule] Using checkpoint schedule:",
            [float(x) for x in schedule.detach().cpu()],
        )
        return schedule

    sigma_max = float(args.noise_max_ratio) * float(median_nn)
    sigma_min = float(args.noise_min_ratio) * float(median_nn)
    schedule = torch.logspace(
        math.log10(max(sigma_max, 1.0e-12)),
        math.log10(max(sigma_min, 1.0e-12)),
        steps=int(args.denoise_iterations),
        device=device,
        dtype=dtype,
    )
    print(
        "[Schedule] OVERRIDDEN schedule:",
        [float(x) for x in schedule.detach().cpu()],
    )
    return schedule


def _weighted_trial_mean(value_stack, weight_stack, eps=1e-8):
    # [S,N], [S,N] -> [N]
    return (value_stack * weight_stack).sum(0) / weight_stack.sum(0).clamp_min(eps)


def _signed_log_ratio(static_value, moved_value, eps=1e-6):
    """Positive means MOVED is better; negative means STATIC is better.

    log((static+eps)/(moved+eps)) is symmetric around zero and behaves better
    than a relative ratio when the static residual is very small.
    """
    return torch.log(
        (static_value.clamp_min(0.0) + eps)
        / (moved_value.clamp_min(0.0) + eps)
    )


def _nearest_gt_labels(canonical_xyz, ply_path: Path):
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "part"}
    missing = required - names
    if missing:
        raise RuntimeError(
            f"{ply_path} is missing GT fields {sorted(missing)}; fields={sorted(names)}"
        )

    gt_xyz = np.stack(
        [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])],
        axis=1,
    ).astype(np.float32)
    gt_part = np.asarray(vertex["part"]).astype(np.int64)
    gt_dynamic = (gt_part != 0).astype(np.int64)

    query = canonical_xyz.detach().cpu().numpy().astype(np.float32)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(gt_xyz)
        distance, index = tree.query(query, k=1)
    except Exception:
        # Dependency-safe chunked fallback.
        gt = torch.from_numpy(gt_xyz)
        q = torch.from_numpy(query)
        distance_chunks = []
        index_chunks = []
        chunk = 512
        for start in range(0, q.shape[0], chunk):
            d = torch.cdist(q[start : start + chunk], gt)
            dv, di = d.min(dim=1)
            distance_chunks.append(dv)
            index_chunks.append(di)
        distance = torch.cat(distance_chunks).numpy()
        index = torch.cat(index_chunks).numpy()

    labels = gt_dynamic[index]
    parts = gt_part[index]
    return (
        torch.from_numpy(labels).long(),
        torch.from_numpy(parts).long(),
        torch.from_numpy(distance.astype(np.float32)),
    )


def _binary_metrics(labels, scores):
    """Dependency-free ROC-AUC, AP, and best F1 threshold.

    Scores can be any real-valued dynamic evidence. Higher must mean "more dynamic".
    """
    y = torch.as_tensor(labels).detach().cpu().numpy().astype(np.int64)
    s = torch.as_tensor(scores).detach().cpu().numpy().astype(np.float64)

    valid = np.isfinite(s) & ((y == 0) | (y == 1))
    y = y[valid]
    s = s[valid]
    n = int(y.size)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n == 0 or n_pos == 0 or n_neg == 0:
        return {
            "n": n,
            "n_dynamic": n_pos,
            "n_static": n_neg,
            "roc_auc": None,
            "pr_auc": None,
            "best_f1": None,
            "best_threshold": None,
            "precision_at_best": None,
            "recall_at_best": None,
            "accuracy_at_best": None,
            "iou_at_best": None,
        }

    # Stable descending ordering.
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ss = s[order]

    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    fn = n_pos - tp
    tn = n_neg - fp

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    accuracy = (tp + tn) / n
    iou = tp / np.maximum(tp + fp + fn, 1)

    best = int(np.nanargmax(f1))

    # Fixed physically interpretable threshold: score > 0 means MOVED beats STATIC.
    pred0 = s > 0.0
    tp0 = int(np.sum(pred0 & (y == 1)))
    fp0 = int(np.sum(pred0 & (y == 0)))
    fn0 = int(np.sum((~pred0) & (y == 1)))
    tn0 = int(np.sum((~pred0) & (y == 0)))
    precision0 = tp0 / max(tp0 + fp0, 1)
    recall0 = tp0 / max(tp0 + fn0, 1)
    f10 = 2.0 * precision0 * recall0 / max(precision0 + recall0, 1e-12)
    iou0 = tp0 / max(tp0 + fp0 + fn0, 1)
    accuracy0 = (tp0 + tn0) / max(n, 1)

    # ROC AUC by trapezoidal integration, adding (0,0) and (1,1).
    tpr = recall.astype(np.float64)
    fpr = (fp / max(n_neg, 1)).astype(np.float64)
    # Version-safe trapezoidal ROC integral implemented directly.
    roc_x = np.concatenate([[0.0], fpr, [1.0]]).astype(np.float64)
    roc_y = np.concatenate([[0.0], tpr, [1.0]]).astype(np.float64)
    roc_auc = float(
        np.sum(
            (roc_x[1:] - roc_x[:-1])
            * 0.5
            * (roc_y[1:] + roc_y[:-1])
        )
    )

    # Average precision / step-integral PR AUC.
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    pr_auc = float(np.sum((recall - recall_prev) * precision))

    return {
        "n": n,
        "n_dynamic": n_pos,
        "n_static": n_neg,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_f1": float(f1[best]),
        "best_threshold": float(ss[best]),
        "precision_at_best": float(precision[best]),
        "recall_at_best": float(recall[best]),
        "accuracy_at_best": float(accuracy[best]),
        "iou_at_best": float(iou[best]),
        "threshold_zero": {
            "precision": float(precision0),
            "recall": float(recall0),
            "f1": float(f10),
            "iou": float(iou0),
            "accuracy": float(accuracy0),
            "tp": tp0,
            "fp": fp0,
            "fn": fn0,
            "tn": tn0,
        },
    }


def _evaluate_signals_against_gt(
    labels,
    gt_distance,
    observed,
    signals,
    max_gt_distance,
):
    reliable = gt_distance <= float(max_gt_distance)
    visible = observed & reliable

    result = {}
    for subset_name, mask in (
        ("gt_reliable_all", reliable),
        ("gt_reliable_visible", visible),
    ):
        result[subset_name] = {}
        for name, score in signals.items():
            result[subset_name][name] = _binary_metrics(labels[mask], score[mask])
    return result


def _write_score_html(path, canonical, score, observed, max_points):
    xyz = canonical.detach().cpu().float()
    score = score.detach().cpu().float()
    observed = observed.detach().cpu().bool()

    if xyz.shape[0] > max_points:
        ids = (
            torch.linspace(0, xyz.shape[0] - 1, max_points)
            .round()
            .long()
            .unique()
        )
        xyz, score, observed = xyz[ids], score[ids], observed[ids]

    finite = score[torch.isfinite(score)]
    if finite.numel() > 0:
        scale = float(torch.quantile(finite.abs(), 0.95).clamp_min(1e-6))
    else:
        scale = 1.0

    colors = score.clamp(-scale, scale).tolist()
    text = [
        f"counterfactual={float(s):+.5f} observed={bool(v)}"
        for s, v in zip(score, observed)
    ]
    trace = {
        "type": "scatter3d",
        "mode": "markers",
        "x": xyz[:, 0].tolist(),
        "y": xyz[:, 1].tolist(),
        "z": xyz[:, 2].tolist(),
        "marker": {
            "size": 3,
            "color": colors,
            "colorscale": "RdBu",
            "reversescale": True,
            "cmin": -scale,
            "cmax": scale,
            "opacity": 0.8,
            "colorbar": {"title": "log static/moved"},
        },
        "text": text,
    }
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body><div id="p" style="height:95vh"></div>
<script>
Plotly.newPlot('p',{json.dumps([trace])},
{{scene:{{aspectmode:'data'}},title:'Frame 0→1 counterfactual RGB evidence'}});
</script></body></html>"""
    path.write_text(html, encoding="utf-8")


def _write_camera_comparison(
    output_dir,
    cameras,
    gaussians,
    pipe,
    background,
    canonical,
    rotations,
    moved_xyz,
):
    from gaussian_renderer import render

    debug = output_dir / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        gt = camera.original_image.to(canonical)
        static = render(
            camera,
            gaussians,
            pipe,
            background,
            means3D_override=canonical,
            rotations_override=rotations,
        )["render"]
        moved = render(
            camera,
            gaussians,
            pipe,
            background,
            means3D_override=moved_xyz,
            rotations_override=rotations,
        )["render"]
        static_res = torch.abs(static - gt).mean(0)
        moved_res = torch.abs(moved - gt).mean(0)
        improvement = (static_res - moved_res)
        # Map signed improvement around 0 to [0,1] for visualization only.
        lim = torch.quantile(improvement.abs().flatten(), 0.99).clamp_min(1e-8)
        improvement_vis = (0.5 + 0.5 * improvement / lim).clamp(0, 1)

        panel = np.concatenate(
            [
                _to_png(gt),
                _to_png(static),
                _to_png(moved),
                _to_png(static_res),
                _to_png(moved_res),
                _to_png(improvement_vis),
            ],
            axis=1,
        )
        cv2.imwrite(
            str(debug / f"{_camera_id(camera)}_counterfactual.png"),
            cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
        )


def run(args):
    print("[COUNTERFACTUAL V4] SELF-CONTAINED evaluator; version-safe GT metrics")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene, gaussians, pipe, white_background = build_scene(args)
    args.white_background = white_background
    frame_map = _frame_maps(scene)

    if int(args.source_frame) not in frame_map:
        raise RuntimeError(f"source frame {args.source_frame} not available")
    if int(args.target_frame) not in frame_map:
        raise RuntimeError(f"target frame {args.target_frame} not available")

    source_cameras = frame_map[int(args.source_frame)]
    target_cameras = sorted(frame_map[int(args.target_frame)], key=_camera_id)

    source_time = _frame_time(source_cameras, int(args.source_frame), float(args.fps))
    target_time = _frame_time(target_cameras, int(args.target_frame), float(args.fps))
    delta_t = target_time - source_time
    if delta_t <= 0:
        delta_t = 1.0 / float(args.fps)

    device = gaussians.get_canonical_xyz.device
    dtype = gaussians.get_canonical_xyz.dtype
    canonical = gaussians.get_canonical_xyz.detach().clone()
    rotations = gaussians.get_rotation.detach().clone()
    canonical_guard = canonical.clone()

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

    denoiser, ckpt = _load_v1_checkpoint(args.denoiser_checkpoint, device)

    knn_idx, median_nn = build_canonical_knn(canonical, int(args.knn))
    knn_idx = knn_idx.to(device)
    median_nn_dev = median_nn.to(device=device, dtype=dtype)
    schedule = _resolve_schedule(
        ckpt, median_nn, device, dtype, args
    )

    print(f"[Scene] Gaussians={canonical.shape[0]} cameras={len(target_cameras)}")
    print(f"[Scene] median_nn={float(median_nn):.8f} delta_t={delta_t:.8f}")
    print("[Mode] FROZEN denoiser. ZERO classification-training losses.")

    # H0: STATIC
    static_eval = _render_loss_and_evidence(
        target_cameras,
        gaussians,
        pipe,
        background,
        canonical,
        rotations,
        args,
        collect_evidence=True,
    )
    static_e = static_eval["evidence"]

    trial_displacements = []
    trial_rgb_evidence = []
    trial_depth_evidence = []
    trial_combined_evidence = []
    trial_valid_mass = []
    trial_raw_support = []
    trial_visible_mass = []

    first_moved_xyz = None
    first_trajectory = None

    for trial in tqdm(
        range(int(args.inference_trials)),
        desc="Frozen v1 counterfactual trials",
        unit="trial",
    ):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.seed) + 100_000 + trial)
        noise = schedule[0] * torch.randn(
            canonical.shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        out = _run_v1_denoising_pass(
            denoiser,
            target_cameras,
            gaussians,
            pipe,
            background,
            canonical,
            rotations,
            knn_idx,
            noise,
            schedule,
            median_nn_dev,
            args,
            keep_trajectory=(trial == 0),
        )

        moved_xyz = out["xyz"].detach()
        if first_moved_xyz is None:
            first_moved_xyz = moved_xyz
            first_trajectory = out["trajectory"]

        d = moved_xyz - canonical
        trial_displacements.append(d)

        e = out["final"]["evidence"]
        trial_rgb_evidence.append(e["rgb_evidence"].detach())
        trial_depth_evidence.append(e["depth_evidence"].detach())
        trial_combined_evidence.append(e["combined_residual"].detach())
        trial_valid_mass.append(e["valid_visible_mass"].detach())
        trial_raw_support.append(e["raw_support"].detach())
        trial_visible_mass.append(e["visible_mass"].detach())

    trial_displacements = torch.stack(trial_displacements)       # [S,N,3]
    rgb_stack = torch.stack(trial_rgb_evidence)                  # [S,N]
    depth_stack = torch.stack(trial_depth_evidence)
    combined_stack = torch.stack(trial_combined_evidence)
    valid_stack = torch.stack(trial_valid_mass)
    raw_stack = torch.stack(trial_raw_support)
    visible_stack = torch.stack(trial_visible_mass)

    mean_displacement = trial_displacements.mean(0)
    mean_moved_xyz = canonical + mean_displacement
    motion_magnitude = torch.linalg.norm(mean_displacement, dim=-1)
    trial_norm = torch.linalg.norm(trial_displacements, dim=-1)
    seed_consistency = (
        motion_magnitude / trial_norm.mean(0).clamp_min(1e-8)
    ).clamp(0.0, 1.0)

    moved_rgb = _weighted_trial_mean(rgb_stack, valid_stack)
    moved_depth = _weighted_trial_mean(depth_stack, valid_stack)
    moved_combined = _weighted_trial_mean(combined_stack, valid_stack)

    # Direct, individually interpretable signals. NO product/composite score.
    rgb_counterfactual = _signed_log_ratio(
        static_e["rgb_evidence"], moved_rgb, eps=float(args.score_epsilon)
    )
    depth_counterfactual = _signed_log_ratio(
        static_e["depth_evidence"], moved_depth, eps=float(args.score_epsilon)
    )
    combined_counterfactual = _signed_log_ratio(
        static_e["combined_residual"], moved_combined, eps=float(args.score_epsilon)
    )

    valid_visible_mass = valid_stack.sum(0)
    raw_support = raw_stack.sum(0)
    visible_mass = visible_stack.sum(0)

    observed = valid_visible_mass > float(args.min_valid_visible_mass)
    visibility_ratio = (
        visible_mass / raw_support.clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    valid_fraction = (
        valid_visible_mass / visible_mass.clamp_min(1e-8)
    ).clamp(0.0, 1.0)

    # Evaluate the MEAN motion proposal as a scene-level sanity check.
    moved_eval = _render_loss_and_evidence(
        target_cameras,
        gaussians,
        pipe,
        background,
        mean_moved_xyz,
        rotations,
        args,
        collect_evidence=False,
    )

    global_improvement = float(static_eval["loss"] - moved_eval["loss"])
    global_rgb_improvement = float(static_eval["rgb"] - moved_eval["rgb"])

    print(
        f"[Global] static total={float(static_eval['loss']):.9f} "
        f"moved total={float(moved_eval['loss']):.9f} "
        f"improvement={global_improvement:+.9f}"
    )
    print(
        f"[Global] static RGB={float(static_eval['rgb']):.9f} "
        f"moved RGB={float(moved_eval['rgb']):.9f} "
        f"improvement={global_rgb_improvement:+.9f}"
    )

    # Optional GT.
    gt_path = None
    if args.gt_ply:
        gt_path = Path(args.gt_ply).expanduser().resolve()
    else:
        candidates = [
            Path(args.root_scene) / "pointcloud" / "points3D_parts.ply",
            Path(args.root_scene) / "points3D_parts.ply",
        ]
        gt_path = next((p for p in candidates if p.is_file()), None)

    gt_report = None
    gt_dynamic = gt_part = gt_distance = None
    if gt_path is not None and Path(gt_path).is_file():
        gt_dynamic, gt_part, gt_distance = _nearest_gt_labels(canonical, Path(gt_path))
        gt_dynamic = gt_dynamic.to(device)
        gt_part = gt_part.to(device)
        gt_distance = gt_distance.to(device)

        signals = {
            "rgb_counterfactual": rgb_counterfactual,
            "depth_counterfactual": depth_counterfactual,
            "combined_counterfactual": combined_counterfactual,
            "motion_magnitude": motion_magnitude,
            "seed_consistency": seed_consistency,
        }
        gt_report = _evaluate_signals_against_gt(
            gt_dynamic,
            gt_distance,
            observed,
            signals,
            float(args.max_gt_distance),
        )

        print("\n[GT] Individual signal quality — NO COMBINATIONS")
        for subset, metrics in gt_report.items():
            print(f"  {subset}")
            for name, m in metrics.items():
                print(
                    f"    {name:26s} "
                    f"ROC={m['roc_auc'] if m['roc_auc'] is not None else 'NA'} "
                    f"AP={m['pr_auc'] if m['pr_auc'] is not None else 'NA'} "
                    f"bestF1={m['best_f1'] if m['best_f1'] is not None else 'NA'} "
                    f"F1@0={m.get('threshold_zero', {}).get('f1', 'NA')}"
                )
    else:
        print("[GT] points3D_parts.ply not found; skipping GT metrics")

    canonical_change = float(
        (gaussians.get_canonical_xyz.detach() - canonical_guard).abs().max()
    )
    if canonical_change > 1e-10:
        raise RuntimeError(
            f"canonical Gaussian geometry changed: {canonical_change}"
        )

    result = {
        "canonical_xyz": canonical.cpu(),
        "mean_moved_xyz": mean_moved_xyz.cpu(),
        "mean_displacement": mean_displacement.cpu(),
        "motion_magnitude": motion_magnitude.cpu(),
        "trial_displacements": trial_displacements.cpu(),
        "seed_consistency": seed_consistency.cpu(),
        "static_rgb_evidence": static_e["rgb_evidence"].cpu(),
        "moved_rgb_evidence": moved_rgb.cpu(),
        "rgb_counterfactual": rgb_counterfactual.cpu(),
        "static_depth_evidence": static_e["depth_evidence"].cpu(),
        "moved_depth_evidence": moved_depth.cpu(),
        "depth_counterfactual": depth_counterfactual.cpu(),
        "static_combined_evidence": static_e["combined_residual"].cpu(),
        "moved_combined_evidence": moved_combined.cpu(),
        "combined_counterfactual": combined_counterfactual.cpu(),
        "raw_support": raw_support.cpu(),
        "visible_mass": visible_mass.cpu(),
        "valid_visible_mass": valid_visible_mass.cpu(),
        "observed": observed.cpu(),
        "visibility_ratio": visibility_ratio.cpu(),
        "valid_fraction": valid_fraction.cpu(),
        "delta_t": float(delta_t),
        "median_nn_distance": float(median_nn),
        "noise_schedule": schedule.detach().cpu(),
    }
    if gt_dynamic is not None:
        result.update(
            {
                "gt_dynamic": gt_dynamic.cpu(),
                "gt_part": gt_part.cpu(),
                "gt_distance": gt_distance.cpu(),
            }
        )
    torch.save(result, output_dir / "frame01_counterfactual.pt")

    summary = {
        "num_gaussians": int(canonical.shape[0]),
        "num_cameras": int(len(target_cameras)),
        "source_frame": int(args.source_frame),
        "target_frame": int(args.target_frame),
        "delta_t": float(delta_t),
        "median_nearest_neighbor_distance": float(median_nn),
        "noise_schedule": [float(x) for x in schedule.detach().cpu()],
        "inference_trials": int(args.inference_trials),
        "frozen_denoiser": True,
        "classification_training_losses": 0,
        "static_render_loss": float(static_eval["loss"]),
        "moved_render_loss": float(moved_eval["loss"]),
        "global_render_improvement": global_improvement,
        "static_rgb_loss": float(static_eval["rgb"]),
        "moved_rgb_loss": float(moved_eval["rgb"]),
        "global_rgb_improvement": global_rgb_improvement,
        "motion_magnitude_quantiles": _quantiles(motion_magnitude),
        "seed_consistency_quantiles": _quantiles(seed_consistency),
        "rgb_counterfactual_quantiles": _quantiles(
            rgb_counterfactual[observed]
        ),
        "depth_counterfactual_quantiles": _quantiles(
            depth_counterfactual[observed]
        ),
        "combined_counterfactual_quantiles": _quantiles(
            combined_counterfactual[observed]
        ),
        "observed_fraction": float(observed.float().mean()),
        "canonical_geometry_max_change": canonical_change,
        "gt_ply": str(gt_path) if gt_path is not None else None,
        "gt_metrics": gt_report,
    }
    (output_dir / "frame01_counterfactual_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _write_score_html(
        output_dir / "rgb_counterfactual.html",
        canonical,
        rgb_counterfactual,
        observed,
        int(args.max_html_points),
    )
    _write_camera_comparison(
        output_dir,
        target_cameras,
        gaussians,
        pipe,
        background,
        canonical,
        rotations,
        mean_moved_xyz,
    )

    if first_trajectory is not None:
        ids = (
            torch.linspace(
                0,
                canonical.shape[0] - 1,
                min(int(args.max_html_points), canonical.shape[0]),
            )
            .round()
            .long()
            .unique()
        )
        torch.save(
            {
                "indices": ids,
                "trajectory": torch.stack([x[ids] for x in first_trajectory]),
            },
            output_dir / "denoising_trajectory_subset.pt",
        )

    print("\n" + json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--root-scene", required=True)
    p.add_argument("--static-model-dir", required=True)
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--denoiser-checkpoint", required=True)

    p.add_argument("--source-frame", type=int, default=0)
    p.add_argument("--target-frame", type=int, default=1)
    p.add_argument("--fps", type=float, default=30.0)

    p.add_argument("--inference-trials", type=int, default=16)
    p.add_argument("--knn", type=int, default=16)

    # By default use the exact checkpoint schedule. These only matter with
    # --override-noise-schedule.
    p.add_argument("--override-noise-schedule", action="store_true")
    p.add_argument("--denoise-iterations", type=int, default=6)
    p.add_argument("--noise-max-ratio", type=float, default=0.15)
    p.add_argument("--noise-min-ratio", type=float, default=0.02)

    # These define only the renderer residual used by the frozen denoiser and
    # the independent evidence readouts. They are NOT classification losses.
    p.add_argument("--lambda-rgb", type=float, default=1.0)
    p.add_argument("--lambda-ssim", type=float, default=0.2)
    p.add_argument("--lambda-depth", type=float, default=0.1)
    p.add_argument("--use-depth", action=BooleanOptionalAction, default=True)

    p.add_argument("--score-epsilon", type=float, default=1e-6)
    p.add_argument("--min-valid-visible-mass", type=float, default=1e-8)

    p.add_argument("--gt-ply", default="")
    p.add_argument("--max-gt-distance", type=float, default=0.02)

    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--sh-degree", type=int, default=None)
    p.add_argument("--cameras", default="")

    bg = p.add_mutually_exclusive_group()
    bg.add_argument(
        "--white-background", dest="white_background", action="store_true"
    )
    bg.add_argument(
        "--black-background", dest="white_background", action="store_false"
    )
    p.set_defaults(white_background=None)

    p.add_argument("--max-html-points", type=int, default=5000)
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    args.inference_trials = max(1, int(args.inference_trials))
    args.knn = max(1, int(args.knn))
    args.denoise_iterations = max(1, int(args.denoise_iterations))
    args.noise_max_ratio = max(1e-8, float(args.noise_max_ratio))
    args.noise_min_ratio = max(
        1e-8, min(float(args.noise_min_ratio), args.noise_max_ratio)
    )
    args.lambda_rgb = max(0.0, float(args.lambda_rgb))
    args.lambda_ssim = max(0.0, float(args.lambda_ssim))
    args.lambda_depth = max(0.0, float(args.lambda_depth))
    args.max_html_points = max(10, int(args.max_html_points))
    return args


if __name__ == "__main__":
    run(parse_args())
