"""Counterfactual H0/Hk renders for progressive rigid Stage 1.

The important invariant is that the two hypotheses are two different physical
scenes: independent full position tensors are built before rendering.  If a
visible component moves in 3-D but RGB/depth renders are numerically identical,
the caller gets a hard failure instead of a misleading static/moving score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from utils.affected_render_support import affected_union_mask
from utils.motion_correspondence import project_world_points
from utils.rigid_endpoint_fit import transform_points


@dataclass
class CounterfactualRenderConfig:
    cameras_per_frame: int = 2
    radius_px: int = 4
    use_depth: bool = True
    numerical_epsilon: float = 1.0e-8
    min_affected_pixels: int = 1


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().reshape(-1)
    if int(values.numel()) == 0:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().cpu()),
        "median": float(values.median().cpu()),
        "max": float(values.max().cpu()),
    }


def _masked_mean_abs(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None]
    mask = mask.to(device=a.device, dtype=a.dtype)
    while mask.ndim < a.ndim:
        mask = mask.expand_as(a)
    return ((a - b).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def _save_png(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if data.ndim == 2:
        data = data[None]
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)
    if data.shape[0] > 3:
        data = data[:3]
    array = (data.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    from PIL import Image

    Image.fromarray(array).save(path)


def render_difference_stats(
    image_h0: torch.Tensor,
    image_hk: torch.Tensor,
    depth_h0: torch.Tensor | None,
    depth_hk: torch.Tensor | None,
    affected_mask: torch.Tensor,
) -> dict[str, float]:
    rgb = _masked_mean_abs(image_h0, image_hk, affected_mask)
    if depth_h0 is None or depth_hk is None:
        depth = rgb * 0.0
    else:
        if depth_h0.ndim == 2:
            depth_h0 = depth_h0[None]
        if depth_hk.ndim == 2:
            depth_hk = depth_hk[None]
        depth = _masked_mean_abs(depth_h0, depth_hk, affected_mask)
    return {
        "rgb_diff": float(rgb.detach().cpu()),
        "depth_diff": float(depth.detach().cpu()),
        "affected_pixels": int(affected_mask.sum().detach().cpu()),
        "affected_alpha_mass": float(affected_mask.float().sum().detach().cpu()),
    }


@torch.no_grad()
def score_component_counterfactual(
    *,
    state,
    inputs: dict[str, Any],
    component_mask: torch.Tensor,
    part_label: int,
    component_id: int,
    output_dir: str | Path,
    cfg: CounterfactualRenderConfig,
) -> dict[str, Any]:
    """Score H0=static and Hk=moved for one component with real renders."""
    if int(part_label) <= 0:
        raise ValueError("part_label must be a moving label")
    mask = component_mask.to(device=state.device).bool()
    if not bool(mask.any()):
        raise ValueError("empty component counterfactual")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    current_fg = state.hypothesized_positions()
    xyz_h0 = state.scatter_positions_to_full(current_fg)
    xyz_hk = xyz_h0.clone()
    full_ids = state.foreground_idx[mask].to(device=state.device).long()
    canonical_component = state.canonical_fg[mask]
    moved_component = transform_points(state.part_transforms_T[int(part_label) - 1], canonical_component)
    xyz_h0[full_ids] = canonical_component.to(xyz_h0)
    xyz_hk[full_ids] = moved_component.to(xyz_hk)

    component_delta = xyz_hk[full_ids] - xyz_h0[full_ids]
    displacement = torch.linalg.norm(component_delta, dim=-1)
    if float(displacement.max()) <= float(cfg.numerical_epsilon):
        print(
            "[COUNTERFACTUAL_GEOMETRY]\n"
            f"component id           = {int(component_id)}\n"
            f"part hypothesis        = {int(part_label)}\n"
            f"num gaussians          = {int(mask.sum())}\n"
            "3D displacement is numerically zero; H0/Hk are expected to match."
        )

    from gaussian_renderer import render

    gaussians = inputs["gaussians"]
    pipe = inputs["pipe"]
    white = bool(inputs.get("white_background", False))
    bg = torch.tensor(
        [1.0, 1.0, 1.0] if white else [0.0, 0.0, 0.0],
        device=state.device,
        dtype=state.canonical_fg.dtype,
    )
    cameras = list(inputs["camera_groups"][-1])[: max(1, int(cfg.cameras_per_frame))]

    totals = {
        "static": {"rgb": 0.0, "depth": 0.0},
        "moving": {"rgb": 0.0, "depth": 0.0},
    }
    diff_rows: list[dict[str, Any]] = []
    projected_values: list[torch.Tensor] = []
    visible_total = 0
    saved_first = False
    for camera in cameras:
        out0 = render(camera, gaussians, pipe, bg, means3D_override=xyz_h0)
        outk = render(camera, gaussians, pipe, bg, means3D_override=xyz_hk)
        img0 = out0["render"].clamp(0.0, 1.0)
        imgk = outk["render"].clamp(0.0, 1.0)
        target = camera.original_image.to(img0.device).clamp(0.0, 1.0)
        support = affected_union_mask(
            camera,
            xyz_h0,
            xyz_hk,
            full_ids,
            radius_px=int(cfg.radius_px),
            min_points=1,
        )
        diff = render_difference_stats(img0, imgk, out0.get("depth"), outk.get("depth"), support)
        diff_rows.append({"image_name": str(getattr(camera, "image_name", "")), **diff})
        totals["static"]["rgb"] += float(_masked_mean_abs(img0, target, support).detach().cpu())
        totals["moving"]["rgb"] += float(_masked_mean_abs(imgk, target, support).detach().cpu())

        depth0 = out0.get("depth")
        depthk = outk.get("depth")
        if bool(cfg.use_depth) and depth0 is not None and depthk is not None and hasattr(camera, "invdepthmap") and hasattr(camera, "depth_mask"):
            target_depth = camera.invdepthmap.to(img0.device)
            depth_mask = camera.depth_mask.to(img0.device).float()
            if target_depth.ndim == 2:
                target_depth = target_depth[None]
            if depth_mask.ndim == 2:
                depth_mask = depth_mask[None]
            local = support[None].to(depth_mask.device).float() * depth_mask
            if depth0.ndim == 2:
                depth0 = depth0[None]
            if depthk.ndim == 2:
                depthk = depthk[None]
            totals["static"]["depth"] += float(((depth0 - target_depth).abs() * local).sum().detach().cpu() / local.sum().clamp_min(1.0).detach().cpu())
            totals["moving"]["depth"] += float(((depthk - target_depth).abs() * local).sum().detach().cpu() / local.sum().clamp_min(1.0).detach().cpu())

        pix0, _d0, vis0 = project_world_points(xyz_h0[full_ids], camera)
        pixk, _dk, visk = project_world_points(xyz_hk[full_ids], camera)
        visible = vis0 & visk
        visible_total += int(visible.sum())
        if bool(visible.any()):
            projected_values.append(torch.linalg.norm(pixk[visible] - pix0[visible], dim=-1))

        if not saved_first:
            _save_png(img0, output / "H0_render.png")
            _save_png(imgk, output / "Hk_render.png")
            _save_png((img0 - imgk).abs().mean(0, keepdim=True), output / "H0_Hk_difference.png")
            _save_png(support.float(), output / "affected_mask.png")
            saved_first = True

    denom = max(1, len(cameras))
    for label in ("static", "moving"):
        totals[label]["rgb"] /= denom
        totals[label]["depth"] /= denom
    projected = torch.cat(projected_values) if projected_values else torch.empty(0, device=state.device)
    geometry = {
        "component_id": int(component_id),
        "part_hypothesis": int(part_label),
        "num_gaussians": int(mask.sum()),
        "displacement_3d": _stats(displacement),
        "projected_displacement_px": _stats(projected),
        "visible_gaussian_count": int(visible_total),
        "visible_alpha_mass": float(sum(row["affected_alpha_mass"] for row in diff_rows)),
    }
    diff_summary = {
        "affected_pixels": int(sum(row["affected_pixels"] for row in diff_rows)),
        "affected_alpha_mass": float(sum(row["affected_alpha_mass"] for row in diff_rows)),
        "rgb_diff": float(sum(row["rgb_diff"] for row in diff_rows) / denom),
        "depth_diff": float(sum(row["depth_diff"] for row in diff_rows) / denom),
        "per_camera": diff_rows,
    }
    print(
        "[COUNTERFACTUAL_GEOMETRY]\n"
        f"component id           = {geometry['component_id']}\n"
        f"part hypothesis        = {geometry['part_hypothesis']}\n"
        f"num gaussians          = {geometry['num_gaussians']}\n"
        f"3D displacement:\n"
        f"    mean               = {geometry['displacement_3d']['mean']:.6g}\n"
        f"    median             = {geometry['displacement_3d']['median']:.6g}\n"
        f"    max                = {geometry['displacement_3d']['max']:.6g}\n"
        f"projected displacement:\n"
        f"    mean px            = {geometry['projected_displacement_px']['mean']:.6g}\n"
        f"    median px          = {geometry['projected_displacement_px']['median']:.6g}\n"
        f"    max px             = {geometry['projected_displacement_px']['max']:.6g}\n"
        f"visible gaussian count = {geometry['visible_gaussian_count']}\n"
        f"visible alpha mass     = {geometry['visible_alpha_mass']:.6g}"
    )
    print(
        "[COUNTERFACTUAL_RENDER_DIFF]\n"
        f"affected pixels     = {diff_summary['affected_pixels']}\n"
        f"affected alpha mass = {diff_summary['affected_alpha_mass']:.6g}\n"
        f"H0-vs-Hk RGB diff   = {diff_summary['rgb_diff']:.6g}\n"
        f"H0-vs-Hk depth diff = {diff_summary['depth_diff']:.6g}"
    )
    if (
        geometry["displacement_3d"]["max"] > float(cfg.numerical_epsilon)
        and geometry["visible_gaussian_count"] > 0
        and diff_summary["affected_alpha_mass"] >= float(cfg.min_affected_pixels)
        and diff_summary["rgb_diff"] <= float(cfg.numerical_epsilon)
        and diff_summary["depth_diff"] <= float(cfg.numerical_epsilon)
    ):
        raise RuntimeError(
            "Visible moved component produced identical static/moving renders. "
            "Counterfactual renderer is not testing distinct scenes."
        )
    payload = {
        "geometry": geometry,
        "render_difference": diff_summary,
        "scores": totals,
        "ground_truth_used_as_algorithm_input": False,
    }
    (output / "geometry_debug.json").write_text(json.dumps(geometry, indent=2, allow_nan=False), encoding="utf-8")
    (output / "hypothesis_scores.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload
