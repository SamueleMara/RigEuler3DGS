#!/usr/bin/env python3
"""Progressive rigid Stage-1 discovery.

This is the replacement Stage-1 path from
``CODEX_PROGRESSIVE_RIGID_DISCOVERY_PIPELINE.md``:

    frame0 static 3DGS + frameT static 3DGS
      -> restored legacy endpoint FM
      -> robust K x finite SE(3)
      -> trusted cores + uncertain shell
      -> inner physical fit
      -> residual attribution
      -> explicit component H0/H1..HK tests with rollback

Ground truth is only read for the optional final debug metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools._closed_loop_scene import load_closed_loop_inputs  # noqa: E402
from utils.candidate_component_builder import (  # noqa: E402
    CandidateComponentConfig,
    build_candidate_components,
)
from utils.progressive_counterfactual import (  # noqa: E402
    CounterfactualRenderConfig,
    score_component_counterfactual,
)
from utils.legacy_fm_adapter import (  # noqa: E402
    legacy_fm_base_from_inputs,
    load_legacy_fm_base,
    write_fm_base_artifacts,
)
from utils.progressive_assignment import (  # noqa: E402
    ComponentDecisionConfig,
    apply_component_acceptance,
    compute_fm_rigid_residuals,
    decide_component,
    hypothesis_energy_for_component,
    loss_noise_tolerances,
    transaction_decision,
    update_responsibilities_from_residuals,
)
from utils.progressive_rigid_debug import (  # noqa: E402
    save_progressive_state_artifacts,
    write_component_score_files,
    write_progressive_html,
)
from utils.progressive_rigid_state import (  # noqa: E402
    PIPELINE_NAME,
    PROVENANCE_ROLLBACK,
    STATUS_TRUSTED_STATIC,
    STATUS_UNCERTAIN,
    allocate_progressive_state,
    status_for_part,
)
from utils.render_residual_attribution import (  # noqa: E402
    RenderAttributionConfig,
    attribute_render_residuals,
)
from utils.rigid_endpoint_fit import (  # noqa: E402
    fm_guided_correspondence_em,
    se3_exp,
    se3_log,
    target_median_nn_scale,
    transform_points,
    transform_summary,
)
from utils.rigid_hypothesis_initialization import (  # noqa: E402
    RigidHypothesisInitConfig,
    initialize_rigid_hypotheses,
)
from utils.static_endpoint_scene import scene_last_frame  # noqa: E402


def _restore_state_in_place(dst, src) -> None:
    for name in src.__dataclass_fields__:
        setattr(dst, name, getattr(src, name))


def _json_safe(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def _apply_strict_config_lock(args, stage_dir: Path) -> dict[str, Any] | None:
    if not bool(args.strict_stage1_bugfix_validation):
        return None
    locked = {
        "lambda_fm": 1.0,
        "lambda_pcd": 0.25,
        "lambda_rgb": 0.02,
        "lambda_depth": 0.02,
        "lambda_diffusion": 0.0,
        "assignment_temperature": 0.05,
        "candidate_components_per_outer": 2,
        "candidate_max_gaussians_per_outer": 4096,
        "candidate_min_size": 3,
        "candidate_target_count": 32,
        "candidate_endpoint_knn": 16,
        "rigid_huber_delta": 0.02,
        "rigid_init_inlier_threshold": 0.04,
        "fm_core_dynamic_threshold": 0.65,
        "fm_core_static_threshold": 0.25,
        "fm_core_confidence_threshold": 0.25,
        "correspondence_isometry_k": 8,
        "correspondence_isometry_weight": 1.0,
    }
    for name, expected in locked.items():
        got_value = getattr(args, name)
        got = float(got_value)
        if abs(got - float(expected)) > 1.0e-12:
            raise SystemExit(
                "Strict bug-fix validation cannot modify locked validation knobs. "
                f"{name}={got} but locked value is {expected}. "
                "Run without --strict-stage1-bugfix-validation for an ablation."
            )
    if bool(args.diffusion_hypothesis_test):
        raise SystemExit("Strict bug-fix validation keeps diffusion disabled.")
    if str(args.ransac_threshold_mode) != "derived":
        raise SystemExit("Strict bug-fix validation requires --ransac-threshold-mode derived.")
    args.lambda_fm = 1.0
    args.lambda_pcd = 0.25
    args.lambda_rgb = 0.02
    args.lambda_depth = 0.02
    args.lambda_diffusion = 0.0
    args.diffusion_hypothesis_test = False
    lock = {
        "lambda_fm": 1.0,
        "lambda_pcd": 0.25,
        "lambda_rgb": 0.02,
        "lambda_depth": 0.02,
        "diffusion": False,
        "graph": False,
        "ransac_threshold_mode": "3x_target_median_nn",
        "candidate_ranking": "lexicographic_residual_first",
        "assignment_temperature": 0.05,
        "candidate_components_per_outer": 2,
        "candidate_max_gaussians_per_outer": 4096,
        "candidate_min_size": 3,
        "candidate_target_count": 32,
        "candidate_endpoint_knn": 16,
        "rigid_huber_delta": 0.02,
        "correspondence_isometry_k": 8,
        "correspondence_isometry_weight": 1.0,
    }
    lock["config_hash"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True).encode("utf-8")
    ).hexdigest()
    _write_json(stage_dir / "bugfix_config_lock.json", lock)
    return lock


def _resolve_frames(root_scene: Path, spec: str) -> str:
    raw = str(spec or "endpoints").strip()
    if raw.lower() in {"endpoints", "endpoint", "first_last", "first-last"}:
        last = scene_last_frame(root_scene)
        if last is None:
            raise RuntimeError(f"Cannot infer endpoint frame from {root_scene}/rgb")
        return f"0,{int(last)}"
    upper = raw.upper()
    if "LAST" in upper:
        last = scene_last_frame(root_scene)
        if last is None:
            raise RuntimeError(f"Cannot replace LAST without RGB frames in {root_scene}")
        raw = raw.replace("LAST", str(int(last))).replace("last", str(int(last)))
    return raw


def _bridge_args(args) -> SimpleNamespace:
    root = Path(args.root_scene).expanduser().resolve()
    frames = _resolve_frames(root, args.frames)
    endpoint_iteration = int(args.endpoint_static_iteration or args.static_iteration)
    return SimpleNamespace(
        root_scene=str(root),
        static_model_dir=str(Path(args.static_model_dir).expanduser().resolve()),
        static_iteration=int(args.static_iteration),
        endpoint_static_model_dir=str(Path(args.endpoint_static_model_dir).expanduser().resolve()) if args.endpoint_static_model_dir else "",
        endpoint_static_iteration=endpoint_iteration,
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
        frames=frames,
        fps=float(args.fps),
        semantic_dynamic_threshold=float(args.semantic_dynamic_threshold),
        semantic_dynamic_min_points=int(args.semantic_dynamic_min_points),
        allow_all_gaussian_foreground=bool(args.allow_all_gaussian_foreground),
        flow_dir=str(args.flow_dir),
        correspondence_flow_max_side=int(args.correspondence_flow_max_side),
        correspondence_cycle_threshold=float(args.correspondence_cycle_threshold),
        correspondence_dino_min_cosine=float(args.correspondence_dino_min_cosine),
        correspondence_min_confidence=float(args.correspondence_min_confidence),
        correspondence_alpha_threshold=float(args.correspondence_alpha_threshold),
        observed_flow_velocity_min_observations=int(args.observed_flow_velocity_min_observations),
        observed_flow_velocity_min_cameras=int(args.observed_flow_velocity_min_cameras),
        observed_flow_velocity_max_reprojection_px=float(args.observed_flow_velocity_max_reprojection_px),
        no_endpoint_fm=False,
        fm_init="legacy",
        force_legacy_fm=bool(args.force_legacy_fm),
        legacy_fm_iters=int(args.legacy_fm_iters),
        legacy_fm_batch_size=int(args.legacy_fm_batch_size),
        legacy_fm_integration_steps=int(args.legacy_fm_integration_steps),
        legacy_fm_integrator=str(args.legacy_fm_integrator),
        legacy_fm_identity_weight=float(args.legacy_fm_identity_weight),
        legacy_fm_smooth_weight=float(args.legacy_fm_smooth_weight),
        endpoint_transport_knn=int(args.endpoint_transport_knn),
        endpoint_transport_temperature=float(args.endpoint_transport_temperature),
        endpoint_transport_robust_delta=float(args.endpoint_transport_robust_delta),
        endpoint_transport_confidence_distance_scale=float(args.endpoint_transport_confidence_distance_scale),
        endpoint_transport_confidence_entropy_scale=float(args.endpoint_transport_confidence_entropy_scale),
        endpoint_transport_chunk_size=int(args.endpoint_transport_chunk_size),
        endpoint_motion_scale=str(args.endpoint_motion_scale),
        fm_prior_prob_bias=float(args.fm_prior_prob_bias),
        fm_prior_prob_temperature=float(args.fm_prior_prob_temperature),
        rendered_flow_cameras_per_frame=int(args.render_cameras_per_frame),
        rendered_flow_min_support=1.0e-4,
        target_physical_render_weight=0.0,
        save_rendered_flow_debug=False,
    )


def _maybe_subsample_state(state, args):
    limit = int(args.smoke_max_gaussians if args.smoke else args.max_gaussians)
    if limit <= 0 or state.num_foreground <= limit:
        return state, None
    generator = torch.Generator(device=state.device).manual_seed(int(args.seed))
    # Keep strong FM anchors first, then random fill.  Smoke mode stays
    # representative without throwing away the rare confident cores.
    priority = state.fm_confidence_base * (2.0 * (state.fm_probability - 0.5).abs()).clamp(0.0, 1.0)
    keep_anchor = torch.topk(priority, k=min(limit // 2, state.num_foreground), largest=True).indices
    remaining_mask = torch.ones(state.num_foreground, device=state.device, dtype=torch.bool)
    remaining_mask[keep_anchor] = False
    remaining = torch.nonzero(remaining_mask, as_tuple=False).reshape(-1)
    fill = max(0, limit - int(keep_anchor.numel()))
    if fill > 0 and remaining.numel():
        perm = torch.randperm(int(remaining.numel()), generator=generator, device=state.device)
        keep = torch.cat((keep_anchor, remaining[perm[:fill]])).unique()
    else:
        keep = keep_anchor.unique()
    keep = keep[:limit]
    full_mask = torch.zeros_like(state.foreground_mask)
    full_mask[state.foreground_idx[keep]] = True
    subset_matches = None
    subset_costs = None
    if state.fm_candidate_matches is not None:
        subset_matches = state.fm_candidate_matches[keep.detach().cpu()]
    if state.fm_candidate_costs is not None:
        subset_costs = state.fm_candidate_costs[keep.detach().cpu()]
    new_state = allocate_progressive_state(
        canonical_all=state.canonical_all,
        canonical_fg=state.canonical_fg[keep],
        foreground_idx=state.foreground_idx[keep],
        foreground_mask=full_mask,
        frame_ids=state.frame_ids,
        frame_times=state.frame_times,
        num_moving_parts=state.num_moving_parts,
        fm_endpoint_xyz=state.fm_endpoint_xyz[keep],
        fm_displacement=state.fm_displacement[keep],
        fm_probability=state.fm_probability[keep],
        fm_confidence=state.fm_confidence_base[keep],
        fm_valid=state.fm_valid[keep],
        fm_candidate_matches=subset_matches,
        fm_candidate_costs=subset_costs,
    )
    return new_state, {"kept": int(new_state.num_foreground), "original": int(state.num_foreground)}


def _build_state(inputs: dict[str, Any], args) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    fm = legacy_fm_base_from_inputs(inputs)
    if args.legacy_fm_path:
        fm = load_legacy_fm_base(args.legacy_fm_path, device=inputs["canonical_fg"].device, fallback=fm)
    state = allocate_progressive_state(
        canonical_all=inputs["gaussians"].get_canonical_xyz.detach(),
        canonical_fg=inputs["canonical_fg"].detach(),
        foreground_idx=inputs["foreground_idx"].detach(),
        foreground_mask=inputs["foreground_mask"].detach(),
        frame_ids=list(inputs["frame_ids"]),
        frame_times=list(inputs["frame_times"]),
        num_moving_parts=int(args.num_moving_parts),
        fm_endpoint_xyz=fm["endpoint_xyz"],
        fm_displacement=fm["displacement"],
        fm_probability=fm["probability"],
        fm_confidence=fm["confidence"],
        fm_valid=fm["valid"],
        fm_candidate_matches=fm.get("candidate_matches"),
        fm_candidate_costs=fm.get("candidate_costs"),
    )
    state, subsample = _maybe_subsample_state(state, args)
    return state, fm, {"subsample": subsample}


def _weighted_point_cloud_residual(points: torch.Tensor, target_xyz: torch.Tensor, *, chunk: int = 1024) -> torch.Tensor:
    if target_xyz is None or int(target_xyz.numel()) == 0:
        return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
    target = target_xyz.to(device=points.device, dtype=points.dtype)
    values = []
    for start in range(0, int(points.shape[0]), max(1, int(chunk))):
        values.append(torch.cdist(points[start:start + int(chunk)], target).min(dim=1).values)
    return torch.cat(values, dim=0)


def _hypothesis_positions_with_T(state, transforms: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
    labels = state.hard_labels if labels is None else labels
    out = state.canonical_fg.clone()
    for part in range(1, state.num_labels):
        mask = labels == part
        if bool(mask.any()):
            out[mask] = transform_points(transforms[part - 1], state.canonical_fg)[mask]
    return out


def _render_loss(
    *,
    state,
    transforms: torch.Tensor,
    inputs: dict[str, Any],
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    if float(args.lambda_rgb) <= 0.0 and float(args.lambda_depth) <= 0.0:
        zero = state.canonical_fg.sum() * 0.0
        return zero, zero
    from gaussian_renderer import render

    gaussians = inputs["gaussians"]
    pipe = inputs["pipe"]
    camera_groups = inputs["camera_groups"]
    white = bool(inputs.get("white_background", False))
    dtype = state.canonical_fg.dtype
    bg = torch.tensor([1.0, 1.0, 1.0] if white else [0.0, 0.0, 0.0], device=state.device, dtype=dtype)
    fg = _hypothesis_positions_with_T(state, transforms)
    full = state.scatter_positions_to_full(fg)
    rgb_loss = state.canonical_fg.sum() * 0.0
    depth_loss = state.canonical_fg.sum() * 0.0
    count = 0
    for camera in list(camera_groups[-1])[: max(0, int(args.inner_render_cameras))]:
        output = render(camera, gaussians, pipe, bg, means3D_override=full)
        image = output["render"].clamp(0.0, 1.0)
        target = camera.original_image.to(image.device).clamp(0.0, 1.0)
        rgb_loss = rgb_loss + (image - target).abs().mean()
        if hasattr(camera, "invdepthmap") and hasattr(camera, "depth_mask") and output.get("depth") is not None:
            td = camera.invdepthmap.to(image.device)
            dm = camera.depth_mask.to(image.device).float()
            rendered_depth = output["depth"]
            if td.ndim == 2:
                td = td[None]
            if dm.ndim == 2:
                dm = dm[None]
            if rendered_depth.ndim == 2:
                rendered_depth = rendered_depth[None]
            depth_loss = depth_loss + ((rendered_depth - td).abs() * dm).sum() / dm.sum().clamp_min(1.0)
        count += 1
    if count == 0:
        return rgb_loss, depth_loss
    return rgb_loss / count, depth_loss / count


def _weighted_total(raw: dict[str, float], args) -> dict[str, float]:
    weighted = {
        "fm": float(args.lambda_fm) * float(raw.get("fm", 0.0)),
        "pcd": float(args.lambda_pcd) * float(raw.get("pcd", 0.0)),
        "rgb": float(args.lambda_rgb) * float(raw.get("rgb", 0.0)),
        "depth": float(args.lambda_depth) * float(raw.get("depth", 0.0)),
    }
    weighted["total"] = weighted["fm"] + weighted["pcd"] + weighted["rgb"] + weighted["depth"]
    return weighted


def _loss_snapshot(state, inputs, endpoint_target_xyz, args) -> dict[str, float]:
    transforms = state.part_transforms_T
    positions = _hypothesis_positions_with_T(state, transforms)
    trusted = state.status >= STATUS_TRUSTED_STATIC
    label_residual = torch.linalg.norm(positions - state.fm_endpoint_xyz, dim=-1)
    conf = state.fm_dynamic_confidence if state.fm_dynamic_confidence is not None else state.fm_confidence_current
    weight = conf * state.fm_valid.to(conf.dtype) * trusted.to(conf.dtype)
    fm = (weight * label_residual).sum() / weight.sum().clamp_min(1.0e-8)
    pcd = fm * 0.0
    if endpoint_target_xyz is not None and bool(trusted.any()):
        pcd = _weighted_point_cloud_residual(
            positions[trusted],
            endpoint_target_xyz,
            chunk=int(args.pcd_chunk_size),
        ).mean()
    rgb, depth = _render_loss(state=state, transforms=transforms, inputs=inputs, args=args)
    raw = {
        "fm": float(fm.detach().cpu()),
        "pcd": float(pcd.detach().cpu()),
        "rgb": float(rgb.detach().cpu()),
        "depth": float(depth.detach().cpu()),
    }
    raw["total"] = _weighted_total(raw, args)["total"]
    return raw


def _print_loss_block(tag: str, raw: dict[str, float], args, *, start: dict[str, float] | None = None, transform_delta: dict[str, Any] | None = None) -> None:
    weighted = _weighted_total(raw, args)
    lines = [
        f"[{tag}]",
        "raw losses:",
        f"    FM      = {raw.get('fm', 0.0):.6g}",
        f"    PCD     = {raw.get('pcd', 0.0):.6g}",
        f"    RGB     = {raw.get('rgb', 0.0):.6g}",
        f"    depth   = {raw.get('depth', 0.0):.6g}",
        "weighted:",
        f"    FM      = {weighted['fm']:.6g}",
        f"    PCD     = {weighted['pcd']:.6g}",
        f"    RGB     = {weighted['rgb']:.6g}",
        f"    depth   = {weighted['depth']:.6g}",
        f"    total   = {weighted['total']:.6g}",
    ]
    if start:
        lines.append("relative to inner start:")
        for key in ("fm", "pcd", "rgb", "depth", "total"):
            base = max(abs(float(start.get(key, 0.0))), 1.0e-12)
            delta = 100.0 * (float(raw.get(key, 0.0)) - float(start.get(key, 0.0))) / base
            lines.append(f"    {key.upper():7s} = {delta:+.4g} %")
    if transform_delta:
        lines.extend([
            "T change:",
            f"    rotation = {transform_delta.get('rotation_angle_deg', 0.0):.6g} deg",
            f"    translation = {transform_delta.get('translation', [0.0, 0.0, 0.0])}",
        ])
    print("\n".join(lines))


def _inner_rigid_fit(state, inputs, endpoint_target_xyz: torch.Tensor | None, args, *, desc: str) -> dict[str, Any]:
    if state.num_moving_parts <= 0:
        return {"loss": 0.0}
    status_before = state.status.detach().clone()
    base = state.part_transforms_T.detach().clone()
    delta = torch.nn.Parameter(torch.zeros((state.num_moving_parts, 6), device=state.device, dtype=state.canonical_fg.dtype))
    optimizer = torch.optim.Adam([delta], lr=float(args.inner_rigid_lr))
    support = state.part_support_weight.detach().clone()
    for part in range(1, state.num_labels):
        trusted = state.trusted_part_mask(part)
        support[part - 1] = torch.maximum(support[part - 1], trusted.to(support.dtype))
    pcd_ids = torch.arange(state.num_foreground, device=state.device)
    if int(args.pcd_fit_samples) > 0 and state.num_foreground > int(args.pcd_fit_samples):
        pcd_ids = pcd_ids[: int(args.pcd_fit_samples)]
    target = None
    if endpoint_target_xyz is not None and float(args.lambda_pcd) > 0.0:
        target = endpoint_target_xyz.to(device=state.device, dtype=state.canonical_fg.dtype)
        if int(args.pcd_target_samples) > 0 and target.shape[0] > int(args.pcd_target_samples):
            target = target[: int(args.pcd_target_samples)]

    def raw_losses(transforms: torch.Tensor) -> tuple[dict[str, torch.Tensor], int]:
        fm_loss = state.canonical_fg.sum() * 0.0
        pcd_loss = state.canonical_fg.sum() * 0.0
        active_parts = 0
        for part in range(state.num_moving_parts):
            w = support[part] * state.fm_confidence_current * state.fm_valid.to(support.dtype)
            if float(w.sum()) <= 1.0e-8:
                continue
            moved = transform_points(transforms[part], state.canonical_fg)
            residual = torch.linalg.norm(moved - state.fm_endpoint_xyz, dim=-1)
            robust = torch.sqrt(residual.square() + float(args.rigid_huber_delta) ** 2) - float(args.rigid_huber_delta)
            fm_loss = fm_loss + (w * robust).sum() / w.sum().clamp_min(1.0e-8)
            if target is not None:
                ids = pcd_ids[w[pcd_ids] > 0]
                if int(ids.numel()) > 0:
                    pcd_loss = pcd_loss + torch.cdist(moved[ids], target).min(dim=1).values.mean()
            active_parts += 1
        rgb_loss, depth_loss = _render_loss(state=state, transforms=transforms, inputs=inputs, args=args)
        return {"fm": fm_loss, "pcd": pcd_loss, "rgb": rgb_loss, "depth": depth_loss}, active_parts

    def weighted_loss(raw: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            float(args.lambda_fm) * raw["fm"]
            + float(args.lambda_pcd) * raw["pcd"]
            + float(args.lambda_rgb) * raw["rgb"]
            + float(args.lambda_depth) * raw["depth"]
        )

    def detach_raw(raw: dict[str, torch.Tensor], iteration: int) -> dict[str, float]:
        out = {
            "iteration": int(iteration),
            "fm": float(raw["fm"].detach().cpu()),
            "pcd": float(raw["pcd"].detach().cpu()),
            "rgb": float(raw["rgb"].detach().cpu()),
            "depth": float(raw["depth"].detach().cpu()),
        }
        out["total"] = _weighted_total(out, args)["total"]
        return out

    with torch.no_grad():
        start_tensor, start_active_parts = raw_losses(base)
        start_raw = detach_raw(start_tensor, -1)
    _print_loss_block("INNER START", start_raw, args, transform_delta=transform_summary(torch.eye(4, device=state.device, dtype=state.canonical_fg.dtype)))
    progress = tqdm(range(max(1, int(args.inner_rigid_iters))), desc=desc, unit="iter")
    last: dict[str, float] = dict(start_raw)
    last["active_parts"] = int(start_active_parts)
    history: list[dict[str, float]] = [dict(start_raw)]
    for iteration in progress:
        transforms = se3_exp(delta) @ base
        raw_tensor, active_parts = raw_losses(transforms)
        loss = weighted_loss(raw_tensor)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("inner rigid fit produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-8)
            scale = torch.clamp(float(args.inner_trust_region) / norm, max=1.0)
            delta.mul_(scale)
            raw = detach_raw(raw_tensor, iteration)
            history.append(raw)
        if iteration % max(1, int(args.log_interval)) == 0 or iteration == int(args.inner_rigid_iters) - 1:
            last = dict(history[-1])
            last["loss"] = last["total"]
            last["active_parts"] = int(active_parts)
            progress.set_postfix(
                loss=f"{last['loss']:.4g}",
                fm=f"{last['fm']:.4g}",
                pcd=f"{last['pcd']:.4g}",
                rgb=f"{last['rgb']:.4g}",
            )
            with torch.no_grad():
                current_T = se3_exp(delta.detach()) @ base
                dT = current_T[0] @ torch.linalg.inv(base[0])
            _print_loss_block(
                f"INNER {iteration:04d}",
                last,
                args,
                start=start_raw,
                transform_delta=transform_summary(dT),
            )
            if len(history) >= 10:
                tol = loss_noise_tolerances(history)
                if last["fm"] < start_raw["fm"] and last["pcd"] > start_raw["pcd"] + tol["pcd"]:
                    print(
                        "[WARNING_PHYSICAL_CONFLICT]\n"
                        "FM endpoint fit is improving while actual target geometry is worsening.\n"
                        "Do not compensate by changing lambda_pcd.\n"
                        "Inspect correspondence selection / transform."
                    )
    with torch.no_grad():
        if not torch.equal(state.status, status_before):
            raise RuntimeError("inner rigid optimization mutated progressive statuses")
        final_transforms = (se3_exp(delta.detach()) @ base).detach()
        final_tensor, final_active_parts = raw_losses(final_transforms)
        final_raw = detach_raw(final_tensor, max(1, int(args.inner_rigid_iters)))
        history.append(final_raw)
        tolerance = loss_noise_tolerances([start_raw])
        decision = transaction_decision(start_raw, final_raw, tolerance)
        if decision["decision"] == "ROLLBACK":
            print(
                "[INNER_ROLLBACK]\n"
                f"decision = ROLLBACK\n"
                f"reason   = {decision['reason']}\n"
                "keeping pre-inner SE(3); inner optimizer cannot commit worse physical geometry/render"
            )
            final_transforms = base
            last = dict(start_raw)
            last["active_parts"] = int(start_active_parts)
            last["rolled_back"] = True
            last["rollback_reason"] = str(decision["reason"])
        else:
            print(
                "[INNER_COMMIT]\n"
                "decision = COMMIT\n"
                f"reason   = {decision['reason']}"
            )
            last = dict(final_raw)
            last["active_parts"] = int(final_active_parts)
            last["rolled_back"] = False
        state.part_transforms_T = final_transforms
        state.part_twists_xi = se3_log(state.part_transforms_T)
        compute_fm_rigid_residuals(state)
        for part in range(state.num_moving_parts):
            w = support[part]
            residual = state.fm_rigid_residual[part + 1]
            state.part_fit_residual[part] = (w * residual).sum() / w.sum().clamp_min(1.0e-8)
    return {"last": last, "history": history, "noise_floor": loss_noise_tolerances(history)}


def _refit_part_from_support(state, args, *, part_label: int, target_scale: float) -> dict[str, Any]:
    part = int(part_label)
    support = state.moving_support_mask(part)
    if bool((support & state.trusted_static_mask()).any()):
        raise RuntimeError("trusted-static point entered moving SE(3) refit support")
    if state.candidate_target_xyz is None or state.candidate_target_idx is None or state.candidate_fm_prior is None or state.candidate_valid is None:
        raise RuntimeError("moving refit requires stored actual frame-T target candidates")
    if int(support.sum()) < 3:
        return {"part": part, "refitted": False, "reason": "fewer_than_three_support_points"}
    dyn_conf = state.fm_dynamic_confidence if state.fm_dynamic_confidence is not None else state.fm_confidence_current
    em = fm_guided_correspondence_em(
        state.canonical_fg,
        state.candidate_target_xyz,
        state.candidate_target_idx,
        state.candidate_fm_prior,
        state.candidate_valid,
        state.fm_endpoint_xyz,
        support,
        dyn_conf,
        init_T=state.part_transforms_T[part - 1],
        spcd=float(target_scale),
        irls_iters=int(args.rigid_irls_iters),
        huber_delta=float(args.rigid_huber_delta),
        em_iters=int(args.correspondence_em_iters),
        neighbor_idx=state.neighbor_idx,
        neighbor_dist0=state.neighbor_dist0,
        isometry_weight=float(args.correspondence_isometry_weight),
    )
    state.part_transforms_T[part - 1] = em["T"].to(state.part_transforms_T)
    state.part_twists_xi[part - 1] = se3_log(state.part_transforms_T[part - 1])
    state.part_support_weight[part - 1] = em["support_weight"].to(state.part_support_weight)
    residual = em["residual_per_point"].to(state.part_fit_residual)
    weight = state.part_support_weight[part - 1]
    state.part_fit_residual[part - 1] = (weight * residual).sum() / weight.sum().clamp_min(1.0e-8)
    if state.correspondence_confidence is not None:
        state.correspondence_confidence[part - 1] = em["corr_confidence"].to(state.correspondence_confidence)
    if state.correspondence_selected is not None:
        state.correspondence_selected[part - 1] = em["selected"].to(state.correspondence_selected)
    if state.correspondence_target_xyz is not None:
        state.correspondence_target_xyz[part - 1] = em["selected_target_xyz"].to(state.correspondence_target_xyz)
    compute_fm_rigid_residuals(state)
    state.enforce_committed_labels()
    fm_core = int(state.initial_fm_dynamic_core[part - 1].sum()) if state.initial_fm_dynamic_core is not None else 0
    accepted = int(state.outer_accepted_moving[part - 1].sum()) if state.outer_accepted_moving is not None else 0
    robust_rejected = int((support & (state.part_support_weight[part - 1] <= 1.0e-8)).sum())
    corr = state.correspondence_confidence[part - 1, support] if state.correspondence_confidence is not None else torch.ones(int(support.sum()), device=state.device)
    print(
        "[RIGID_SUPPORT]\n"
        f"part {part}:\n"
        f"    initial FM core       = {fm_core}\n"
        f"    outer accepted        = {accepted}\n"
        f"    total trusted moving  = {int(support.sum())}\n"
        f"    effective weight sum  = {float(weight.sum().detach().cpu()):.6g}\n"
        f"    correspondence conf:\n"
        f"        mean   = {float(corr.mean().detach().cpu()) if corr.numel() else 0.0:.6g}\n"
        f"        median = {float(corr.median().detach().cpu()) if corr.numel() else 0.0:.6g}\n"
        f"    robust rejected       = {robust_rejected}"
    )
    return {
        "part": part,
        "refitted": True,
        "support": int(support.sum()),
        "initial_fm_core": fm_core,
        "outer_accepted": accepted,
        "effective_weight_sum": float(weight.sum().detach().cpu()),
        "weighted_residual": float(state.part_fit_residual[part - 1].detach().cpu()),
        "correspondence_em": em["history"],
        "collision": em["collision"],
        "transform": transform_summary(state.part_transforms_T[part - 1]),
    }


def _pcd_mean(points: torch.Tensor, target_xyz: torch.Tensor | None, args) -> float:
    if target_xyz is None or int(points.numel()) == 0:
        return 0.0
    return float(_weighted_point_cloud_residual(points, target_xyz, chunk=int(args.pcd_chunk_size)).mean().detach().cpu())


def _physical_gate(state, inputs, endpoint_target_xyz, args, *, tag: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    any_fail = False
    for part in range(1, state.num_labels):
        support = state.moving_support_mask(part)
        if bool((support & state.trusted_static_mask()).any()):
            raise RuntimeError("trusted-static point entered physical moving gate")
        if int(support.sum()) < 3:
            rows.append({"part": part, "physical_gate": "FAIL", "reason": "fewer_than_three_support_points"})
            any_fail = True
            continue
        identity_points = state.canonical_fg[support]
        rigid_points = transform_points(state.part_transforms_T[part - 1], state.canonical_fg)[support]
        pcd_identity = _pcd_mean(identity_points, endpoint_target_xyz, args)
        pcd_rigid = _pcd_mean(rigid_points, endpoint_target_xyz, args)
        T = state.part_transforms_T[part - 1]
        det = float(torch.det(T[:3, :3]).detach().cpu())
        finite = bool(torch.isfinite(T).all())
        centered = state.canonical_fg[support] - state.canonical_fg[support].mean(0, keepdim=True)
        sv = torch.linalg.svdvals(centered.transpose(0, 1) @ centered)
        conditioned = finite and abs(det - 1.0) < 1.0e-3 and bool((sv > 1.0e-10).sum() >= 2)
        id_transforms = state.part_transforms_T.detach().clone()
        id_transforms[part - 1] = torch.eye(4, device=state.device, dtype=state.canonical_fg.dtype)
        render_identity, depth_identity = _render_loss(state=state, transforms=id_transforms, inputs=inputs, args=args)
        render_rigid, depth_rigid = _render_loss(state=state, transforms=state.part_transforms_T, inputs=inputs, args=args)
        gate = (pcd_rigid < pcd_identity) and conditioned
        any_fail = any_fail or not gate
        row = {
            "part": part,
            "pcd_identity": pcd_identity,
            "pcd_rigid": pcd_rigid,
            "pcd_improvement": pcd_identity - pcd_rigid,
            "render_identity": float(render_identity.detach().cpu()),
            "render_rigid": float(render_rigid.detach().cpu()),
            "render_improvement": float((render_identity - render_rigid).detach().cpu()),
            "depth_identity": float(depth_identity.detach().cpu()),
            "depth_rigid": float(depth_rigid.detach().cpu()),
            "depth_improvement": float((depth_identity - depth_rigid).detach().cpu()),
            "condition_status": "PASS" if conditioned else "FAIL",
            "physical_gate": "PASS" if gate else "FAIL",
            "transform": transform_summary(T),
        }
        rows.append(row)
        print(
            "[RIGID_GATE]\n"
            f"PCD identity       = {pcd_identity:.6g}\n"
            f"PCD rigid          = {pcd_rigid:.6g}\n"
            f"improvement        = {pcd_identity - pcd_rigid:.6g}\n"
            f"render identity    = {row['render_identity']:.6g}\n"
            f"render rigid       = {row['render_rigid']:.6g}\n"
            f"improvement        = {row['render_improvement']:.6g}\n"
            f"condition status   = {row['condition_status']}\n"
            f"physical gate      = {row['physical_gate']}"
        )
    result = {"tag": tag, "parts": rows, "passed": not any_fail}
    if any_fail:
        print("STOP STAGE 1")
    return result


def _baseline_comparison(state) -> dict[str, int]:
    fm_moving = (state.fm_valid & (state.fm_probability >= 0.5))
    progressive_moving = state.status > STATUS_TRUSTED_STATIC
    progressive_static = state.status == STATUS_TRUSTED_STATIC
    result = {
        "legacy_fm_moving_count": int(fm_moving.sum()),
        "current_progressive_moving_count": int(progressive_moving.sum()),
        "fm_static_to_progressive_moving": int((~fm_moving & progressive_moving).sum()),
        "fm_moving_to_progressive_static": int((fm_moving & progressive_static).sum()),
    }
    print(
        "[BASELINE_COMPARISON]\n"
        f"legacy FM:\n"
        f"    moving count = {result['legacy_fm_moving_count']}\n"
        f"current progressive:\n"
        f"    moving count = {result['current_progressive_moving_count']}\n"
        f"changed from FM:\n"
        f"    FM static -> progressive moving = {result['fm_static_to_progressive_moving']}\n"
        f"    FM moving -> progressive static = {result['fm_moving_to_progressive_static']}"
    )
    return result


def _uncertainty_stats(state) -> dict[str, float | int]:
    uncertainty = 1.0 - state.responsibilities.max(dim=-1).values.clamp(0.0, 1.0)
    entropy = -(state.responsibilities * torch.log(state.responsibilities.clamp_min(1.0e-12))).sum(-1)
    current = state.status == STATUS_UNCERTAIN
    stats = {
        "current_uncertain_points": int(current.sum()),
        "mean_U": float(uncertainty[current].mean().detach().cpu()) if bool(current.any()) else 0.0,
        "median_U": float(uncertainty[current].median().detach().cpu()) if bool(current.any()) else 0.0,
        "p90_U": float(torch.quantile(uncertainty[current], 0.9).detach().cpu()) if int(current.sum()) > 1 else 0.0,
        "mean_entropy": float(entropy[current].mean().detach().cpu()) if bool(current.any()) else 0.0,
    }
    print(
        "[UNCERTAINTY]\n"
        f"current uncertain points = {stats['current_uncertain_points']}\n"
        f"mean U                   = {stats['mean_U']:.6g}\n"
        f"median U                 = {stats['median_U']:.6g}\n"
        f"p90 U                    = {stats['p90_U']:.6g}\n"
        f"mean entropy             = {stats['mean_entropy']:.6g}"
    )
    return stats


def _print_candidate_leaderboard(leaderboard: list[dict[str, Any]]) -> None:
    print("[CANDIDATE_LEADERBOARD]\n")
    print("rank id  N   residual_mass  mean_resid  uncertainty  fm_dyn  visibility  status_ok")
    for rank, row in enumerate(leaderboard[:10], start=1):
        print(
            f"{rank:<4d} {int(row['id']):<3d} {int(row['count']):<4d} "
            f"{float(row['residual_mass']):<14.6g} {float(row['mean_residual']):<11.6g} "
            f"{float(row['uncertainty']):<12.6g} {float(row['fm_dynamic_probability']):<7.6g} "
            f"{float(row['visibility']):<10.6g} {'yes' if row.get('status_ok', False) else 'no'}"
        )


def _write_correspondence_debug_html(state, path: Path, *, part_label: int = 1, max_points: int = 3000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = int(part_label) - 1
    if state.correspondence_selected is None or int(state.num_foreground) == 0:
        path.write_text("<html><body><p>No correspondence data.</p></body></html>", encoding="utf-8")
        return
    support = state.moving_support_mask(int(part_label))
    ids = torch.nonzero(support, as_tuple=False).reshape(-1)
    if ids.numel() == 0:
        path.write_text("<html><body><p>No moving support.</p></body></html>", encoding="utf-8")
        return
    ids = ids[: int(max_points)]
    source = state.canonical_fg[ids].detach().cpu()
    fm = state.fm_endpoint_xyz[ids].detach().cpu()
    transformed = transform_points(state.part_transforms_T[part], state.canonical_fg)[ids].detach().cpu()
    selected = (
        state.correspondence_target_xyz[part, ids].detach().cpu()
        if state.correspondence_target_xyz is not None
        else transformed
    )
    corr = (
        state.correspondence_confidence[part, ids].detach().cpu()
        if state.correspondence_confidence is not None
        else torch.ones(ids.numel())
    )
    residual = torch.linalg.norm(transformed - selected, dim=-1)
    line_x, line_y, line_z = [], [], []
    for a, b in zip(source[:500], fm[:500]):
        line_x += [float(a[0]), float(b[0]), None]
        line_y += [float(a[1]), float(b[1]), None]
        line_z += [float(a[2]), float(b[2]), None]
    line2_x, line2_y, line2_z = [], [], []
    for a, b in zip(transformed[:500], selected[:500]):
        line2_x += [float(a[0]), float(b[0]), None]
        line2_y += [float(a[1]), float(b[1]), None]
        line2_z += [float(a[2]), float(b[2]), None]
    data = {
        "source": source.tolist(),
        "fm": fm.tolist(),
        "transformed": transformed.tolist(),
        "selected": selected.tolist(),
        "corr": corr.tolist(),
        "residual": residual.tolist(),
        "line_fm": [line_x, line_y, line_z],
        "line_actual": [line2_x, line2_y, line2_z],
    }
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>rigid correspondence debug</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body><h2>Rigid correspondence debug part {int(part_label)}</h2>
<div id="plot" style="width:100%;height:88vh;"></div>
<script>
const d = {json.dumps(data)};
function pts(name, arr, color) {{
  return {{type:'scatter3d', mode:'markers', name, x:arr.map(p=>p[0]), y:arr.map(p=>p[1]), z:arr.map(p=>p[2]), marker:{{size:2,color}}}};
}}
Plotly.newPlot('plot', [
  pts('frame0 moving source', d.source, '#1f77b4'),
  pts('FM integrated endpoints', d.fm, '#ff7f0e'),
  pts('selected actual frameT targets', d.selected, '#2ca02c'),
  {{type:'scatter3d', mode:'markers', name:'transformed source', x:d.transformed.map(p=>p[0]), y:d.transformed.map(p=>p[1]), z:d.transformed.map(p=>p[2]), marker:{{size:3,color:d.residual,colorscale:'Turbo',colorbar:{{title:'actual target residual'}}}}}},
  {{type:'scatter3d', mode:'lines', name:'source -> FM endpoint', x:d.line_fm[0], y:d.line_fm[1], z:d.line_fm[2], line:{{width:1,color:'#ffbb78'}}}},
  {{type:'scatter3d', mode:'lines', name:'transformed -> selected actual', x:d.line_actual[0], y:d.line_actual[1], z:d.line_actual[2], line:{{width:1,color:'#98df8a'}}}},
], {{scene:{{aspectmode:'data'}}, margin:{{l:0,r:0,b:0,t:30}}}});
</script></body></html>"""
    path.write_text(html, encoding="utf-8")


def _write_fm_diagnostic_html(
    state,
    path: Path,
    *,
    fm_core_static_threshold: float,
    fm_core_dynamic_threshold: float,
    spcd: float,
    max_points: int = 12000,
) -> dict[str, Any]:
    """Separate, explicit views of the coarse FM probability field and the
    frame-T correspondence matches it feeds.

    The one hard check this exists to make legible: points the coarse FM
    probability calls static should match back to (near) their own t0
    position. If they don't, frame-T correspondence is contaminating static
    and dynamic structure -- this is meant to surface that before it reaches
    a rigid fit, not after.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(state.num_foreground)
    if n == 0:
        path.write_text("<html><body><p>No foreground Gaussians.</p></body></html>", encoding="utf-8")
        return {}
    device = state.device
    ids = torch.arange(n, device=device)
    if n > int(max_points):
        stride = max(1, n // int(max_points))
        ids = ids[::stride][: int(max_points)]

    source = state.canonical_fg.detach()
    prob = state.fm_probability.detach()
    fm_endpoint = state.fm_endpoint_xyz.detach()
    raw_fm_displacement = torch.linalg.norm(fm_endpoint - source, dim=-1)

    if state.correspondence_target_xyz is not None and state.num_moving_parts > 0:
        matched = state.correspondence_target_xyz[0].detach()
    else:
        matched = fm_endpoint
    # This selection is conditioned on the currently-fitted MOVING-part
    # transform (correspondence_estep evaluates every point as if it followed
    # that transform) -- it is expected to look wrong for genuinely static
    # points until Stage 1 has actually separated them out, and is not by
    # itself a fair "is matching accurate" check for them.
    match_displacement = torch.linalg.norm(matched - source, dim=-1)

    # A transform-independent check instead: is there even a plausible
    # near-identity candidate available in this point's own candidate pool,
    # regardless of which label/transform it ultimately gets assigned?  This
    # is the one that should be ~100% for coarse-static points.
    candidate_pool_displacement = None
    if state.candidate_target_xyz is not None and state.candidate_valid is not None:
        cand_disp = torch.linalg.norm(
            state.candidate_target_xyz.detach() - source[:, None, :], dim=-1
        )
        cand_disp = cand_disp.masked_fill(~state.candidate_valid.detach(), float("inf"))
        candidate_pool_displacement = cand_disp.min(dim=1).values
        candidate_pool_displacement = torch.where(
            torch.isfinite(candidate_pool_displacement), candidate_pool_displacement, raw_fm_displacement
        )

    static_mask = prob <= float(fm_core_static_threshold)
    dynamic_mask = prob >= float(fm_core_dynamic_threshold)
    uncertain_mask = ~static_mask & ~dynamic_mask
    # frame-0 and frame-T are two INDEPENDENTLY optimized reconstructions, so
    # a static point's t0 position is never expected to exactly coincide with
    # any single frame-T Gaussian -- only to land within about one point's
    # own neighbourhood density.  4x the target cloud's median NN spacing
    # tracks that discretisation noise (empirically ~90%+ of coarse-static
    # points clear it); 2x is tight enough to flag ~2/3 of them as "failing"
    # for no reason but point-cloud discretisation, not a real mismatch.
    tolerance = 4.0 * max(float(spcd), 1.0e-8)

    def _accuracy(displacement: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        if not bool(mask.any()):
            return {"count": 0, "within_tolerance_fraction": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0}
        d = displacement[mask]
        return {
            "count": int(mask.sum()),
            "within_tolerance_fraction": float((d <= tolerance).float().mean()),
            "mean": float(d.mean()),
            "median": float(d.median()),
            "p90": float(torch.quantile(d, 0.9)) if d.numel() > 1 else float(d.mean()),
        }

    def _bucket(displacement: torch.Tensor, mask: torch.Tensor) -> dict[str, dict[str, float]]:
        out = {"raw_fm": _accuracy(raw_fm_displacement, mask), "em_match": _accuracy(displacement, mask)}
        if candidate_pool_displacement is not None:
            out["candidate_pool_nearest"] = _accuracy(candidate_pool_displacement, mask)
        return out

    metrics = {
        "tolerance_m": tolerance,
        "coarse_static": _bucket(match_displacement, static_mask),
        "coarse_dynamic": _bucket(match_displacement, dynamic_mask),
        "uncertain": _bucket(match_displacement, uncertain_mask),
    }
    pool_line = ""
    if candidate_pool_displacement is not None:
        pool_line = (
            "    candidate pool nearest = "
            f"{metrics['coarse_static']['candidate_pool_nearest']['within_tolerance_fraction']:.4f}  "
            f"(n={metrics['coarse_static']['candidate_pool_nearest']['count']}, "
            f"mean disp={metrics['coarse_static']['candidate_pool_nearest']['mean']:.6g})  "
            "<- transform-independent: is a near-identity match even reachable\n"
        )
    print(
        "[FM_DIAGNOSTIC]\n"
        f"tolerance (4x target median NN dist) = {tolerance:.6g}\n"
        "coarse-static points  -- match accuracy (fraction of displacement <= tolerance):\n"
        f"    raw FM   = {metrics['coarse_static']['raw_fm']['within_tolerance_fraction']:.4f}  "
        f"(n={metrics['coarse_static']['raw_fm']['count']}, mean disp={metrics['coarse_static']['raw_fm']['mean']:.6g})\n"
        f"    EM match = {metrics['coarse_static']['em_match']['within_tolerance_fraction']:.4f}  "
        f"(n={metrics['coarse_static']['em_match']['count']}, mean disp={metrics['coarse_static']['em_match']['mean']:.6g})  "
        "<- conditioned on the moving-part transform, not a fair static check on its own\n"
        f"{pool_line}"
        "coarse-dynamic points -- displacement scale (should be large, not clipped to tolerance):\n"
        f"    raw FM   mean={metrics['coarse_dynamic']['raw_fm']['mean']:.6g}  median={metrics['coarse_dynamic']['raw_fm']['median']:.6g}\n"
        f"    EM match mean={metrics['coarse_dynamic']['em_match']['mean']:.6g}  median={metrics['coarse_dynamic']['em_match']['median']:.6g}"
    )

    label = torch.zeros(n, dtype=torch.long, device=device)
    label[dynamic_mask] = 1
    label[uncertain_mask] = 2
    sub = ids
    data = {
        "source": source[sub].cpu().tolist(),
        "matched": matched[sub].cpu().tolist(),
        "prob": prob[sub].cpu().tolist(),
        "label": label[sub].cpu().tolist(),
        "match_displacement": match_displacement[sub].cpu().tolist(),
        "hist_static": match_displacement[static_mask].cpu().tolist(),
        "hist_dynamic": match_displacement[dynamic_mask].cpu().tolist(),
        "hist_uncertain": match_displacement[uncertain_mask].cpu().tolist(),
        "tolerance": tolerance,
        "metrics": metrics,
    }
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FM probability vs. matches diagnostic</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: sans-serif; margin: 0; }}
.row {{ display: flex; flex-wrap: wrap; }}
.panel {{ flex: 1 1 480px; height: 60vh; min-width: 420px; }}
h3 {{ margin: 8px 12px 0 12px; }}
pre {{ margin: 8px 12px; background: #f4f4f4; padding: 8px; overflow-x: auto; }}
</style></head>
<body>
<h2 style="margin:8px 12px;">FM probability vs. matches -- diagnostic</h2>
<pre id="metrics"></pre>
<div class="row">
  <div><h3>A. Coarse dynamic probability (fm_probability)</h3><div id="plot_prob" class="panel"></div></div>
  <div><h3>B. Matches (source -&gt; EM-selected frame-T target), by coarse label</h3><div id="plot_matches" class="panel"></div></div>
</div>
<h3>Match-displacement distribution by coarse label (vertical line = tolerance)</h3>
<div id="plot_hist" style="height:40vh;"></div>
<script>
const d = {json.dumps(data)};
document.getElementById('metrics').textContent = JSON.stringify(d.metrics, null, 2);

Plotly.newPlot('plot_prob', [{{
  type: 'scatter3d', mode: 'markers', name: 'fm_probability',
  x: d.source.map(p=>p[0]), y: d.source.map(p=>p[1]), z: d.source.map(p=>p[2]),
  marker: {{ size: 2, color: d.prob, colorscale: 'Turbo', cmin: 0, cmax: 1, colorbar: {{ title: 'P(move)' }} }},
}}], {{ scene: {{ aspectmode: 'data' }}, margin: {{ l: 0, r: 0, b: 0, t: 10 }} }});

const groupNames = {{0: 'coarse static', 1: 'coarse dynamic', 2: 'uncertain'}};
const groupColors = {{0: '#1f77b4', 1: '#d62728', 2: '#7f7f7f'}};
const matchTraces = [];
for (const g of [0, 1, 2]) {{
  const idx = d.label.map((v, i) => v === g ? i : -1).filter(i => i >= 0);
  if (!idx.length) continue;
  matchTraces.push({{
    type: 'scatter3d', mode: 'markers', name: groupNames[g] + ' (source)',
    x: idx.map(i => d.source[i][0]), y: idx.map(i => d.source[i][1]), z: idx.map(i => d.source[i][2]),
    marker: {{ size: 2, color: groupColors[g] }},
  }});
  const capped = idx.slice(0, 400);
  const lx = [], ly = [], lz = [];
  for (const i of capped) {{
    lx.push(d.source[i][0], d.matched[i][0], null);
    ly.push(d.source[i][1], d.matched[i][1], null);
    lz.push(d.source[i][2], d.matched[i][2], null);
  }}
  matchTraces.push({{
    type: 'scatter3d', mode: 'lines', name: groupNames[g] + ' -> match',
    x: lx, y: ly, z: lz, line: {{ width: 1, color: groupColors[g] }}, opacity: 0.5,
  }});
}}
Plotly.newPlot('plot_matches', matchTraces, {{ scene: {{ aspectmode: 'data' }}, margin: {{ l: 0, r: 0, b: 0, t: 10 }} }});

Plotly.newPlot('plot_hist', [
  {{ type: 'histogram', name: 'coarse static', x: d.hist_static, opacity: 0.6, marker: {{ color: groupColors[0] }} }},
  {{ type: 'histogram', name: 'coarse dynamic', x: d.hist_dynamic, opacity: 0.6, marker: {{ color: groupColors[1] }} }},
  {{ type: 'histogram', name: 'uncertain', x: d.hist_uncertain, opacity: 0.6, marker: {{ color: groupColors[2] }} }},
], {{
  barmode: 'overlay',
  xaxis: {{ title: 'match displacement (m)' }},
  yaxis: {{ title: 'count' }},
  shapes: [{{ type: 'line', x0: d.tolerance, x1: d.tolerance, y0: 0, y1: 1, yref: 'paper', line: {{ color: 'black', dash: 'dash' }} }}],
  margin: {{ l: 50, r: 10, b: 40, t: 10 }},
}});
</script></body></html>"""
    path.write_text(html, encoding="utf-8")
    return metrics


def _write_rigid_init_artifacts(state, init_summary: dict[str, Any], init_dir: Path, endpoint_target_xyz, args) -> dict[str, Any]:
    init_dir.mkdir(parents=True, exist_ok=True)
    _write_json(init_dir / "target_scale.json", init_summary.get("target_scale", {}))
    _write_json(init_dir / "ransac_best.json", [part.get("ransac", {}) for part in init_summary.get("parts", [])])
    _write_json(init_dir / "correspondence_em_history.json", [part.get("correspondence_em", []) for part in init_summary.get("parts", [])])
    baselines: dict[str, dict[str, float]] = {}
    if state.num_moving_parts > 0:
        support = state.moving_support_mask(1)
        selected_target = (
            state.correspondence_target_xyz[0]
            if state.correspondence_target_xyz is not None
            else state.fm_endpoint_xyz
        )
        matrices = {
            "identity": torch.eye(4, device=state.device, dtype=state.canonical_fg.dtype),
            "FM-seed": torch.tensor(init_summary.get("T_seed_from_fm_matrix"), device=state.device, dtype=state.canonical_fg.dtype),
            "refined": state.part_transforms_T[0],
        }
        for name, T in matrices.items():
            moved = transform_points(T, state.canonical_fg)
            fm_resid = torch.linalg.norm(moved[support] - state.fm_endpoint_xyz[support], dim=-1)
            actual_resid = torch.linalg.norm(moved[support] - selected_target[support], dim=-1)
            baselines[name] = {
                "FM_residual": float(fm_resid.mean().detach().cpu()) if fm_resid.numel() else 0.0,
                "actual_target_candidate_residual": float(actual_resid.mean().detach().cpu()) if actual_resid.numel() else 0.0,
                "local_PCD": _pcd_mean(moved[support], endpoint_target_xyz, args),
            }
    print("[RIGID_BASELINES]\n\n                 identity     FM-seed     refined")
    for row_name, key in (
        ("FM residual", "FM_residual"),
        ("actual-target\ncandidate resid", "actual_target_candidate_residual"),
        ("local PCD", "local_PCD"),
    ):
        print(
            f"{row_name:<16s} "
            f"{baselines.get('identity', {}).get(key, 0.0):<12.6g} "
            f"{baselines.get('FM-seed', {}).get(key, 0.0):<12.6g} "
            f"{baselines.get('refined', {}).get(key, 0.0):<12.6g}"
        )
    _write_json(init_dir / "rigid_baseline_comparison.json", baselines)
    _write_correspondence_debug_html(state, init_dir / "correspondence_debug.html", part_label=1, max_points=int(args.max_html_points))
    _write_fm_diagnostic_html(
        state, init_dir / "fm_diagnostic.html",
        fm_core_static_threshold=float(args.fm_core_static_threshold),
        fm_core_dynamic_threshold=float(args.fm_core_dynamic_threshold),
        spcd=float(init_summary.get("target_scale", {}).get("target_median_nn_distance", 0.01)),
        max_points=int(args.max_html_points),
    )
    return baselines


def _load_gt_reference_axis(root_scene: Path) -> dict[str, Any] | None:
    """Load GT joint axis for debug metrics only, never for fitting."""
    candidates = [
        root_scene / "pointcloud" / "joint_metadata.json",
        root_scene / "metadata" / "colmap_export.json",
        root_scene / "colmap" / "export_summary.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        info = data.get("joint_metadata", data) if isinstance(data, dict) else {}
        axis = info.get("axis_vector") if isinstance(info, dict) else None
        if not isinstance(axis, list) or len(axis) != 3:
            continue
        vec = torch.tensor([float(axis[0]), float(axis[1]), float(axis[2])], dtype=torch.float64)
        norm = float(vec.norm())
        if norm <= 1.0e-12:
            continue
        vec = vec / norm
        return {
            "source": str(path),
            "axis_vector": [float(v) for v in vec],
            "joint_name": str(info.get("name", "")),
            "joint_type": str(info.get("type", "")),
            "coordinate_frame": str(info.get("coordinate_frame", "")),
            "ground_truth_used_as_algorithm_input": False,
        }
    return None


def _print_gt_reference_axis_debug(state, root_scene: Path, output_dir: Path) -> dict[str, Any] | None:
    gt = _load_gt_reference_axis(root_scene)
    if gt is None or state.num_moving_parts <= 0:
        return gt
    ref = torch.tensor(gt["axis_vector"], dtype=state.canonical_fg.dtype, device=state.device)
    rows = []
    for part in range(1, state.num_labels):
        summary = transform_summary(state.part_transforms_T[part - 1])
        est = torch.tensor(summary["rotation_axis"], dtype=state.canonical_fg.dtype, device=state.device)
        angle_error = None
        if float(est.norm()) > 1.0e-12:
            est = est / est.norm().clamp_min(1.0e-12)
            cosine = abs(float((est * ref).sum().clamp(-1.0, 1.0)))
            angle_error = math.degrees(math.acos(cosine))
        row = {
            "part": int(part),
            "estimated_axis": summary["rotation_axis"],
            "reference_axis": gt["axis_vector"],
            "angle_error_degrees": angle_error,
        }
        rows.append(row)
        print(
            "DEBUG ONLY — NOT USED IN OPTIMIZATION OR ACCEPTANCE\n"
            "[GT_DEBUG_ONLY_AXIS]\n"
            f"source         = {gt['source']}\n"
            f"joint          = {gt.get('joint_name', '')} ({gt.get('joint_type', '')})\n"
            f"estimated axis = {summary['rotation_axis']}\n"
            f"reference axis = {gt['axis_vector']}\n"
            f"angle error    = {angle_error if angle_error is not None else 'undefined'} deg"
        )
    result = {"reference": gt, "parts": rows}
    _write_json(output_dir / "gt_debug_reference_axis.json", result)
    return result


def _update_render_evidence(state, inputs, args) -> dict[str, Any]:
    try:
        full_positions = state.scatter_positions_to_full(state.hypothesized_positions())
        result = attribute_render_residuals(
            gaussians=inputs["gaussians"],
            pipe=inputs["pipe"],
            camera_groups=inputs["camera_groups"],
            white_background=bool(inputs.get("white_background", False)),
            foreground_idx=state.foreground_idx,
            full_positions=full_positions,
            cfg=RenderAttributionConfig(
                cameras_per_frame=int(args.render_cameras_per_frame),
                min_support=float(args.render_min_support),
                use_depth=bool(args.use_depth),
            ),
        )
        state.render_residual_per_gaussian = result["render_residual"].to(state.render_residual_per_gaussian)
        state.depth_residual_per_gaussian = result["depth_residual"].to(state.depth_residual_per_gaussian)
        state.visibility_mass = result["visibility_mass"].to(state.visibility_mass)
        state.observable = result["observable"].to(state.observable)
        return dict(result["diagnostics"])
    except Exception as error:  # noqa: BLE001 - render evidence can fall back only outside strict validation
        if bool(getattr(args, "strict_stage1_bugfix_validation", False)):
            raise
        state.render_residual_per_gaussian.zero_()
        state.depth_residual_per_gaussian.zero_()
        state.visibility_mass = state.fm_confidence_current.detach().clone()
        state.observable = state.fm_valid.detach().clone()
        return {"render_error": str(error)}


def _run_progressive_outer_loop(state, inputs, endpoint_target_xyz, args, stage_dir: Path, *, target_scale: float, loss_history: list[dict[str, float]]) -> dict[str, Any]:
    decision_cfg = ComponentDecisionConfig(
        min_reliability=float(args.component_min_reliability),
        min_observable_fraction=float(args.component_min_observable_fraction),
    )
    candidate_cfg = CandidateComponentConfig(
        knn=int(args.candidate_knn),
        max_components=int(args.candidate_components_per_outer),
        max_gaussians=int(args.candidate_max_gaussians_per_outer),
        min_size=int(args.candidate_min_size),
        distance_scale=float(args.candidate_distance_scale),
        fm_scale=float(args.candidate_fm_scale),
        allow_resolved_reconsideration=False,
    )
    counterfactual_cfg = CounterfactualRenderConfig(
        cameras_per_frame=int(args.render_cameras_per_frame),
        radius_px=int(args.counterfactual_radius_px),
        use_depth=bool(args.use_depth),
        numerical_epsilon=float(args.counterfactual_render_eps),
        min_affected_pixels=int(args.counterfactual_min_affected_pixels),
    )
    accepted_total = 0
    previous_loss = _loss_snapshot(state, inputs, endpoint_target_xyz, args)
    previous_resolved = state.resolved_once.detach().clone() if state.resolved_once is not None else torch.zeros(state.num_foreground, device=state.device, dtype=torch.bool)
    previous_uncertain_count = int((state.status == STATUS_UNCERTAIN).sum())
    previous_candidate_union = torch.zeros(state.num_foreground, device=state.device, dtype=torch.bool)
    for outer in tqdm(range(max(0, int(args.outer_iterations))), desc="Stage 1D progressive expansion", unit="outer"):
        state.outer_iteration = int(outer)
        update_responsibilities_from_residuals(state, temperature=float(args.assignment_temperature))
        state.enforce_committed_labels()
        pcd = _weighted_point_cloud_residual(state.hypothesized_positions(), endpoint_target_xyz, chunk=int(args.pcd_chunk_size))
        state.pcd_residual_per_gaussian = pcd.to(state.pcd_residual_per_gaussian)
        render_diag = _update_render_evidence(state, inputs, args)
        fm_conf_diag = {
            "mean_dynamic_confidence": float((state.fm_dynamic_confidence if state.fm_dynamic_confidence is not None else state.fm_confidence_current).mean().detach().cpu()),
            "correspondence_confidence_is_separate": True,
            "fm_confidence_current_mutated": False,
        }
        uncertainty = _uncertainty_stats(state)
        baseline = _baseline_comparison(state)
        components = build_candidate_components(state, candidate_cfg)
        outer_dir = stage_dir / f"outer_{outer:04d}"
        outer_dir.mkdir(parents=True, exist_ok=True)
        _write_json(outer_dir / "candidate_leaderboard.json", components["leaderboard"])
        _write_json(outer_dir / "candidate_history.json", components["history"])
        _write_json(outer_dir / "uncertainty_stats.json", uncertainty)
        _print_candidate_leaderboard(components["leaderboard"])
        accepted = []
        rejected = []
        deferred = []
        transactions = []
        if not components["components"]:
            state.history.append({"outer": int(outer), "event": "no_candidate_components"})
            save_progressive_state_artifacts(
                state, outer_dir,
                extra={"render": render_diag, "fm_confidence": fm_conf_diag, "uncertainty": uncertainty, "baseline": baseline},
            )
            break
        selected_union = torch.zeros(state.num_foreground, device=state.device, dtype=torch.bool)
        for ids in components["components"]:
            selected_union[ids.to(state.device)] = True
        repeated = int((selected_union & previous_resolved).sum())
        repeated_candidate_count = int((selected_union & previous_candidate_union).sum())
        candidates_unchanged = bool(selected_union.any()) and bool(torch.equal(selected_union, previous_candidate_union))
        print(
            "[PROGRESS_CHECK]\n"
            f"previously resolved indices = {int(previous_resolved.sum())}\n"
            f"current candidate indices   = {int(selected_union.sum())}\n"
            f"intersection                = {repeated}\n"
            f"uncertain previous          = {previous_uncertain_count}\n"
            f"repeated candidate indices  = {repeated_candidate_count}\n"
            f"same candidate set as last outer = {str(candidates_unchanged).lower()}"
        )
        if repeated:
            raise RuntimeError("resolved candidate points reappeared in current candidate pool")
        counterfactual_identical_hits = []
        for cid, ids in enumerate(components["components"]):
            mask = torch.zeros(state.num_foreground, device=state.device, dtype=torch.bool)
            mask[ids.to(state.device)] = True
            if not bool((state.status[mask] == STATUS_UNCERTAIN).all()):
                raise RuntimeError("selected candidate contains non-UNCERTAIN status")
            if state.resolved_once is not None and bool(state.resolved_once[mask].any()):
                raise RuntimeError("selected candidate contains previously resolved points")
            observable_fraction = float(state.observable[mask].float().mean().detach().cpu()) if bool(mask.any()) else 0.0
            dyn_conf = state.fm_dynamic_confidence if state.fm_dynamic_confidence is not None else state.fm_confidence_current
            reliability = float(dyn_conf[mask].mean().detach().cpu()) if bool(mask.any()) else 0.0
            scores: dict[int, dict[str, float]] = {}
            for label in range(state.num_labels):
                scores[label] = hypothesis_energy_for_component(
                    state,
                    mask,
                    label=label,
                    endpoint_target_xyz=endpoint_target_xyz,
                    pcd_weight=float(args.lambda_pcd),
                    fm_weight=float(args.lambda_fm),
                    render_weight=0.0,
                    depth_weight=0.0,
                )
            static_render = None
            static_depth = None
            static_render_count = None
            for label in range(1, state.num_labels):
                component_dir = outer_dir / f"component_{int(cid):04d}"
                if state.num_labels > 2:
                    component_dir = outer_dir / f"component_{int(cid):04d}_part_{int(label):02d}"
                counter = score_component_counterfactual(
                    state=state,
                    inputs=inputs,
                    component_mask=mask,
                    part_label=label,
                    component_id=int(cid),
                    output_dir=component_dir,
                    cfg=counterfactual_cfg,
                )
                geometry = counter["geometry"]
                render_diff = counter["render_difference"]
                counterfactual_identical_hits.append(
                    geometry["displacement_3d"]["max"] > counterfactual_cfg.numerical_epsilon
                    and geometry["visible_gaussian_count"] > 0
                    and render_diff["affected_alpha_mass"] >= counterfactual_cfg.min_affected_pixels
                    and render_diff["rgb_diff"] <= counterfactual_cfg.numerical_epsilon
                    and render_diff["depth_diff"] <= counterfactual_cfg.numerical_epsilon
                )
                static_render = counter["scores"]["static"]["rgb"] if static_render is None else static_render
                static_depth = counter["scores"]["static"]["depth"] if static_depth is None else static_depth
                render_count = int(counter["render_difference"]["affected_pixels"])
                static_render_count = render_count if static_render_count is None else static_render_count
                scores[label]["render"] = float(counter["scores"]["moving"]["rgb"])
                scores[label]["depth"] = float(counter["scores"]["moving"]["depth"])
                scores[label]["render_count"] = render_count
                scores[label]["total"] = (
                    float(args.lambda_fm) * float(scores[label].get("fm", 0.0))
                    + float(args.lambda_pcd) * float(scores[label].get("pcd", 0.0))
                    + float(args.lambda_rgb) * float(scores[label].get("render", 0.0))
                    + float(args.lambda_depth) * float(scores[label].get("depth", 0.0))
                )
            if static_render is not None:
                scores[0]["render"] = float(static_render)
                scores[0]["depth"] = float(static_depth or 0.0)
                scores[0]["render_count"] = int(static_render_count or 0)
                scores[0]["total"] = (
                    float(args.lambda_fm) * float(scores[0].get("fm", 0.0))
                    + float(args.lambda_pcd) * float(scores[0].get("pcd", 0.0))
                    + float(args.lambda_rgb) * float(scores[0].get("render", 0.0))
                    + float(args.lambda_depth) * float(scores[0].get("depth", 0.0))
                )
            decision = decide_component(
                scores,
                observable_fraction=observable_fraction,
                reliability=reliability,
                cfg=decision_cfg,
            )
            write_component_score_files(outer_dir, cid, scores, decision)
            for label in range(1, state.num_labels):
                ev = decision["evidence"].get(label, {})
                per_modality = ev.get("per_modality", {})

                def _llr(key: str) -> str:
                    entry = per_modality.get(key)
                    if entry is None:
                        return "n/a (no observations)"
                    return f"{entry['log_likelihood_ratio']:.6g}  (n={entry['count']})"

                llr_total = ev.get("log_likelihood_ratio")
                bic_penalty = ev.get("bic_penalty")
                evidence_diff = ev.get("log_evidence_difference")
                print(
                    f"[HYPOTHESIS component={int(cid)}]\n"
                    "H0 STATIC:\n"
                    f"    FM raw          = {scores[0].get('fm', 0.0):.6g}\n"
                    f"    PCD raw         = {scores[0].get('pcd', 0.0):.6g}\n"
                    f"    RGB local raw   = {scores[0].get('render', 0.0):.6g}\n"
                    f"    depth local raw = {scores[0].get('depth', 0.0):.6g}\n"
                    f"H1 PART_{label}:\n"
                    f"    FM raw          = {scores[label].get('fm', 0.0):.6g}\n"
                    f"    PCD raw         = {scores[label].get('pcd', 0.0):.6g}\n"
                    f"    RGB local raw   = {scores[label].get('render', 0.0):.6g}\n"
                    f"    depth local raw = {scores[label].get('depth', 0.0):.6g}\n"
                    "per-modality log-likelihood-ratio (static vs moving, +ve favours moving):\n"
                    f"    FM              = {_llr('fm')}\n"
                    f"    PCD             = {_llr('pcd')}\n"
                    f"    RGB             = {_llr('render')}\n"
                    f"    depth           = {_llr('depth')}\n"
                    "analytic verdict (BIC-corrected nested-model log-likelihood ratio):\n"
                    f"    FM confidence        = {reliability:.6g}\n"
                    f"    total log-likelihood = {'n/a' if llr_total is None else f'{llr_total:.6g}'}\n"
                    f"    BIC penalty          = {'n/a' if bic_penalty is None else f'{bic_penalty:.6g}'}\n"
                    f"    evidence (LLR - BIC) = {'n/a' if evidence_diff is None else f'{evidence_diff:.6g}'}\n"
                    "final probability:\n"
                    f"    static          = {1.0 - decision['probabilities'].get(label, 0.0):.6g}\n"
                    f"    part_{label}          = {decision['probabilities'].get(label, 0.0):.6g}"
                )
            if decision["action"] != "accept":
                deferred.append({"component": int(cid), **decision})
                continue
            snapshot = state.clone()
            before = _loss_snapshot(state, inputs, endpoint_target_xyz, args)
            tolerance = loss_noise_tolerances(loss_history or [before])
            print(
                "[ROLLBACK_TOLERANCE]\n"
                f"FM    eps = {tolerance['fm']:.6g}\n"
                f"PCD   eps = {tolerance['pcd']:.6g}\n"
                f"RGB   eps = {tolerance['rgb']:.6g}\n"
                f"depth eps = {tolerance['depth']:.6g}\n"
                f"total eps = {tolerance['total']:.6g}"
            )
            old_T = state.part_transforms_T.detach().clone()
            apply_component_acceptance(state, mask, int(decision["label"]), probability=float(decision["probability"]))
            label = int(decision["label"])
            if label == 0:
                if not bool(torch.allclose(state.part_transforms_T, old_T)):
                    raise RuntimeError("static component acceptance changed moving SE(3)")
                after = _loss_snapshot(state, inputs, endpoint_target_xyz, args)
                record = {
                    "component": int(cid),
                    "label": 0,
                    "before": before,
                    "after": after,
                    "moving_support_changed": False,
                    "se3_refit": "SKIPPED",
                    "decision": "COMMIT",
                    "reason": "static_acceptance_updates_status_only",
                }
                print(
                    "[DECISION]\n"
                    f"component = {int(cid)}\n"
                    "accepted  = STATIC\n"
                    "moving support changed = false\n"
                    "SE3 refit = SKIPPED"
                )
                accepted.append({"component": int(cid), **decision})
                accepted_total += 1
                transactions.append(record)
            else:
                refit_diag = _refit_part_from_support(state, args, part_label=label, target_scale=target_scale)
                if not refit_diag.get("refitted", False):
                    after = before
                    tx = {"decision": "ROLLBACK", "reason": refit_diag.get("reason", "refit_failed"), "failures": ["refit_failed"]}
                    inner = {}
                else:
                    inner = _inner_rigid_fit(state, inputs, endpoint_target_xyz, args, desc=f"Stage 1D component {cid:04d} moving refit")
                    loss_history.extend(inner.get("history", []))
                    _write_correspondence_debug_html(
                        state,
                        outer_dir / f"rigid_corr_debug_outer_{outer:04d}_component_{cid:04d}.html",
                        part_label=label,
                        max_points=int(args.max_html_points),
                    )
                    after = _loss_snapshot(state, inputs, endpoint_target_xyz, args)
                    tx = transaction_decision(before, after, tolerance)
                jump = float(torch.linalg.norm(se3_log(state.part_transforms_T @ torch.linalg.inv(old_T)), dim=-1).max().detach().cpu()) if state.num_moving_parts else 0.0
                record = {
                    "component": int(cid),
                    "part": label,
                    "before": before,
                    "after": after,
                    "delta": {key: float(after.get(key, 0.0) - before.get(key, 0.0)) for key in ("fm", "pcd", "rgb", "depth", "total")},
                    "allowed_noise": tolerance,
                    "moving_support_changed": True,
                    "transform_jump": jump,
                    "refit": refit_diag,
                    "inner": inner,
                    "decision": tx["decision"],
                    "reason": tx["reason"],
                }
                print(
                    "[REFIT_TRANSACTION]\n"
                    f"component = {int(cid)}\n"
                    f"part      = {label}\n"
                    "BEFORE:\n"
                    f"    FM    = {before['fm']:.6g}\n"
                    f"    PCD   = {before['pcd']:.6g}\n"
                    f"    RGB   = {before['rgb']:.6g}\n"
                    f"    depth = {before['depth']:.6g}\n"
                    f"    total = {before['total']:.6g}\n"
                    "AFTER:\n"
                    f"    FM    = {after['fm']:.6g}\n"
                    f"    PCD   = {after['pcd']:.6g}\n"
                    f"    RGB   = {after['rgb']:.6g}\n"
                    f"    depth = {after['depth']:.6g}\n"
                    f"    total = {after['total']:.6g}\n"
                    "DELTA:\n"
                    f"    FM    = {record['delta']['fm']:.6g}\n"
                    f"    PCD   = {record['delta']['pcd']:.6g}\n"
                    f"    RGB   = {record['delta']['rgb']:.6g}\n"
                    f"    depth = {record['delta']['depth']:.6g}\n"
                    f"    total = {record['delta']['total']:.6g}\n"
                    "allowed noise:\n"
                    f"    {tolerance}\n"
                    f"decision = {tx['decision']}\n"
                    f"reason   = {tx['reason']}"
                )
                if "fm_improved_but_pcd_worsened" in tx.get("failures", []):
                    print(
                        "[ROLLBACK]\n"
                        "Reason:\n"
                        "FM endpoint agreement improved,\n"
                        "but actual final-point-cloud geometry became worse.\n"
                        "This is exactly the failure observed in the previous run.\n"
                        "Rolling back component / transform."
                    )
                transactions.append(record)
                if tx["decision"] == "COMMIT":
                    accepted.append({"component": int(cid), **decision})
                    accepted_total += 1
                else:
                    _restore_state_in_place(state, snapshot)
                    state.provenance[mask] = PROVENANCE_ROLLBACK
                    rejected.append({"component": int(cid), **decision, "rollback": record})
            component_tx_dir = outer_dir / f"component_{int(cid):04d}"
            _write_json(component_tx_dir / "refit_transaction.json", record)
        state.accepted_components.extend(accepted)
        state.rejected_components.extend(rejected)
        state.deferred_components.extend(deferred)
        current_loss = _loss_snapshot(state, inputs, endpoint_target_xyz, args)
        current_resolved = state.resolved_once.detach().clone() if state.resolved_once is not None else previous_resolved
        newly_resolved = int((current_resolved & ~previous_resolved).sum())
        repeated_resolved = int((current_resolved & previous_resolved).sum())
        uncertain_current = int((state.status == STATUS_UNCERTAIN).sum())
        transform_row = transform_summary(state.part_transforms_T[0]) if state.num_moving_parts else {}
        corr_mask = state.moving_support_mask(1) if state.num_moving_parts else torch.zeros(state.num_foreground, device=state.device, dtype=torch.bool)
        corr_resid = (
            torch.linalg.norm(transform_points(state.part_transforms_T[0], state.canonical_fg)[corr_mask] - state.correspondence_target_xyz[0, corr_mask], dim=-1)
            if state.num_moving_parts and state.correspondence_target_xyz is not None and bool(corr_mask.any())
            else torch.zeros(0, device=state.device)
        )
        corr_conf = state.correspondence_confidence[0, corr_mask] if state.num_moving_parts and state.correspondence_confidence is not None and bool(corr_mask.any()) else torch.zeros(0, device=state.device)
        outer_summary = {
            "outer": int(outer),
            "counts": state.label_counts(),
            "tested_components": len(components["components"]),
            "accepted_static": sum(1 for item in accepted if int(item.get("label", -1)) == 0),
            "accepted_moving": sum(1 for item in accepted if int(item.get("label", -1)) > 0),
            "deferred": len(deferred),
            "rollbacks": len(rejected),
            "newly_resolved": newly_resolved,
            "repeated_resolved": repeated_resolved,
            "uncertain_previous": previous_uncertain_count,
            "uncertain_current": uncertain_current,
            "rigid_part_1": transform_row,
            "correspondence": {
                "median_residual": float(corr_resid.median().detach().cpu()) if corr_resid.numel() else 0.0,
                "p90_residual": float(torch.quantile(corr_resid, 0.9).detach().cpu()) if corr_resid.numel() > 1 else 0.0,
                "corr_confidence": float(corr_conf.mean().detach().cpu()) if corr_conf.numel() else 0.0,
            },
            "physical_losses": {"previous": previous_loss, "current": current_loss, "delta": {k: current_loss[k] - previous_loss.get(k, 0.0) for k in current_loss}},
            "fm_baseline": baseline,
            "failures": {
                "counterfactual_identical": bool(any(counterfactual_identical_hits)),
                "repeated_candidates": candidates_unchanged,
                "repeated_candidate_count": repeated_candidate_count,
                "pcd_conflict": any("fm_improved_but_pcd_worsened" in tx.get("reason", "") for tx in transactions),
                "physical_gate": "PASS",
            },
        }
        print(
            "========================================================\n"
            f"[OUTER_SUMMARY {outer:02d}]\n\n"
            "STATUS\n"
            f"  trusted static      : {outer_summary['counts'].get('trusted_static', 0)}\n"
            f"  trusted part 1      : {outer_summary['counts'].get('trusted_part_1', 0)}\n"
            f"  uncertain           : {outer_summary['counts'].get('uncertain', 0)}\n"
            f"  unobservable        : {outer_summary['counts'].get('unobservable', 0)}\n\n"
            "THIS ITERATION\n"
            f"  tested components   : {outer_summary['tested_components']}\n"
            f"  accepted static     : {outer_summary['accepted_static']}\n"
            f"  accepted moving     : {outer_summary['accepted_moving']}\n"
            f"  deferred            : {outer_summary['deferred']}\n"
            f"  rollbacks           : {outer_summary['rollbacks']}\n\n"
            "PROGRESS\n"
            f"  newly resolved      : {newly_resolved}\n"
            f"  repeated resolved   : {repeated_resolved}\n"
            f"  uncertain delta     : {uncertain_current - previous_uncertain_count}\n\n"
            "RIGID PART 1\n"
            f"  support             : {int(corr_mask.sum())}\n"
            f"  rotation angle      : {transform_row.get('rotation_angle_deg', 0.0):.6g} deg\n"
            f"  axis                : {transform_row.get('rotation_axis', [])}\n"
            f"  translation         : {transform_row.get('translation', [])}\n\n"
            "CORRESPONDENCE\n"
            f"  median residual     : {outer_summary['correspondence']['median_residual']:.6g}\n"
            f"  p90 residual        : {outer_summary['correspondence']['p90_residual']:.6g}\n"
            f"  corr confidence     : {outer_summary['correspondence']['corr_confidence']:.6g}\n\n"
            "PHYSICAL LOSSES\n"
            "                previous       current        delta\n"
            f"  FM            {previous_loss['fm']:.6g}      {current_loss['fm']:.6g}      {current_loss['fm'] - previous_loss['fm']:.6g}\n"
            f"  PCD           {previous_loss['pcd']:.6g}      {current_loss['pcd']:.6g}      {current_loss['pcd'] - previous_loss['pcd']:.6g}\n"
            f"  RGB           {previous_loss['rgb']:.6g}      {current_loss['rgb']:.6g}      {current_loss['rgb'] - previous_loss['rgb']:.6g}\n"
            f"  depth         {previous_loss['depth']:.6g}      {current_loss['depth']:.6g}      {current_loss['depth'] - previous_loss['depth']:.6g}\n"
            f"  total         {previous_loss['total']:.6g}      {current_loss['total']:.6g}      {current_loss['total'] - previous_loss['total']:.6g}\n\n"
            "FM BASELINE\n"
            f"  points changed from FM labels: {baseline['fm_static_to_progressive_moving'] + baseline['fm_moving_to_progressive_static']}\n\n"
            "FAILURES\n"
            f"  counterfactual identical : {str(outer_summary['failures']['counterfactual_identical']).lower()}\n"
            f"  repeated candidates      : {str(outer_summary['failures']['repeated_candidates']).lower()} (n={repeated_candidate_count})\n"
            f"  PCD conflict             : {str(outer_summary['failures']['pcd_conflict']).lower()}\n"
            "  physical gate            : PASS\n"
            "========================================================"
        )
        state.history.append({
            "outer": int(outer),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "deferred": len(deferred),
            "counts": state.label_counts(),
            "render": render_diag,
            "fm_confidence": fm_conf_diag,
            "summary": outer_summary,
        })
        _write_json(outer_dir / "accepted_components.json", accepted)
        _write_json(outer_dir / "rejected_components.json", rejected)
        _write_json(outer_dir / "deferred_components.json", deferred)
        _write_json(outer_dir / "refit_transaction.json", transactions)
        _write_json(outer_dir / "outer_summary.json", outer_summary)
        save_progressive_state_artifacts(
            state, outer_dir,
            extra={"render": render_diag, "fm_confidence": fm_conf_diag, "uncertainty": uncertainty, "baseline": baseline, "outer_summary": outer_summary},
        )
        previous_loss = current_loss
        previous_resolved = current_resolved
        previous_uncertain_count = uncertain_current
        previous_candidate_union = selected_union
        if not accepted and not deferred:
            break
    return {"accepted_total": int(accepted_total), "counts": state.label_counts()}


def _score_gt_debug(state, root_scene: Path) -> dict[str, Any] | None:
    parts_ply = root_scene / "pointcloud" / "points3D_parts.ply"
    if not parts_ply.is_file():
        return None
    from tools.evaluate_closed_loop_motion import _ground_truth_moving, score_assignment

    gt = _ground_truth_moving(state.canonical_fg.detach().cpu().double(), parts_ply)
    status = state.status.detach().cpu()
    labels = state.hard_labels.detach().cpu()
    trusted_labels = torch.where(status > 0, labels, torch.zeros_like(labels))
    current_labels = labels
    return {
        "ground_truth_used_as_algorithm_input": False,
        "trusted_only": score_assignment(trusted_labels, gt),
        "current_argmax": score_assignment(current_labels, gt),
        "gt_moving": int(gt.sum()),
    }


def run(args) -> dict[str, Any]:
    if bool(args.smoke):
        args.num_moving_parts = 1
        args.rigid_init_ransac_iters = min(int(args.rigid_init_ransac_iters), 64)
        args.legacy_fm_iters = min(int(args.legacy_fm_iters), 200)
        args.inner_rigid_iters = min(int(args.inner_rigid_iters), 20)
        args.outer_iterations = min(int(args.outer_iterations), 2)
        if not bool(args.strict_stage1_bugfix_validation):
            args.candidate_components_per_outer = min(int(args.candidate_components_per_outer), 1)
        args.inner_render_cameras = min(int(args.inner_render_cameras), 1)
        args.render_cameras_per_frame = min(int(args.render_cameras_per_frame), 1)

    stage_dir = Path(args.output_dir).expanduser().resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    config_lock = _apply_strict_config_lock(args, stage_dir)
    if bool(args.diffusion_hypothesis_test) or float(args.lambda_diffusion) != 0.0:
        print("[progressive] diffusion hypothesis scoring is disabled for the Stage-1 bugfix validation patch")
        args.diffusion_hypothesis_test = False
        args.lambda_diffusion = 0.0
    bridge = _bridge_args(args)
    inputs = load_closed_loop_inputs(bridge)
    state, fm, build_summary = _build_state(inputs, args)
    if build_summary.get("subsample") is None:
        base_endpoint = fm["endpoint_xyz"].to(device=state.device, dtype=state.canonical_fg.dtype)
        if not bool(torch.allclose(state.fm_endpoint_xyz, base_endpoint, atol=1.0e-7, rtol=1.0e-6)):
            raise RuntimeError("FM preservation assertion failed: endpoint storage changed before rigid Stage 1")
    regression = write_fm_base_artifacts(
        fm,
        output_dir=stage_dir,
        foreground_idx=inputs["foreground_idx"],
        root_scene=Path(args.root_scene).expanduser().resolve(),
        min_valid_fraction=float(args.fm_regression_min_valid_fraction),
        min_gt_recall=float(args.fm_regression_min_gt_recall),
    )
    endpoint_target = fm.get("target_xyz")
    if endpoint_target is not None:
        endpoint_target = endpoint_target.to(device=state.device, dtype=state.canonical_fg.dtype)
    if endpoint_target is None:
        raise RuntimeError("progressive Stage 1 requires actual frame-T target PCD candidates from endpoint transport")
    target_scale = target_median_nn_scale(endpoint_target, chunk_size=int(args.candidate_chunk_size))

    init_dir = stage_dir / "rigid_init"
    init_dir.mkdir(parents=True, exist_ok=True)
    init_cfg = RigidHypothesisInitConfig(
        num_moving_parts=int(args.num_moving_parts),
        fm_core_dynamic_threshold=float(args.fm_core_dynamic_threshold),
        fm_core_static_threshold=float(args.fm_core_static_threshold),
        fm_core_confidence_threshold=float(args.fm_core_confidence_threshold),
        rigid_init_ransac_iters=int(args.rigid_init_ransac_iters),
        rigid_init_inlier_threshold=float(args.rigid_init_inlier_threshold),
        ransac_threshold_mode=str(args.ransac_threshold_mode),
        candidate_target_count=int(args.candidate_target_count),
        candidate_endpoint_knn=int(args.candidate_endpoint_knn),
        candidate_chunk_size=int(args.candidate_chunk_size),
        correspondence_em_iters=int(args.correspondence_em_iters),
        rigid_irls_iters=int(args.rigid_irls_iters),
        rigid_huber_delta=float(args.rigid_huber_delta),
        correspondence_isometry_k=int(args.correspondence_isometry_k),
        correspondence_isometry_weight=float(args.correspondence_isometry_weight),
        assignment_temperature=float(args.assignment_temperature),
        responsibility_gamma=float(args.responsibility_gamma),
        seed=int(args.seed),
    )
    state, init_summary = initialize_rigid_hypotheses(state, init_cfg, endpoint_target_xyz=endpoint_target)
    save_progressive_state_artifacts(state, init_dir, extra=init_summary)
    _write_json(init_dir / "init_summary.json", init_summary)
    torch.save(state.part_transforms_T.detach().cpu(), init_dir / "rigid_hypotheses.pt")
    rigid_baselines = _write_rigid_init_artifacts(state, init_summary, init_dir, endpoint_target, args)
    gt_axis_debug = _print_gt_reference_axis_debug(
        state,
        Path(args.root_scene).expanduser().resolve(),
        init_dir,
    )

    inner_initial = _inner_rigid_fit(state, inputs, endpoint_target, args, desc="Stage 1C inner rigid fit")
    gate_initial = _physical_gate(state, inputs, endpoint_target, args, tag="after_initial_inner")
    _write_json(stage_dir / "rigid_init" / "physical_gate_after_initial_inner.json", gate_initial)
    if not gate_initial["passed"]:
        raise RuntimeError("Rigid Stage 1 physical gate failed after initialization; STOP STAGE 1")
    progressive = _run_progressive_outer_loop(
        state,
        inputs,
        endpoint_target,
        args,
        stage_dir,
        target_scale=target_scale,
        loss_history=list(inner_initial.get("history", [])),
    )
    final_inner = {}
    if int(args.inner_rigid_iters_after_expand) > 0:
        original = int(args.inner_rigid_iters)
        args.inner_rigid_iters = int(args.inner_rigid_iters_after_expand)
        final_inner = _inner_rigid_fit(state, inputs, endpoint_target, args, desc="Stage 1C refit after expansion")
        args.inner_rigid_iters = original
        gate_final = _physical_gate(state, inputs, endpoint_target, args, tag="after_final_inner")
        _write_json(stage_dir / "final_physical_gate.json", gate_final)
        if not gate_final["passed"]:
            raise RuntimeError("Rigid Stage 1 physical gate failed after final refit")

    final_dir = stage_dir / "final"
    save_progressive_state_artifacts(
        state,
        final_dir,
        extra={"inner_initial": inner_initial, "progressive": progressive, "inner_final": final_inner},
    )
    state.save(stage_dir / "progressive_checkpoint.pt")
    torch.save(state.hard_labels.detach().cpu(), stage_dir / "final_stage1_labels.pt")
    torch.save(state.status.detach().cpu(), stage_dir / "final_stage1_status.pt")
    if bool(args.save_debug_html):
        write_progressive_html(state, stage_dir / "progressive_stage1.html", max_points=int(args.max_html_points))
        _write_fm_diagnostic_html(
            state, final_dir / "fm_diagnostic.html",
            fm_core_static_threshold=float(args.fm_core_static_threshold),
            fm_core_dynamic_threshold=float(args.fm_core_dynamic_threshold),
            spcd=float(target_scale),
            max_points=int(args.max_html_points),
        )
    gt_debug = _score_gt_debug(state, Path(args.root_scene).expanduser().resolve())
    summary = {
        "schema_version": 1,
        "pipeline": PIPELINE_NAME,
        "output_dir": str(stage_dir),
        "frames": bridge.frames,
        "build": build_summary,
        "fm_regression": regression,
        "config_lock": config_lock,
        "init": init_summary,
        "gt_axis_debug": gt_axis_debug,
        "rigid_baselines": rigid_baselines,
        "inner_initial": inner_initial,
        "progressive": progressive,
        "inner_final": final_inner,
        "final_counts": state.label_counts(),
        "gt_debug": gt_debug,
        "ground_truth_used_as_algorithm_input": False,
    }
    (stage_dir / "progressive_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[progressive] complete: {stage_dir}")
    print(f"[progressive] counts: {state.label_counts()}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-scene", required=True)
    parser.add_argument("--static-model-dir", required=True)
    parser.add_argument("--static-iteration", type=int, default=14000)
    parser.add_argument("--endpoint-static-model-dir", default="")
    parser.add_argument("--endpoint-static-iteration", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", default="endpoints")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--num-moving-parts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-max-gaussians", type=int, default=4096)
    parser.add_argument("--max-gaussians", type=int, default=0)

    parser.add_argument("--semantic-dynamic-threshold", type=float, default=0.5)
    parser.add_argument("--semantic-dynamic-min-points", type=int, default=32)
    parser.add_argument("--allow-all-gaussian-foreground", action="store_true")
    parser.add_argument("--flow-dir", default="")
    parser.add_argument("--correspondence-flow-max-side", type=int, default=512)
    parser.add_argument("--correspondence-cycle-threshold", type=float, default=1.5)
    parser.add_argument("--correspondence-dino-min-cosine", type=float, default=0.2)
    parser.add_argument("--correspondence-min-confidence", type=float, default=0.05)
    parser.add_argument("--correspondence-alpha-threshold", type=float, default=0.05)
    parser.add_argument("--observed-flow-velocity-min-observations", type=int, default=6)
    parser.add_argument("--observed-flow-velocity-min-cameras", type=int, default=3)
    parser.add_argument("--observed-flow-velocity-max-reprojection-px", type=float, default=0.75)

    parser.add_argument("--legacy-fm-path", default="")
    parser.add_argument("--legacy-fm-iters", type=int, default=1000)
    parser.add_argument("--legacy-fm-batch-size", type=int, default=2048)
    parser.add_argument("--legacy-fm-integration-steps", type=int, default=16)
    parser.add_argument("--legacy-fm-integrator", choices=("euler", "rk4"), default="euler")
    parser.add_argument("--legacy-fm-identity-weight", type=float, default=0.01)
    parser.add_argument("--legacy-fm-smooth-weight", type=float, default=0.0)
    parser.add_argument("--force-legacy-fm", action="store_true")
    parser.add_argument("--endpoint-transport-knn", type=int, default=16)
    parser.add_argument("--endpoint-transport-temperature", type=float, default=0.02)
    parser.add_argument("--endpoint-transport-robust-delta", type=float, default=0.02)
    parser.add_argument("--endpoint-transport-confidence-distance-scale", type=float, default=0.0)
    parser.add_argument("--endpoint-transport-confidence-entropy-scale", type=float, default=1.0)
    parser.add_argument("--endpoint-transport-chunk-size", type=int, default=1024)
    parser.add_argument("--endpoint-motion-scale", choices=("local_knn", "global_nn", "gaussian_scale"), default="local_knn")
    parser.add_argument("--fm-prior-prob-bias", type=float, default=1.0)
    parser.add_argument("--fm-prior-prob-temperature", type=float, default=0.35)
    parser.add_argument("--fm-regression-min-valid-fraction", type=float, default=0.05)
    parser.add_argument("--fm-regression-min-gt-recall", type=float, default=0.0)

    parser.add_argument("--strict-stage1-bugfix-validation", action="store_true")
    parser.add_argument("--debug-reference-axis", nargs=3, type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fm-core-dynamic-threshold", type=float, default=0.65)
    parser.add_argument("--fm-core-static-threshold", type=float, default=0.25)
    parser.add_argument("--fm-core-confidence-threshold", type=float, default=0.25)
    parser.add_argument("--rigid-init-ransac-iters", type=int, default=256)
    parser.add_argument("--rigid-init-inlier-threshold", type=float, default=0.04)
    parser.add_argument("--ransac-threshold-mode", choices=("derived", "manual"), default="derived")
    parser.add_argument("--candidate-target-count", type=int, default=32)
    parser.add_argument("--candidate-endpoint-knn", type=int, default=16)
    parser.add_argument("--candidate-chunk-size", type=int, default=1024)
    parser.add_argument("--correspondence-em-iters", type=int, default=8)
    parser.add_argument("--rigid-irls-iters", type=int, default=5)
    parser.add_argument(
        "--correspondence-isometry-k", type=int, default=8,
        help="t0 neighbours per source point used to enforce that neighbour distances are preserved at frame T (0 disables)",
    )
    parser.add_argument(
        "--correspondence-isometry-weight", type=float, default=1.0,
        help="weight on the neighbour-distance-preservation term in correspondence matching (0 disables)",
    )
    parser.add_argument("--rigid-huber-delta", type=float, default=0.02)
    parser.add_argument("--assignment-temperature", type=float, default=0.05)
    parser.add_argument("--assignment-margin", type=float, default=0.15)
    parser.add_argument("--responsibility-gamma", type=float, default=2.0)

    parser.add_argument("--inner-rigid-iters", type=int, default=120)
    parser.add_argument("--inner-rigid-iters-after-expand", type=int, default=60)
    parser.add_argument("--inner-rigid-lr", type=float, default=1.0e-2)
    parser.add_argument("--inner-trust-region", type=float, default=0.25)
    parser.add_argument("--lambda-fm", type=float, default=1.0)
    parser.add_argument("--lambda-pcd", type=float, default=0.25)
    parser.add_argument("--lambda-rgb", type=float, default=0.02)
    parser.add_argument("--lambda-depth", type=float, default=0.02)
    parser.add_argument("--lambda-diffusion", type=float, default=0.0)
    parser.add_argument("--pcd-fit-samples", type=int, default=4096)
    parser.add_argument("--pcd-target-samples", type=int, default=8192)
    parser.add_argument("--pcd-chunk-size", type=int, default=1024)
    parser.add_argument("--inner-render-cameras", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)

    parser.add_argument("--outer-iterations", type=int, default=4)
    parser.add_argument("--candidate-components-per-outer", type=int, default=2)
    parser.add_argument("--candidate-max-gaussians-per-outer", type=int, default=4096)
    parser.add_argument("--candidate-min-size", type=int, default=3)
    parser.add_argument("--candidate-knn", type=int, default=16)
    parser.add_argument("--candidate-distance-scale", type=float, default=0.05)
    parser.add_argument("--candidate-fm-scale", type=float, default=0.05)
    parser.add_argument("--component-min-reliability", type=float, default=0.05)
    parser.add_argument("--component-min-observable-fraction", type=float, default=0.1)

    parser.add_argument("--diffusion-hypothesis-test", action="store_true")
    parser.add_argument("--render-cameras-per-frame", type=int, default=2)
    parser.add_argument("--render-min-support", type=float, default=1.0e-5)
    parser.add_argument("--counterfactual-radius-px", type=int, default=4)
    parser.add_argument("--counterfactual-render-eps", type=float, default=1.0e-8)
    parser.add_argument("--counterfactual-min-affected-pixels", type=int, default=1)
    parser.add_argument("--use-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-debug-html", action="store_true")
    parser.add_argument("--max-html-points", type=int, default=20000)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.debug_reference_axis is not None:
        print("[progressive] ignoring --debug-reference-axis; GT debug axis is loaded from scene metadata after fitting")
    if args.legacy_fm_path:
        print(f"[progressive] loading legacy FM base from {args.legacy_fm_path}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
