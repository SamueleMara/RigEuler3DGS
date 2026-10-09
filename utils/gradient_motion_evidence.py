"""ScrewAIR-v25 static-pose residual-responsibility attribution.

This module intentionally restores the historical working Phase-1 motion clue:
canonical/frame-0 Gaussians are rendered against frame-0 and future targets and
residual responsibility is back-projected through all available differentiable
Gaussian channels.  The primary motion evidence is exactly
``attr_combined_motion_excess_positive``.

No optimizer step, velocity update, integration, clustering refinement, or
rigidity update happens here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, Sequence

import torch
from tqdm import tqdm


def _ordered_groups(cameras: Sequence):
    groups: Dict[str, list] = {}
    for camera in cameras:
        key = str(getattr(camera, "dynamic_camera_id", getattr(camera, "image_name", "camera")))
        groups.setdefault(key, []).append(camera)
    for key in groups:
        groups[key].sort(key=lambda camera: int(getattr(camera, "frame_id", 0)))
    return sorted(groups.items())


def _uniform_future_indices(frame_count: int, maximum: int):
    if frame_count <= 1:
        return []
    available = torch.arange(1, frame_count)
    if maximum <= 0 or available.numel() <= maximum:
        return available.tolist()
    return torch.linspace(1, frame_count - 1, maximum).round().long().unique().tolist()


def _normalise_fg_channel(values: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
    """Historical v25 per-sample foreground normalization."""
    values = torch.nan_to_num(
        values.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    fg_values = values[foreground_mask]
    positive = fg_values[fg_values > 0]
    if positive.numel() > 16:
        scale = torch.quantile(positive, 0.50).clamp_min(1.0e-12)
    else:
        scale = fg_values.mean().clamp_min(1.0e-12) if fg_values.numel() else values.new_tensor(1.0)
    return (values / scale).clamp(0.0, 50.0)


def _row_norm_or_zero(gradient: torch.Tensor | None, count: int, device) -> torch.Tensor:
    if gradient is None:
        return torch.zeros(count, device=device)
    gradient = gradient.detach()
    if gradient.ndim == 0 or gradient.shape[0] != count:
        return torch.zeros(count, device=device)
    if gradient.ndim == 1:
        output = gradient.abs()
    else:
        output = torch.linalg.norm(gradient.reshape(count, -1), dim=-1)
    return torch.nan_to_num(output.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _scatter_viewspace_grad(
    gradient: torch.Tensor | None,
    visibility_filter,
    count: int,
    device,
) -> torch.Tensor:
    output = torch.zeros(count, device=device)
    if gradient is None:
        return output
    gradient = gradient.detach()
    if gradient.ndim != 2 or gradient.shape[0] == 0:
        return output
    values = torch.linalg.norm(gradient[:, :2], dim=-1)
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    visible = None
    if visibility_filter is not None:
        try:
            visible = visibility_filter.detach().to(device=device).reshape(-1).bool()
            if visible.numel() != count:
                visible = None
        except Exception:
            visible = None
    if values.numel() == count:
        output[:] = values
        if visible is not None:
            output[~visible] = 0.0
    elif visible is not None and int(visible.sum().item()) == int(values.numel()):
        output[visible] = values
    else:
        n = min(count, int(values.numel()))
        output[:n] = values[:n]
    return output


def _hard_binary_split(score: torch.Tensor, foreground_mask: torch.Tensor, min_dynamic: int):
    """Diagnostic-only two-way split; it does not define the motion evidence."""
    ids = torch.nonzero(foreground_mask, as_tuple=False).flatten()
    values = torch.log1p(score[ids].clamp_min(0.0))
    if values.numel() < 2 or float(values.max() - values.min()) <= 1.0e-12:
        hard = torch.zeros_like(foreground_mask)
        if ids.numel():
            hard[ids[: min(max(1, int(min_dynamic)), int(ids.numel()))]] = True
        return hard, values.new_zeros(2)
    centers = torch.stack((values.min(), values.max()))
    labels = torch.zeros_like(values, dtype=torch.long)
    for _ in range(32):
        new_labels = (values[:, None] - centers[None, :]).abs().argmin(dim=1)
        new_centers = centers.clone()
        for cluster in range(2):
            selected = values[new_labels == cluster]
            if selected.numel():
                new_centers[cluster] = selected.mean()
        if torch.equal(new_labels, labels) and torch.allclose(new_centers, centers):
            labels, centers = new_labels, new_centers
            break
        labels, centers = new_labels, new_centers
    moving_cluster = int(centers.argmax())
    selected = labels == moving_cluster
    required = min(max(1, int(min_dynamic)), int(ids.numel()))
    if int(selected.sum()) < required:
        selected = torch.zeros_like(selected)
        selected[torch.topk(values, k=required).indices] = True
    hard = torch.zeros_like(foreground_mask)
    hard[ids[selected]] = True
    return hard, centers




def _optimize_static_consistency_score(
    static_mean: torch.Tensor,
    future_mean: torch.Tensor,
    foreground_mask: torch.Tensor,
    *,
    steps: int = 24,
    lr: float = 0.20,
    prior_weight: float = 0.05,
):
    """Short per-Gaussian minimization of frame0/future attribution mismatch.

    The dynamic prior uses the positive attribution excess

        max(g_future - g_frame0, 0).

    For static evidence we instead ask which Gaussians can keep a large static
    responsibility while minimizing the *relative absolute* attribution
    positive discrepancy used by the dynamic prior

        delta_i+ = max(g_future - g_frame0, 0) / (g_future + g_frame0 + eps).

    A bounded static score s_i=sigmoid(l_i) is optimized with

        L = mean[(s_i delta_i)^2 + lambda (1-s_i)^2].

    Thus a Gaussian is allowed to have high static confidence only when its
    frame-0 and future residual responsibilities are consistent.  This is a
    diagnostic confidence field; it does not alter Phase-1 assignments here.
    """
    static_mean = torch.nan_to_num(static_mean.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    future_mean = torch.nan_to_num(future_mean.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    fg = foreground_mask.reshape(-1).bool().to(static_mean.device)
    denom = (static_mean + future_mean).clamp_min(1.0e-6)
    relative_difference = ((future_mean - static_mean).abs() / denom).clamp(0.0, 1.0)
    relative_positive_excess = ((future_mean - static_mean).clamp_min(0.0) / denom).clamp(0.0, 1.0)

    score = torch.zeros_like(relative_difference)
    if not bool(fg.any()):
        return score, relative_difference, relative_positive_excess, {
            "steps": 0, "lr": float(lr), "prior_weight": float(prior_weight),
            "initial_loss": 0.0, "final_loss": 0.0,
        }

    initial = (1.0 - relative_positive_excess).clamp(0.01, 0.99)
    logits = torch.logit(initial).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=max(1.0e-6, float(lr)))
    lam = max(1.0e-8, float(prior_weight))
    steps = max(0, int(steps))

    def objective():
        s = torch.sigmoid(logits)
        sf = s[fg]
        df = relative_positive_excess[fg]
        return ((sf * df) ** 2 + lam * (1.0 - sf) ** 2).mean()

    with torch.enable_grad():
        initial_loss = float(objective().detach().cpu())
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            optimizer.step()
        final_loss = float(objective().detach().cpu())

    score = torch.sigmoid(logits.detach())
    score = torch.where(fg, score, torch.zeros_like(score))
    return score, relative_difference, relative_positive_excess, {
        "steps": steps, "lr": float(lr), "prior_weight": lam,
        "initial_loss": initial_loss, "final_loss": final_loss,
    }

def compute_gradient_motion_evidence(
    cameras: Sequence,
    gaussians,
    visual_loss_fn: Callable,
    foreground_mask: torch.Tensor,
    max_future_frames: int = 0,
    max_cameras: int = 0,
    min_dynamic_points: int = 32,
    static_refine_steps: int = 24,
    static_refine_lr: float = 0.20,
    static_refine_prior_weight: float = 0.05,
):
    """Restore the historical v25 multi-channel gradient attribution exactly.

    For every physical camera:
      * render canonical/frame-0 Gaussians against that camera's frame-0 target;
      * render the SAME canonical/frame-0 Gaussians against selected future targets;
      * back-project each residual through xyz, screen/viewspace, opacity,
        appearance, DINO-feature, and shape parameters;
      * independently normalize each channel on the semantic foreground;
      * form the historical weighted ``combined`` channel;
      * average static and future responsibility fields separately;
      * define the motion clue as
        ``attr_combined_motion_excess_positive = clamp(future-static, 0)``.

    ``visual_loss_fn`` must return ``(loss, stats, render_package)`` and must
    render the canonical scene.  This function never calls an optimizer.
    """
    groups = _ordered_groups(cameras)
    if max_cameras > 0:
        groups = groups[: int(max_cameras)]
    if not groups:
        raise RuntimeError("No physical camera groups are available for motion attribution")

    xyz = gaussians.get_canonical_xyz
    device = xyz.device
    count = int(xyz.shape[0])
    foreground_mask = foreground_mask.to(device).reshape(-1).bool()
    if foreground_mask.numel() != count:
        raise ValueError("foreground_mask must match the Gaussian count")

    channel_names = ["xyz", "screen", "opacity", "appearance", "dino", "shape", "combined"]
    static_sum = {name: torch.zeros(count, device=device) for name in channel_names}
    future_sum = {name: torch.zeros(count, device=device) for name in channel_names}
    static_count = {name: 0 for name in channel_names}
    future_count = {name: 0 for name in channel_names}

    # Historical v25 temporarily enabled every canonical differentiable channel.
    attribution_params = []
    for name in [
        "_xyz", "_opacity", "_scaling", "_rotation",
        "_features_dc", "_features_rest", "_dino_feature_field",
    ]:
        tensor = getattr(gaussians, name, None)
        if torch.is_tensor(tensor) and tensor.numel() > 0:
            attribution_params.append((name, tensor, bool(tensor.requires_grad)))
            try:
                tensor.requires_grad_(True)
            except Exception:
                pass

    probe_plan = []
    for key, sequence in groups:
        if sequence:
            probe_plan.append((
                key,
                sequence,
                _uniform_future_indices(len(sequence), int(max_future_frames)),
            ))
    total_probes = sum(1 + len(indices) for _, _, indices in probe_plan)
    progress = tqdm(
        total=total_probes,
        desc="P1 attribution init",
        unit="probe",
        dynamic_ncols=True,
        mininterval=0.25,
        smoothing=0.1,
    )
    records = []
    samples = []

    def measure(camera, phase: str, camera_key: str):
        # Canonical/frame-0 geometry for BOTH static and future targets.
        gaussians.clear_time()
        progress.set_description(
            f"P1 attribution | {camera_key} | F{int(getattr(camera, 'frame_id', 0))}",
            refresh=False,
        )
        loss, stats, package = visual_loss_fn(camera, stage="attribution")
        if not loss.requires_grad:
            raise RuntimeError(
                "v25 gradient attribution produced a non-differentiable loss; "
                "canonical Gaussian parameters must be differentiable during attribution"
            )

        named_targets = []
        for name, tensor, _ in attribution_params:
            if torch.is_tensor(tensor) and tensor.requires_grad:
                named_targets.append((name, tensor))
        viewspace = package.get("viewspace_points", None) if isinstance(package, dict) else None
        if torch.is_tensor(viewspace) and viewspace.requires_grad:
            named_targets.append(("viewspace_points", viewspace))
        gradients = torch.autograd.grad(
            loss,
            [tensor for _, tensor in named_targets],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        gradient_by_name = {
            name: gradient for (name, _), gradient in zip(named_targets, gradients)
        }

        raw_xyz = _row_norm_or_zero(gradient_by_name.get("_xyz"), count, device)
        raw_opacity = _row_norm_or_zero(gradient_by_name.get("_opacity"), count, device)
        raw_scaling = _row_norm_or_zero(gradient_by_name.get("_scaling"), count, device)
        raw_rotation = _row_norm_or_zero(gradient_by_name.get("_rotation"), count, device)
        raw_shape = raw_scaling + raw_rotation
        raw_dc = _row_norm_or_zero(gradient_by_name.get("_features_dc"), count, device)
        raw_rest = _row_norm_or_zero(gradient_by_name.get("_features_rest"), count, device)
        raw_appearance = raw_dc + raw_rest
        raw_dino = _row_norm_or_zero(gradient_by_name.get("_dino_feature_field"), count, device)
        raw_screen = _scatter_viewspace_grad(
            gradient_by_name.get("viewspace_points"),
            package.get("visibility_filter", None) if isinstance(package, dict) else None,
            count,
            device,
        )

        channels = {
            "xyz": _normalise_fg_channel(raw_xyz, foreground_mask),
            "screen": _normalise_fg_channel(raw_screen, foreground_mask),
            "opacity": _normalise_fg_channel(raw_opacity, foreground_mask),
            "appearance": _normalise_fg_channel(raw_appearance, foreground_mask),
            "dino": _normalise_fg_channel(raw_dino, foreground_mask),
            "shape": _normalise_fg_channel(raw_shape, foreground_mask),
        }
        # EXACT historical v25 weights.
        weights = {
            "screen": 1.00,
            "xyz": 0.75,
            "opacity": 0.75,
            "appearance": 0.35,
            "dino": 0.35,
            "shape": 0.25,
        }
        combined = torch.zeros(count, device=device)
        denominator = 0.0
        for name, weight in weights.items():
            channel = channels[name]
            if torch.isfinite(channel).any() and float(channel.sum().detach().item()) > 0.0:
                combined = combined + float(weight) * channel
                denominator += float(weight)
        combined = combined / denominator if denominator > 0.0 else channels["xyz"]
        channels["combined"] = _normalise_fg_channel(combined, foreground_mask)

        fg_combined = channels["combined"][foreground_mask]
        record = {
            "camera": str(camera_key),
            "target_frame_idx": int(getattr(camera, "frame_id", 0)),
            "render_pose_frame_idx": 0,
            "phase": str(phase),
            "is_static_frame0_baseline": bool(phase == "static_frame0"),
            "is_future_motion_attribution": bool(phase == "motion_future"),
            "loss_total": float(loss.detach().cpu().item()),
            "combined_fg_mean": float(fg_combined.mean().detach().cpu().item()) if fg_combined.numel() else 0.0,
            "combined_fg_p50": float(torch.quantile(fg_combined, 0.50).detach().cpu().item()) if fg_combined.numel() else 0.0,
            "combined_fg_p90": float(torch.quantile(fg_combined, 0.90).detach().cpu().item()) if fg_combined.numel() else 0.0,
            "visual_terms": stats,
        }
        for name in ["xyz", "screen", "opacity", "appearance", "dino", "shape"]:
            vals = channels[name][foreground_mask]
            record[f"{name}_fg_mean"] = float(vals.mean().detach().cpu().item()) if vals.numel() else 0.0
            record[f"{name}_fg_p90"] = float(torch.quantile(vals, 0.90).detach().cpu().item()) if vals.numel() else 0.0
        records.append(record)
        samples.append((str(camera_key), int(getattr(camera, "frame_id", 0))))
        progress.set_postfix(
            Loss=f"{float(loss.detach()):.5f}",
            Attr=f"{record['combined_fg_mean']:.4f}",
            refresh=False,
        )
        progress.update(1)
        return channels

    try:
        for key, sequence, future_indices in probe_plan:
            channels0 = measure(sequence[0], "static_frame0", key)
            for name in channel_names:
                static_sum[name] += channels0[name]
                static_count[name] += 1
            for index in future_indices:
                channels = measure(sequence[index], "motion_future", key)
                for name in channel_names:
                    future_sum[name] += channels[name]
                    future_count[name] += 1
    finally:
        progress.close()
        gaussians.clear_time()
        for _, tensor, old_requires_grad in attribution_params:
            try:
                tensor.requires_grad_(old_requires_grad)
            except Exception:
                pass

    if static_count["combined"] == 0 or future_count["combined"] == 0:
        raise RuntimeError("Motion attribution needs at least one frame-0 and one future render")

    state = {
        "schema_version": 25,
        "definition": "ScrewAIR v25 canonical frame0-vs-future multi-channel residual responsibility",
        "selected_motion_channel": "combined",
        "selected_motion_evidence_source": "attr_combined_motion_excess_positive",
        "canonical_xyz": xyz.detach().cpu(),
        "semantic_foreground_mask": foreground_mask.detach().cpu(),
        "records": records,
        "samples": samples,
        "static_render_count": int(static_count["combined"]),
        "future_render_count": int(future_count["combined"]),
    }

    for name in channel_names:
        static_mean = static_sum[name] / float(max(1, static_count[name]))
        future_mean = future_sum[name] / float(max(1, future_count[name]))
        excess_positive = (future_mean - static_mean).clamp_min(0.0)
        state[f"attr_{name}_static_frame0_mean"] = static_mean.detach().cpu()
        state[f"attr_{name}_motion_future_mean"] = future_mean.detach().cpu()
        state[f"attr_{name}_motion_excess_positive"] = excess_positive.detach().cpu()

    # EXACT working clue requested by the user.
    raw_evidence = state["attr_combined_motion_excess_positive"].to(device)
    state["frame0_gradient"] = state["attr_combined_static_frame0_mean"]
    state["future_static_pose_gradient"] = state["attr_combined_motion_future_mean"]
    state["motion_excess"] = (
        state["attr_combined_motion_future_mean"] - state["attr_combined_static_frame0_mean"]
    )
    state["motion_evidence"] = raw_evidence.detach().cpu()

    # The current Phase-1 membership parameter expects a bounded clue.  Keep the
    # historical raw field untouched above; derive a separate bounded copy only
    # for initialization of that modern parameterization.
    fg_raw = raw_evidence[foreground_mask]
    scale = torch.quantile(fg_raw, 0.95).clamp_min(1.0e-12) if fg_raw.numel() else raw_evidence.new_tensor(1.0)
    bounded = (raw_evidence / scale).clamp(0.0, 1.0)
    hard_dynamic, centers = _hard_binary_split(
        raw_evidence, foreground_mask, min_dynamic=int(min_dynamic_points)
    )
    state["motion_evidence_normalized"] = bounded.detach().cpu()
    state["initial_motion_membership"] = torch.where(
        foreground_mask, bounded.clamp(0.5, 1.0), torch.zeros_like(bounded)
    ).detach().cpu()
    state["initial_dynamic_mask"] = hard_dynamic.detach().cpu()
    state["kmeans_log_score_centers"] = centers.detach().cpu()

    # Complementary static evidence: unlike the dynamic positive-excess clue,
    # this explicitly minimizes the frame0-vs-future attribution discrepancy.
    static_score, static_rel_diff, static_rel_positive, static_optim = _optimize_static_consistency_score(
        state["attr_combined_static_frame0_mean"].to(device),
        state["attr_combined_motion_future_mean"].to(device),
        foreground_mask,
        steps=int(static_refine_steps),
        lr=float(static_refine_lr),
        prior_weight=float(static_refine_prior_weight),
    )
    state["schema_version"] = 26
    state["selected_static_evidence_source"] = "diagnostic_only_not_used_for_v8_labels"
    state["static_gradient_relative_difference"] = static_rel_diff.detach().cpu()
    state["static_gradient_relative_positive_excess"] = static_rel_positive.detach().cpu()
    state["static_consistency_score"] = static_score.detach().cpu()
    state["static_consistency_optimization"] = static_optim
    return state


def write_gradient_motion_evidence_html(state: Dict, output_path, max_points: int = 50000):
    """Write the historical v25 attribution visualization.

    The old writer is deliberately reused rather than replacing the debug with a
    new interpretation.  ``attr_combined_motion_excess_positive`` is included
    and selected by the saved metadata exactly as in the working implementation.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from utils.screw_air_gradient_attribution_debug import write_gradient_attribution_debug

        xyz = state["canonical_xyz"].float()
        foreground = state["semantic_foreground_mask"].bool()
        raw_cluster = torch.full((xyz.shape[0],), -1, dtype=torch.long)
        raw_cluster[foreground] = 0
        stats = {
            "grad_total_mean": state["attr_combined_motion_future_mean"],
            "grad_static_frame0_mean": state["attr_combined_static_frame0_mean"],
            "grad_motion_future_mean": state["attr_combined_motion_future_mean"],
            "grad_motion_excess": state["motion_excess"],
            "grad_motion_excess_positive": state["attr_combined_motion_excess_positive"],
            "grad_motion_ratio": state["attr_combined_motion_future_mean"] / state["attr_combined_static_frame0_mean"].clamp_min(1e-6),
            "raw_future_evidence": state["attr_combined_motion_future_mean"],
            "selected_motion_evidence": state["attr_combined_motion_excess_positive"],
            "grad_count": torch.full((xyz.shape[0],), float(state.get("static_render_count", 0) + state.get("future_render_count", 0))),
            "meta_selected_motion_channel": "combined",
            "meta_selected_motion_evidence_source": "attr_combined_motion_excess_positive",
            "static_gradient_relative_difference": state.get("static_gradient_relative_difference", torch.zeros_like(xyz[:, 0])),
            "static_gradient_relative_positive_excess": state.get("static_gradient_relative_positive_excess", torch.zeros_like(xyz[:, 0])),
            "static_consistency_score": state.get("static_consistency_score", torch.zeros_like(xyz[:, 0])),
            # v8: the short static-consistency probe is diagnostic only.  There is
            # deliberately no precomputed multiview STATIC/DYNAMIC/UNKNOWN state.
            # Material labels are optimized contextually during Phase 1.
            "meta_selected_static_evidence_source": "none_jointly_optimized_in_phase1",
        }
        for key, value in state.items():
            if isinstance(key, str) and key.startswith("attr_") and torch.is_tensor(value):
                stats[key] = value
        old_path = write_gradient_attribution_debug(
            out_dir=output_path.parent,
            xyz=xyz,
            foreground_mask=foreground,
            raw_cluster_id=raw_cluster,
            selected_moving_mask=state.get("initial_dynamic_mask", None),
            stats=stats,
            samples=state.get("samples", []),
            sample_records=state.get("records", []),
            cluster_quality=None,
            max_points=int(max_points),
            top_k=128,
        )
        old_path = Path(old_path)
        if old_path.resolve() != output_path.resolve():
            shutil.copyfile(old_path, output_path)
        return output_path
    except Exception as exc:
        raise RuntimeError(f"Failed to write restored v25 gradient attribution debug: {exc}") from exc
