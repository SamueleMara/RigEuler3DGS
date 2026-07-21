"""Final articulated-joint estimation from the learned rigid twist trajectory.

The dynamic model predicts one foreground rigid twist

    v(x,t) = omega(t) x (x - motion_center) + u(t)

and one trainable foreground static/moving assignment.  This module performs a
post-training model selection among:

* prismatic: constant spatial velocity parallel to one axis;
* revolute: rotation around one fixed axis with zero axial pitch;
* screw: rotation around one fixed axis plus axial translation h * angular_rate.

Semantic isolation is strict for joint understanding:

* only ``dynamic_foreground_mask`` points can enter the moving-part mask;
* non-foreground Gaussians can have arbitrary learned proposal velocity, but
  they never contribute to type selection, pitch, axis, anchor, or residuals.

The module writes:

* ``final_joint_estimation.json``;
* ``final_joint_estimation.html``;
* ``final_joint_model_scores.html``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import math

import numpy as np
import torch


_EPS = 1e-9


@dataclass
class JointEstimationConfig:
    dynamic_threshold: float = 0.80
    min_moving_points: int = 64
    max_fit_points: int = 4096
    max_html_points: int = 20000
    axis_length_scale: float = 1.25
    omega_activity_threshold: float = 1e-3
    prismatic_rotation_ratio: float = 0.10
    screw_min_pitch: float = 1e-3
    screw_min_axial_ratio: float = 0.03
    screw_min_relative_improvement: float = 0.05
    screw_complexity_penalty: float = 0.005
    frame_activity_quantile: float = 0.10
    random_seed: int = 12345


@dataclass
class CandidateScore:
    joint_type: str
    rmse: float
    normalized_rmse: float
    score: float


def _as_cpu_float(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _unit(x: torch.Tensor, fallback: Optional[torch.Tensor] = None) -> torch.Tensor:
    n = torch.linalg.norm(x)
    if torch.isfinite(n) and float(n) > _EPS:
        return x / n
    if fallback is None:
        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=x.dtype)
    return fallback / torch.linalg.norm(fallback).clamp_min(_EPS)


def _axis_from_unsigned_vectors(vectors: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fit a sign-invariant principal axis from direction vectors."""
    vectors = _as_cpu_float(vectors)
    norms = torch.linalg.norm(vectors, dim=-1)
    valid = torch.isfinite(vectors).all(dim=-1) & (norms > _EPS)
    if not valid.any():
        return torch.tensor([1.0, 0.0, 0.0])
    dirs = vectors[valid] / norms[valid, None]
    if weights is None:
        w = norms[valid]
    else:
        w = _as_cpu_float(weights).reshape(-1)[valid].clamp_min(0.0)
    w = w / w.sum().clamp_min(_EPS)
    scatter = torch.einsum("n,ni,nj->ij", w, dirs, dirs)
    _, eigvec = torch.linalg.eigh(scatter)
    axis = eigvec[:, -1]
    # Deterministic sign: align with the largest-magnitude observed vector.
    reference = vectors[valid][torch.argmax(norms[valid])]
    if torch.dot(axis, reference) < 0:
        axis = -axis
    return _unit(axis)


def _weighted_geometric_median(points: torch.Tensor, weights: torch.Tensor, iterations: int = 64) -> torch.Tensor:
    points = _as_cpu_float(points)
    weights = _as_cpu_float(weights).reshape(-1).clamp_min(0.0)
    valid = torch.isfinite(points).all(dim=-1) & torch.isfinite(weights) & (weights > 0)
    if not valid.any():
        return torch.zeros(3)
    p = points[valid]
    w = weights[valid]
    y = (p * w[:, None]).sum(dim=0) / w.sum().clamp_min(_EPS)
    for _ in range(max(1, int(iterations))):
        d = torch.linalg.norm(p - y[None], dim=-1).clamp_min(1e-6)
        ww = w / d
        y_new = (p * ww[:, None]).sum(dim=0) / ww.sum().clamp_min(_EPS)
        if torch.linalg.norm(y_new - y) < 1e-7:
            y = y_new
            break
        y = y_new
    return y


def _subsample_indices(indices: torch.Tensor, maximum: int, seed: int) -> torch.Tensor:
    indices = indices.detach().cpu().long().reshape(-1)
    if maximum <= 0 or indices.numel() <= maximum:
        return indices
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(indices.numel(), generator=gen)[:maximum]
    return indices[perm]


def _semantic_foreground_mask(tracks: Dict[str, Any]) -> Tuple[torch.Tensor, str]:
    """Return the semantic foreground mask, recovering from stale all-true exports.

    Some older track exporters wrote an all-true ``dynamic_foreground_mask`` even
    though ``semantic_prob`` and ``dynamic_foreground_threshold`` were present.
    In that specific inconsistent case, reconstruct the intended semantic mask
    from the frozen semantic field rather than silently fitting the joint on the
    whole scene.
    """
    x = tracks["canonical_positions"]
    n = int(x.shape[0])
    fg = tracks.get("dynamic_foreground_mask", None)
    if not torch.is_tensor(fg) or fg.numel() != n:
        raise ValueError("final joint estimation requires dynamic_foreground_mask")
    fg = fg.detach().cpu().bool().reshape(-1)
    source = "dynamic_foreground_mask"

    sem = tracks.get("semantic_prob", None)
    threshold = tracks.get("dynamic_foreground_threshold", None)
    if torch.is_tensor(sem) and sem.numel() == n and threshold is not None:
        sem = sem.detach().cpu().float().reshape(-1)
        if torch.is_tensor(threshold):
            threshold = float(threshold.detach().cpu().reshape(-1)[0].item())
        else:
            threshold = float(threshold)
        derived = sem >= threshold
        if bool(fg.all()) and bool((~derived).any()):
            fg = derived
            source = "recovered_from_semantic_prob_threshold"
    return fg, source


def _select_moving_mask(tracks: Dict[str, Any], cfg: JointEstimationConfig) -> Tuple[torch.Tensor, str]:
    x = tracks["canonical_positions"]
    n = int(x.shape[0])
    fg, foreground_source = _semantic_foreground_mask(tracks)

    p = tracks.get("dynamic_prob", None)
    if not torch.is_tensor(p) or p.numel() != n:
        raise ValueError("final joint estimation requires per-Gaussian dynamic_prob")
    p = p.detach().cpu().float().reshape(-1)

    cid = tracks.get("joint_cluster_id", None)
    source = f"dynamic_probability|{foreground_source}"
    moving = fg & (p >= float(cfg.dynamic_threshold))

    # In the corrected exact branch the trainable part field has exactly two
    # slots {0: static, 1: moving}. Use it when and only when this invariant is
    # actually present. Older over-clustered runs may contain many ids and must
    # not be interpreted as multiple moving parts here.
    if torch.is_tensor(cid) and cid.numel() == n:
        cid = cid.detach().cpu().long().reshape(-1)
        unique_fg = torch.unique(cid[fg])
        unique_nonnegative = unique_fg[unique_fg >= 0]
        if unique_nonnegative.numel() == 2 and set(unique_nonnegative.tolist()) == {0, 1}:
            moving = fg & (cid == 1)
            source = f"exact_two_slot_joint_cluster|{foreground_source}"

    if int(moving.sum()) < int(cfg.min_moving_points):
        raise RuntimeError(
            f"Only {int(moving.sum())} semantic-foreground Gaussians are assigned to the moving part; "
            f"at least {int(cfg.min_moving_points)} are required. Refusing a top-k/background fallback."
        )
    return moving, source


def _extract_twist(tracks: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    omega = tracks.get("rigid_omega", None)
    trans = tracks.get("rigid_translation", None)
    center = tracks.get("rigid_center", None)
    if not (torch.is_tensor(omega) and omega.ndim == 2 and omega.shape[-1] == 3):
        raise ValueError("tracks are missing rigid_omega [T,3]")
    if not (torch.is_tensor(trans) and trans.shape == omega.shape):
        raise ValueError("tracks are missing rigid_translation [T,3]")
    if not (torch.is_tensor(center) and center.numel() == 3):
        raise ValueError("tracks are missing rigid_center [3]")
    return _as_cpu_float(omega), _as_cpu_float(trans), _as_cpu_float(center).reshape(3)


def _target_rigid_velocity(
    tracks: Dict[str, Any],
    omega: torch.Tensor,
    trans: torch.Tensor,
    center: torch.Tensor,
    point_ids: torch.Tensor,
) -> torch.Tensor:
    rv = tracks.get("rigid_velocities", None)
    if torch.is_tensor(rv) and rv.ndim == 3 and rv.shape[0] == omega.shape[0] and rv.shape[-1] == 3:
        return _as_cpu_float(rv[:, point_ids])
    x = _as_cpu_float(tracks["canonical_positions"])[point_ids]
    rel = x[None] - center[None, None]
    return torch.cross(omega[:, None].expand_as(rel), rel, dim=-1) + trans[:, None]


def _axis_anchor_from_center(
    omega: torch.Tensor,
    trans: torch.Tensor,
    center: torch.Tensor,
    axis: torch.Tensor,
    active: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Axis point closest to the learned motion center.

    For v = omega x (x-c) + u, decompose u into axial and perpendicular
    components.  The equivalent axis point q closest to c is

        q = c + (omega x u_perp) / ||omega||^2.
    """
    om = omega[active]
    uu = trans[active]
    norm2 = (om * om).sum(dim=-1).clamp_min(1e-8)
    om_dir = om / torch.sqrt(norm2)[:, None]
    u_parallel = (uu * om_dir).sum(dim=-1, keepdim=True) * om_dir
    u_perp = uu - u_parallel
    q_t = center[None] + torch.cross(om, u_perp, dim=-1) / norm2[:, None]
    weights = torch.sqrt(norm2)
    q_robust = _weighted_geometric_median(q_t, weights)
    # Enforce the canonical anchor convention: closest point on the fitted axis
    # to the learned motion center.
    delta = q_robust - center
    q = center + delta - torch.dot(delta, axis) * axis
    return q, q_t


def _weighted_rmse(error: torch.Tensor, point_weights: torch.Tensor, frame_weights: torch.Tensor) -> float:
    sq = (error * error).sum(dim=-1)
    w = frame_weights[:, None] * point_weights[None]
    return float(torch.sqrt((sq * w).sum() / w.sum().clamp_min(_EPS)).item())


def estimate_final_joint(
    tracks: Dict[str, Any],
    config: Optional[JointEstimationConfig] = None,
) -> Dict[str, Any]:
    cfg = config or JointEstimationConfig()
    moving_mask, moving_source = _select_moving_mask(tracks, cfg)
    omega, trans, center = _extract_twist(tracks)
    x_all = _as_cpu_float(tracks["canonical_positions"])
    p_all = _as_cpu_float(tracks["dynamic_prob"]).reshape(-1)
    fg, foreground_mask_source = _semantic_foreground_mask(tracks)

    moving_ids_all = torch.where(moving_mask)[0]
    fit_ids = _subsample_indices(moving_ids_all, cfg.max_fit_points, cfg.random_seed)
    x = x_all[fit_ids]
    target = _target_rigid_velocity(tracks, omega, trans, center, fit_ids)

    point_weights = p_all[fit_ids].clamp_min(1e-3)
    point_weights = point_weights / point_weights.mean().clamp_min(_EPS)
    prismatic_anchor = (x * point_weights[:, None]).sum(dim=0) / point_weights.sum().clamp_min(_EPS)
    frame_speed = torch.sqrt((target * target).sum(dim=-1).mean(dim=-1))
    activity_floor = float(torch.quantile(frame_speed, min(max(cfg.frame_activity_quantile, 0.0), 0.95)).item())
    active = frame_speed > max(activity_floor, 1e-7)
    if int(active.sum()) < 3:
        active = torch.ones_like(frame_speed, dtype=torch.bool)
    frame_weights = frame_speed.clamp_min(frame_speed[active].median().clamp_min(1e-6) * 0.05)
    frame_weights = frame_weights / frame_weights.mean().clamp_min(_EPS)

    # Prismatic candidate.
    mean_velocity = (target * point_weights[None, :, None]).sum(dim=1) / point_weights.sum().clamp_min(_EPS)
    prism_axis = _axis_from_unsigned_vectors(mean_velocity[active], frame_weights[active])
    prism_rate = mean_velocity.matmul(prism_axis)
    v_prismatic = prism_rate[:, None, None] * prism_axis[None, None]
    v_prismatic = v_prismatic.expand_as(target)

    # Rotational/screw candidates.
    omega_norm = torch.linalg.norm(omega, dim=-1)
    omega_active = active & (omega_norm > float(cfg.omega_activity_threshold))
    if int(omega_active.sum()) < 3:
        omega_active = omega_norm > max(float(cfg.omega_activity_threshold) * 0.1, 1e-8)
    rot_axis = _axis_from_unsigned_vectors(omega[omega_active], omega_norm[omega_active]) if omega_active.any() else prism_axis
    signed_alpha = omega.matmul(rot_axis)

    if omega_active.any():
        axis_anchor, per_frame_axis_points = _axis_anchor_from_center(
            omega, trans, center, rot_axis, omega_active
        )
    else:
        axis_anchor = center.clone()
        per_frame_axis_points = center.reshape(1, 3)

    rel_axis = x - axis_anchor[None]
    cross_basis = torch.cross(rot_axis[None].expand_as(rel_axis), rel_axis, dim=-1)
    v_revolute = signed_alpha[:, None, None] * cross_basis[None]

    beta = trans.matmul(rot_axis)
    pitch_weights = frame_weights * signed_alpha.abs()
    denom = (pitch_weights * signed_alpha * signed_alpha).sum().clamp_min(_EPS)
    pitch_per_radian = float(((pitch_weights * signed_alpha * beta).sum() / denom).item())
    axial_speed = pitch_per_radian * signed_alpha
    v_screw = v_revolute + axial_speed[:, None, None] * rot_axis[None, None]

    target_rms = _weighted_rmse(target, point_weights, frame_weights)
    target_rms = max(target_rms, 1e-8)
    rmse_p = _weighted_rmse(v_prismatic - target, point_weights, frame_weights)
    rmse_r = _weighted_rmse(v_revolute - target, point_weights, frame_weights)
    rmse_s = _weighted_rmse(v_screw - target, point_weights, frame_weights)

    norm_p = rmse_p / target_rms
    norm_r = rmse_r / target_rms
    norm_s = rmse_s / target_rms
    score_p = norm_p
    score_r = norm_r
    score_s = norm_s + float(cfg.screw_complexity_penalty)

    radius = torch.linalg.norm(
        (x - axis_anchor[None]) - ((x - axis_anchor[None]).matmul(rot_axis))[:, None] * rot_axis[None],
        dim=-1,
    )
    characteristic_radius = float(torch.quantile(radius, 0.5).item()) if radius.numel() else 0.0
    angular_linear_rms = float(torch.sqrt((signed_alpha * signed_alpha * frame_weights).sum() / frame_weights.sum().clamp_min(_EPS)).item()) * characteristic_radius
    translation_rms = float(torch.sqrt((prism_rate * prism_rate * frame_weights).sum() / frame_weights.sum().clamp_min(_EPS)).item())
    axial_rms = float(torch.sqrt((axial_speed * axial_speed * frame_weights).sum() / frame_weights.sum().clamp_min(_EPS)).item())
    axial_ratio = axial_rms / max(angular_linear_rms, 1e-8)
    screw_improvement = (rmse_r - rmse_s) / max(rmse_r, 1e-8)

    # Score-first model selection.
    #
    # The previous implementation could reject the lowest-error prismatic model
    # solely because the learned unconstrained SE(3) twist contained a small
    # nonzero omega. It then compared screw only against revolute. That can select
    # screw even when prismatic is several times better. Physical tests now only
    # decide whether the more complex screw model is admissible; they never
    # override a lower penalized score from a simpler model.
    rotation_is_negligible = (
        angular_linear_rms <= float(cfg.prismatic_rotation_ratio) * max(translation_rms, 1e-8)
        or float(torch.sqrt((omega_norm * omega_norm).mean()).item()) < float(cfg.omega_activity_threshold)
    )
    best_simple_type = "prismatic" if score_p <= score_r else "revolute"
    best_simple_score = min(score_p, score_r)
    screw_improvement_over_best_simple = (
        (best_simple_score - score_s) / max(best_simple_score, 1e-8)
    )
    meaningful_screw = (
        abs(pitch_per_radian) >= float(cfg.screw_min_pitch)
        and axial_ratio >= float(cfg.screw_min_axial_ratio)
        and screw_improvement_over_best_simple >= float(cfg.screw_min_relative_improvement)
        and score_s < best_simple_score
    )

    admissible_scores = {
        "prismatic": score_p,
        "revolute": score_r,
    }
    if meaningful_screw:
        admissible_scores["screw"] = score_s
    joint_type = min(admissible_scores, key=admissible_scores.get)

    if joint_type == "prismatic":
        axis = prism_axis
        # Translation determines direction but not a unique spatial line. Place
        # the visualization line through the weighted moving-part centroid.
        anchor = prismatic_anchor
        axis_anchor_source = "moving_part_weighted_centroid_prismatic"
    else:
        axis = rot_axis
        anchor = axis_anchor
        axis_anchor_source = "twist_axis_closest_to_motion_center"

    pitch_per_revolution = 2.0 * math.pi * pitch_per_radian
    candidates = [
        CandidateScore("prismatic", rmse_p, norm_p, score_p),
        CandidateScore("revolute", rmse_r, norm_r, score_r),
        CandidateScore("screw", rmse_s, norm_s, score_s),
    ]

    result: Dict[str, Any] = {
        "joint_type": joint_type,
        "axis_direction": axis.tolist(),
        "axis_anchor": anchor.tolist(),
        "axis_point_canonical": anchor.tolist(),
        "axis_anchor_source": axis_anchor_source,
        "motion_center": center.tolist(),
        "prismatic_visualization_anchor": prismatic_anchor.tolist(),
        "pitch_per_radian": pitch_per_radian if joint_type == "screw" else 0.0,
        "pitch_per_revolution": pitch_per_revolution if joint_type == "screw" else 0.0,
        "raw_screw_pitch_per_radian": pitch_per_radian,
        "raw_screw_pitch_per_revolution": pitch_per_revolution,
        "moving_mask_source": moving_source,
        "semantic_foreground_count": int(fg.sum()),
        "semantic_background_count": int((~fg).sum()),
        "moving_part_count": int(moving_mask.sum()),
        "fit_point_count": int(fit_ids.numel()),
        "frame_count": int(omega.shape[0]),
        "active_frame_count": int(active.sum()),
        "characteristic_radius": characteristic_radius,
        "angular_linear_rms": angular_linear_rms,
        "prismatic_translation_rms": translation_rms,
        "screw_axial_rms": axial_rms,
        "screw_axial_to_rotational_ratio": axial_ratio,
        "screw_relative_improvement_over_revolute": screw_improvement,
        "screw_relative_improvement_over_best_simple": screw_improvement_over_best_simple,
        "best_simple_joint_type": best_simple_type,
        "rotation_is_negligible_diagnostic": bool(rotation_is_negligible),
        "foreground_mask_source": foreground_mask_source,
        "selection_rule": "minimum penalized score among prismatic/revolute; screw admitted only if it beats the best simpler model by the configured margin",
        "target_rigid_velocity_rms": target_rms,
        "candidate_scores": [asdict(c) for c in candidates],
        "thresholds": asdict(cfg),
        "semantic_isolation": {
            "non_foreground_used_for_velocity_field": True,
            "non_foreground_used_for_joint_fit": False,
            "non_foreground_used_for_type_selection": False,
            "non_foreground_used_for_axis_or_pitch": False,
        },
        "per_frame": {
            "omega": omega.tolist(),
            "translation_at_motion_center": trans.tolist(),
            "signed_angular_rate": signed_alpha.tolist(),
            "axial_translation_speed": beta.tolist(),
            "prismatic_rate": prism_rate.tolist(),
            "frame_activity": frame_speed.tolist(),
            "axis_points_from_twist": per_frame_axis_points.tolist(),
        },
    }
    return result


def _axis_segment(anchor: np.ndarray, axis: np.ndarray, points: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        length = 1.0
    else:
        extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        length = max(float(extent) * float(scale), 1e-3)
    return anchor - 0.5 * length * axis, anchor + 0.5 * length * axis


def write_final_joint_html(
    tracks: Dict[str, Any],
    result: Dict[str, Any],
    output_html: str | Path,
    max_points: int = 20000,
    axis_length_scale: float = 1.25,
    seed: int = 12345,
) -> Path:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Plotly is required for final joint HTML visualization") from exc

    x = _as_cpu_float(tracks["canonical_positions"])
    fg, _ = _semantic_foreground_mask(tracks)
    moving, _ = _select_moving_mask(tracks, JointEstimationConfig(
        dynamic_threshold=float(result["thresholds"]["dynamic_threshold"]),
        min_moving_points=int(result["thresholds"]["min_moving_points"]),
    ))
    static_fg = fg & ~moving
    background = ~fg

    def sample(mask: torch.Tensor, budget: int, offset: int) -> torch.Tensor:
        ids = torch.where(mask)[0]
        return _subsample_indices(ids, budget, seed + offset)

    total = int(x.shape[0])
    budget = min(max(1, int(max_points)), total)
    n_moving = min(int(moving.sum()), max(1, int(round(budget * 0.45))))
    n_static = min(int(static_fg.sum()), max(1, int(round(budget * 0.35))))
    n_bg = min(int(background.sum()), max(1, budget - n_moving - n_static))
    ids_m = sample(moving, n_moving, 1)
    ids_s = sample(static_fg, n_static, 2)
    ids_b = sample(background, n_bg, 3)

    fig = go.Figure()

    def add_points(ids: torch.Tensor, name: str, size: float, opacity: float, color: str):
        if ids.numel() == 0:
            return
        pts = x[ids].numpy()
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers", name=name,
            marker=dict(size=size, opacity=opacity, color=color),
            text=[f"gaussian={int(i)}" for i in ids.tolist()],
            hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        ))

    add_points(ids_b, "Non-foreground: proposal velocity only", 1.8, 0.22, "#9ca3af")
    add_points(ids_s, "Foreground static component", 2.2, 0.50, "#2563eb")
    add_points(ids_m, "Foreground moving rigid component", 2.8, 0.80, "#dc2626")

    axis = np.asarray(result["axis_direction"], dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    anchor = np.asarray(result["axis_anchor"], dtype=np.float64)
    center = np.asarray(result["motion_center"], dtype=np.float64)
    moving_points = x[ids_m].numpy() if ids_m.numel() else x[fg].numpy()
    a0, a1 = _axis_segment(anchor, axis, moving_points, axis_length_scale)
    fig.add_trace(go.Scatter3d(
        x=[a0[0], a1[0]], y=[a0[1], a1[1]], z=[a0[2], a1[2]],
        mode="lines", name=f"Estimated {result['joint_type']} axis",
        line=dict(width=8, color="#111827"),
        hovertemplate="Estimated joint axis<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[anchor[0]], y=[anchor[1]], z=[anchor[2]], mode="markers",
        name="Axis anchor", marker=dict(size=7, color="#16a34a", symbol="diamond"),
        hovertemplate="Axis anchor<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]], mode="markers",
        name="Learned motion center", marker=dict(size=7, color="#9333ea", symbol="x"),
        hovertemplate="Motion center<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
    ))

    score_text = ", ".join(
        f"{c['joint_type']}={c['normalized_rmse']:.4f}" for c in result["candidate_scores"]
    )
    pitch_text = ""
    if result["joint_type"] == "screw":
        pitch_text = f" · pitch={result['pitch_per_radian']:.6g} units/rad ({result['pitch_per_revolution']:.6g} units/rev)"
    fig.update_layout(
        title=(
            f"Final joint: {result['joint_type'].upper()}{pitch_text}"
            f"<br><sup>Semantic foreground only for joint fit · normalized residuals: {score_text}</sup>"
        ),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, t=75, b=0),
        legend=dict(x=0.01, y=0.99),
    )

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True, auto_open=False)
    return output_html


def write_model_score_html(result: Dict[str, Any], output_html: str | Path) -> Path:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Plotly is required for final joint score visualization") from exc
    names = [c["joint_type"] for c in result["candidate_scores"]]
    values = [c["normalized_rmse"] for c in result["candidate_scores"]]
    fig = go.Figure(go.Bar(x=names, y=values, text=[f"{v:.5f}" for v in values], textposition="auto"))
    fig.update_layout(
        title=f"Final joint model comparison · selected: {result['joint_type']}",
        xaxis_title="Candidate joint model",
        yaxis_title="Normalized rigid-velocity RMSE (lower is better)",
        template="plotly_white",
    )
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True, auto_open=False)
    return output_html


def estimate_and_write_final_joint(
    tracks: Dict[str, Any],
    output_dir: str | Path,
    config: Optional[JointEstimationConfig] = None,
) -> Dict[str, Any]:
    cfg = config or JointEstimationConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = estimate_final_joint(tracks, cfg)
    json_path = output_dir / "final_joint_estimation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_final_joint_html(
        tracks,
        result,
        output_dir / "final_joint_estimation.html",
        max_points=cfg.max_html_points,
        axis_length_scale=cfg.axis_length_scale,
        seed=cfg.random_seed,
    )
    write_model_score_html(result, output_dir / "final_joint_model_scores.html")
    return result


def estimate_and_write_final_joint_from_file(
    tracks_path: str | Path,
    output_dir: str | Path,
    config: Optional[JointEstimationConfig] = None,
) -> Dict[str, Any]:
    tracks = torch.load(str(tracks_path), map_location="cpu")
    if not isinstance(tracks, dict):
        raise TypeError(f"Expected a dict in {tracks_path}, got {type(tracks)!r}")
    return estimate_and_write_final_joint(tracks, output_dir, config)
