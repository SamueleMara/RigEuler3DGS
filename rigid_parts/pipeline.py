from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from gaussian_renderer import render as render_gaussian
from .endpoint import (
    compute_counterfactual_endpoint_unary,
    load_ply_xyz,
)
from .labels import RigidPartLabels
from .losses import (
    build_knn_graph,
    build_sequence_state,
    com_transport_loss,
    distance_rate_loss,
    graph_label_coherence_loss,
    inertia_invariant_loss,
    inertia_transport_loss,
    occupancy_loss,
    pairwise_distance_loss,
    raw_graph_affinity,
    trajectory_unary,
    trajectory_unary_loss,
    twist_unary,
    twist_unary_loss,
)
from utils.rigid_transform_utils import (
    apply_rotation_to_gaussian_quaternions,
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    se3_exp_spatial,
)
from utils.temporal_visibility_pretraining import _atomic_torch_save


@dataclass
class ObservedMotionState:
    velocities: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor


@dataclass
class RawMotionState:
    tracks: torch.Tensor
    rotations: torch.Tensor
    velocities: torch.Tensor
    interval_velocities: torch.Tensor
    interval_omega: torch.Tensor
    interval_b: torch.Tensor


@dataclass
class RigidPartState:
    probabilities: torch.Tensor
    labels: torch.Tensor
    confidence: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor
    omega: torch.Tensor
    b: torch.Tensor
    com: torch.Tensor
    inertia: torch.Tensor


@dataclass
class RigidProjectionState:
    tracks: torch.Tensor
    velocities: torch.Tensor
    rotations: torch.Tensor


def _loss_fractions(weighted: dict[str, torch.Tensor]) -> dict[str, float]:
    detached = {key: float(value.detach().cpu()) for key, value in weighted.items()}
    denom = sum(abs(value) for value in detached.values()) + 1.0e-12
    return {f"frac_{key}": abs(value) / denom for key, value in detached.items()}


def _weighted_sum(raw: dict[str, torch.Tensor], weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weighted = {key: float(weights.get(key, 0.0)) * value for key, value in raw.items()}
    if weighted:
        total = sum(weighted.values())
    else:
        total = torch.tensor(0.0)
    return total, weighted


def _integrate_local_killing_twist(
    xyz0: torch.Tensor,
    rot0: torch.Tensor,
    interval_omega: torch.Tensor,
    interval_b: torch.Tensor,
    times: torch.Tensor,
) -> RawMotionState:
    dt = torch.diff(times.reshape(-1)).clamp_min(1.0e-8)
    xyz_frames = [xyz0]
    rot_frames = [normalize_quaternion(rot0)]
    interval_velocity = []
    for interval in range(int(interval_b.shape[0])):
        source_xyz = xyz_frames[-1]
        omega = interval_omega[interval]
        linear = interval_b[interval]
        interval_velocity.append(torch.cross(omega, source_xyz, dim=-1) + linear)
        r_inc, p_inc, q_inc = se3_exp_spatial(omega, linear, dt[interval])
        xyz_frames.append((r_inc @ source_xyz[..., None])[..., 0] + p_inc)
        rot_frames.append(normalize_quaternion(quaternion_multiply(q_inc, rot_frames[-1])))
    tracks = torch.stack(xyz_frames)
    rotations = torch.stack(rot_frames)
    interval_velocity = (
        torch.stack(interval_velocity)
        if interval_velocity
        else torch.zeros((0, xyz0.shape[0], 3), dtype=xyz0.dtype, device=xyz0.device)
    )
    frame_velocity = (
        torch.cat((interval_velocity, interval_velocity[-1:]), dim=0)
        if interval_velocity.numel()
        else torch.zeros((times.numel(), xyz0.shape[0], 3), dtype=xyz0.dtype, device=xyz0.device)
    )
    return RawMotionState(
        tracks=tracks,
        rotations=rotations,
        velocities=frame_velocity,
        interval_velocities=interval_velocity,
        interval_omega=interval_omega,
        interval_b=interval_b,
    )


def _observed_velocity_state(stage1: dict, raw_times: torch.Tensor, raw_velocity: torch.Tensor) -> ObservedMotionState:
    observed = stage1.get("observed_velocities")
    valid = stage1.get("observed_velocity_valid")
    confidence = stage1.get("observed_velocity_confidence")
    if observed is None:
        # Backward-compatible fallback for runs produced before measured and
        # confidence-weighted velocities were split.  This is a measurement
        # bootstrap only; the closed loop still optimizes a separate raw field.
        observed = raw_velocity[:-1] if raw_velocity.shape[0] == raw_times.numel() else raw_velocity
    observed = torch.as_tensor(observed).detach().float()
    if observed.shape[0] == raw_times.numel():
        observed = observed[:-1]
    if valid is None:
        valid = torch.ones(observed.shape[:2], dtype=torch.bool)
    if confidence is None:
        confidence = torch.ones(observed.shape[:2], dtype=observed.dtype)
    valid = torch.as_tensor(valid).detach().bool()
    confidence = torch.as_tensor(confidence).detach().float().clamp(0.0, 1.0)
    if valid.shape != observed.shape[:2]:
        valid = torch.ones(observed.shape[:2], dtype=torch.bool)
    if confidence.shape != observed.shape[:2]:
        confidence = torch.ones(observed.shape[:2], dtype=observed.dtype)
    return ObservedMotionState(velocities=observed, confidence=confidence, valid=valid)


def _observed_velocity_loss(interval_velocity: torch.Tensor, observed: ObservedMotionState) -> torch.Tensor:
    target = observed.velocities.to(interval_velocity)
    valid = observed.valid.to(device=interval_velocity.device)
    weight = observed.confidence.to(interval_velocity).clamp(0.0, 1.0)
    if not bool(valid.any()):
        return interval_velocity.new_tensor(0.0)
    err = (interval_velocity - target).square().sum(-1)
    w = torch.where(valid, weight, torch.zeros_like(weight))
    return (err * w).sum() / w.sum().clamp_min(1.0e-8)


def _raw_temporal_loss(interval_velocity: torch.Tensor) -> torch.Tensor:
    if int(interval_velocity.shape[0]) <= 1:
        return interval_velocity.new_tensor(0.0)
    return (interval_velocity[1:] - interval_velocity[:-1]).square().mean()


def _raw_twist_temporal_loss(interval_omega: torch.Tensor, interval_b: torch.Tensor) -> torch.Tensor:
    if int(interval_b.shape[0]) <= 1:
        return interval_b.new_tensor(0.0)
    return (
        (interval_omega[1:] - interval_omega[:-1]).square().mean()
        + (interval_b[1:] - interval_b[:-1]).square().mean()
    )


def _raw_killing_loss(
    tracks: torch.Tensor,
    velocities: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
) -> torch.Tensor:
    src = edge_src.to(device=tracks.device).long()
    dst = edge_dst.to(device=tracks.device).long()
    if src.numel() == 0:
        return tracks.new_tensor(0.0)
    losses = []
    for frame in range(int(tracks.shape[0])):
        delta_x = tracks[frame, src] - tracks[frame, dst]
        delta_v = velocities[frame, src] - velocities[frame, dst]
        losses.append(((delta_x * delta_v).sum(-1).square() / delta_x.square().sum(-1).clamp_min(1.0e-8)))
    return torch.stack(losses).mean()


def _rigid_feedback_loss(raw: RawMotionState, previous: dict | None, prob: torch.Tensor | None) -> torch.Tensor:
    if previous is None:
        return raw.tracks.new_tensor(0.0)
    tracks = torch.as_tensor(previous["positions"], dtype=raw.tracks.dtype, device=raw.tracks.device)
    velocities = torch.as_tensor(previous["velocities"], dtype=raw.velocities.dtype, device=raw.velocities.device)
    if tracks.shape != raw.tracks.shape or velocities.shape != raw.velocities.shape:
        return raw.tracks.new_tensor(0.0)
    if prob is None:
        weight = torch.ones(raw.tracks.shape[1], dtype=raw.tracks.dtype, device=raw.tracks.device)
    else:
        p = torch.as_tensor(prob, dtype=raw.tracks.dtype, device=raw.tracks.device)
        weight = p.max(-1).values * (1.0 - p[:, 0]).clamp(0.0, 1.0)
    denom = weight.sum().clamp_min(1.0e-8)
    pos_loss = ((raw.tracks - tracks).square().sum(-1) * weight[None]).sum() / (denom * raw.tracks.shape[0])
    vel_loss = ((raw.velocities - velocities).square().sum(-1) * weight[None]).sum() / (denom * raw.velocities.shape[0])
    return pos_loss + vel_loss


def _graph_repulsive_loss(
    prob: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    affinity: torch.Tensor,
) -> torch.Tensor:
    src = edge_src.to(device=prob.device).long()
    dst = edge_dst.to(device=prob.device).long()
    boundary = (1.0 - torch.as_tensor(affinity, dtype=prob.dtype, device=prob.device)).clamp(0.0, 1.0)
    same_label = (prob[src] * prob[dst]).sum(-1)
    uniform_same = 1.0 / max(1, int(prob.shape[1]))
    excess_same_confidence = (same_label - float(uniform_same)).clamp_min(0.0)
    return (boundary * excess_same_confidence).sum() / boundary.sum().clamp_min(1.0e-8)


def _dynamic_prior_payload(stage1: dict, n: int) -> dict[str, torch.Tensor]:
    prior = stage1.get("dynamic_prior_bootstrap")
    if isinstance(prior, dict) and torch.is_tensor(prior.get("probability")):
        prob = torch.as_tensor(prior["probability"]).detach().float().reshape(-1).clamp(0.0, 1.0)
        conf = torch.as_tensor(prior.get("confidence", torch.ones_like(prob))).detach().float().reshape(-1).clamp(0.0, 1.0)
        valid = torch.as_tensor(prior.get("valid", torch.ones_like(prob, dtype=torch.bool))).detach().bool().reshape(-1)
    else:
        init = torch.as_tensor(stage1["responsibilities"]).detach().float()
        prob = init[:, 1:].sum(-1).clamp(0.0, 1.0)
        conf = torch.full_like(prob, 0.25)
        valid = torch.ones_like(prob, dtype=torch.bool)
    if prob.numel() != int(n):
        prob = torch.full((int(n),), 0.5, dtype=torch.float32)
        conf = torch.zeros_like(prob)
        valid = torch.zeros_like(prob, dtype=torch.bool)
    return {"probability": prob, "confidence": conf, "valid": valid}


def _dynamic_prior_loss(prob: torch.Tensor, prior: dict[str, torch.Tensor]) -> torch.Tensor:
    pred = (1.0 - prob[:, 0]).clamp(1.0e-6, 1.0 - 1.0e-6)
    target = prior["probability"].to(pred).clamp(0.0, 1.0)
    valid = prior["valid"].to(device=pred.device)
    weight = prior["confidence"].to(pred).clamp(0.0, 1.0)
    w = torch.where(valid, weight, torch.zeros_like(weight))
    if float(w.sum().detach().cpu()) <= 0.0:
        return pred.new_tensor(0.0)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    return (bce * w).sum() / w.sum().clamp_min(1.0e-8)


def _moving_complexity_loss(prob: torch.Tensor) -> torch.Tensor:
    return (1.0 - prob[:, 0]).mean()


def _resolve_final_ply(path: str) -> Path | None:
    if not str(path or "").strip():
        return None
    p = Path(path).expanduser()
    if p.is_file():
        return p
    candidates: list[tuple[int, Path]] = []
    for ply in p.glob("point_cloud/iteration_*/point_cloud.ply"):
        try:
            iteration = int(ply.parent.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        candidates.append((iteration, ply))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _selected_indices(frame_count: int, max_frames: int) -> torch.Tensor:
    count = int(frame_count)
    limit = int(max_frames)
    if limit <= 0 or count <= limit:
        return torch.arange(count)
    return torch.linspace(0, count - 1, max(2, limit)).round().long().unique()


def _temperature(args, step: int, steps: int) -> float:
    start = float(getattr(args, "rigid_part_label_temperature", 1.0))
    end = max(0.4, float(getattr(args, "rigid_part_label_temperature_final", 0.7)))
    if int(steps) <= 1:
        return end
    alpha = float(step) / float(int(steps) - 1)
    return start + alpha * (end - start)


def _soften(prob: torch.Tensor, floor: float = 0.05) -> torch.Tensor:
    p = torch.as_tensor(prob).detach().float().clamp(float(floor), 1.0 - float(floor))
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _initial_prob(stage1: dict, num_moving_parts: int) -> torch.Tensor:
    """Weak K+1 label initialization. No class is frozen."""
    k = int(num_moving_parts)
    prior = stage1.get("dynamic_prior_bootstrap")
    if isinstance(prior, dict) and torch.is_tensor(prior.get("probability")):
        p_dyn = torch.as_tensor(prior["probability"]).detach().float().clamp(0.05, 0.95)
        prob = torch.zeros((p_dyn.numel(), k + 1), dtype=torch.float32)
        prob[:, 0] = 1.0 - p_dyn
        if k == 1:
            prob[:, 1] = p_dyn
        else:
            base = torch.as_tensor(stage1["responsibilities"]).detach().float()
            moving = base[:, 1:].clamp_min(1.0e-8)
            moving = moving / moving.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            prob[:, 1:] = p_dyn[:, None] * moving[:, :k]
        return _soften(prob)
    return _soften(torch.as_tensor(stage1["responsibilities"]).detach().float())


def _jsonable(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return float(value) if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _grad_norm(loss: torch.Tensor, logits: torch.Tensor) -> float:
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return 0.0
    grad = torch.autograd.grad(loss, logits, retain_graph=True, allow_unused=True)[0]
    if grad is None:
        return 0.0
    return float(torch.linalg.norm(grad.detach()).cpu())


def _grad_norm_many(loss: torch.Tensor, params: list[torch.Tensor]) -> float:
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().square().sum().cpu())
    return total ** 0.5


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _write_label_html(path: Path, xyz: torch.Tensor, labels: torch.Tensor, confidence: torch.Tensor, max_points: int) -> None:
    ids = torch.arange(xyz.shape[0])
    if xyz.shape[0] > int(max_points):
        ids = torch.linspace(0, xyz.shape[0] - 1, int(max_points)).round().long().unique()
    payload = {
        "xyz": xyz[ids].detach().cpu().float().tolist(),
        "labels": labels[ids].detach().cpu().long().tolist(),
        "confidence": confidence[ids].detach().cpu().float().tolist(),
    }
    path.write_text(
        """<!doctype html><meta charset="utf-8"><title>Corrected rigid-part labels</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
const p = """
        + json.dumps(payload)
        + """;
const xyz = p.xyz;
Plotly.newPlot('plot', [
 {type:'scatter3d',mode:'markers',name:'label',
  x:xyz.map(v=>v[0]), y:xyz.map(v=>v[1]), z:xyz.map(v=>v[2]),
  marker:{size:2,color:p.labels,colorscale:'Turbo',showscale:true}},
 {type:'scatter3d',mode:'markers',name:'confidence',visible:false,
  x:xyz.map(v=>v[0]), y:xyz.map(v=>v[1]), z:xyz.map(v=>v[2]),
  marker:{size:2,color:p.confidence,colorscale:'Viridis',cmin:0,cmax:1,showscale:true}}
], {scene:{aspectmode:'data'}, margin:{l:0,r:0,t:30,b:0}});
</script>""",
        encoding="utf-8",
    )


def _frame_id(camera) -> int | None:
    if hasattr(camera, "frame_id"):
        try:
            return int(getattr(camera, "frame_id"))
        except Exception:
            pass
    name = str(getattr(camera, "image_name", ""))
    import re

    match = re.search(r"frame[_-]?(\d+)", name)
    if match is not None:
        return int(match.group(1))
    return None


def _selected_render_views(scene, stage1: dict, max_frames: int, max_cameras: int) -> list[tuple[int, object]]:
    cameras = list(scene.getTrainCameras())
    by_frame: dict[int, list[object]] = {}
    for camera in cameras:
        fid = _frame_id(camera)
        if fid is None:
            continue
        by_frame.setdefault(fid, []).append(camera)
    frame_ids = [int(v) for v in stage1.get("frame_ids", list(range(int(torch.as_tensor(stage1["raw_tracks"]["positions"]).shape[0]))))]
    selected = _selected_indices(len(frame_ids), int(max_frames)).tolist()
    views: list[tuple[int, object]] = []
    for raw_index in selected:
        fid = frame_ids[int(raw_index)]
        candidates = sorted(
            by_frame.get(fid, []),
            key=lambda cam: (str(getattr(cam, "dynamic_camera_id", "")), str(getattr(cam, "image_name", ""))),
        )
        for camera in candidates[: max(1, int(max_cameras))]:
            views.append((int(raw_index), camera))
    return views


def _save_tensor_png(path: Path, image: torch.Tensor) -> None:
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return
    value = image.detach().float().cpu()
    if value.ndim == 3 and value.shape[0] in (1, 3):
        value = value.permute(1, 2, 0)
    if value.ndim == 2:
        array = (value.clamp(0, 1).numpy() * 255.0).astype("uint8")
        Image.fromarray(array).save(path)
        return
    if value.shape[-1] == 1:
        value = value[..., 0]
        array = (value.clamp(0, 1).numpy() * 255.0).astype("uint8")
        Image.fromarray(array).save(path)
        return
    array = (value[..., :3].clamp(0, 1).numpy() * 255.0).astype("uint8")
    Image.fromarray(array).save(path)


def _render_state_loss(
    *,
    views: list[tuple[int, object]],
    gaussians,
    pipe,
    background: torch.Tensor,
    dataset,
    args,
    positions: torch.Tensor,
    rotations: torch.Tensor | None,
    debug_dir: Path | None = None,
    prefix: str = "",
) -> tuple[torch.Tensor, dict[str, float]]:
    if not views:
        zero = positions.new_tensor(0.0)
        return zero, {"rgb": 0.0, "depth": 0.0, "views": 0}
    total_rgb = positions.new_tensor(0.0)
    total_depth = positions.new_tensor(0.0)
    depth_count = 0
    save_budget = 4
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    for view_index, (frame_index, camera) in enumerate(views):
        mean_override = positions[int(frame_index)]
        rot_override = None if rotations is None else rotations[int(frame_index)]
        package = render_gaussian(
            camera,
            gaussians,
            pipe,
            background,
            use_trained_exp=bool(getattr(dataset, "train_test_exp", False)),
            separate_sh=False,
            means3D_override=mean_override,
            rotations_override=rot_override,
        )
        image = package["render"]
        if getattr(camera, "alpha_mask", None) is not None:
            image = image * camera.alpha_mask.to(device=image.device)
        target = camera.original_image.to(device=image.device).clamp(0.0, 1.0)
        rgb = (image - target).abs().mean()
        total_rgb = total_rgb + rgb
        if bool(getattr(args, "dynamic_use_depth", False)) and bool(getattr(camera, "depth_reliable", False)):
            mask = camera.depth_mask.to(device=image.device)
            depth = package["depth"]
            target_depth = camera.invdepthmap.to(device=image.device)
            total_depth = total_depth + ((depth - target_depth).abs() * mask).mean()
            depth_count += 1
        if debug_dir is not None and view_index < save_budget:
            stem = f"{prefix}_{view_index:02d}_frame{int(frame_index):04d}"
            _save_tensor_png(debug_dir / f"{stem}_render.png", image)
            _save_tensor_png(debug_dir / f"{stem}_target.png", target)
            _save_tensor_png(debug_dir / f"{stem}_absdiff.png", (image - target).abs().mean(0, keepdim=True))
    rgb_loss = total_rgb / max(1, len(views))
    depth_loss = total_depth / max(1, depth_count)
    total = (
        float(getattr(args, "lambda_raw_render_rgb", 1.0)) * rgb_loss
        + float(getattr(args, "lambda_raw_render_depth", 0.1)) * depth_loss
    )
    return total, {"rgb": float(rgb_loss.detach().cpu()), "depth": float(depth_loss.detach().cpu()), "views": len(views)}


def _write_loss_curves_html(path: Path, rows: list[dict[str, object]]) -> None:
    numeric_keys = sorted(
        key
        for key in {k for row in rows for k in row}
        if key not in {"phase"} and any(isinstance(row.get(key), (int, float)) for row in rows)
    )
    payload = {"rows": rows, "keys": numeric_keys}
    path.write_text(
        """<!doctype html><meta charset="utf-8"><title>Exact closed-loop rigid-part losses</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
const p = """
        + json.dumps(_jsonable(payload))
        + """;
const traces = p.keys.filter(k => k.startsWith('loss') || k.startsWith('weighted') || k.startsWith('grad') || k.includes('render')).map(k => ({
  x: p.rows.map((_, i) => i),
  y: p.rows.map(r => r[k] ?? null),
  mode: 'lines',
  name: k
}));
Plotly.newPlot('plot', traces, {margin:{l:50,r:20,t:30,b:40}, yaxis:{type:'log', autorange:true}});
</script>""",
        encoding="utf-8",
    )


def _phase_delta(rows: list[dict[str, object]], key: str) -> tuple[float | None, float | None, float | None]:
    vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not vals:
        return None, None, None
    return vals[0], vals[-1], vals[-1] - vals[0]


def _write_compact_debug(result_dir: Path, history: list[dict[str, object]], outer_summaries: list[dict[str, object]]) -> None:
    raw_rows = [row for row in history if row.get("phase") == "raw_motion"]
    label_rows = [row for row in history if row.get("phase") == "labels"]
    compact = []
    for row in raw_rows:
        compact.append({
            "phase": "raw",
            "outer": row.get("outer"),
            "step": row.get("step"),
            "total": row.get("loss_total"),
            "render": row.get("loss_render"),
            "observed_velocity": row.get("loss_observed_velocity"),
            "killing": row.get("loss_killing"),
            "temporal": row.get("loss_temporal"),
            "rigid_feedback": row.get("loss_rigid_feedback"),
            "feedback_enabled": row.get("rigid_feedback_enabled"),
            "grad_omega": row.get("grad_motion_total_omega"),
            "grad_b": row.get("grad_motion_total_b"),
        })
    for row in label_rows:
        compact.append({
            "phase": "labels",
            "outer": row.get("outer"),
            "step": row.get("step"),
            "total": row.get("loss_total"),
            "trajectory": row.get("loss_trajectory"),
            "twist": row.get("loss_twist"),
            "graph_attr": row.get("loss_graph_attractive"),
            "graph_rep": row.get("loss_graph_repulsive"),
            "prior": row.get("loss_prior"),
            "moving_complexity": row.get("loss_moving_complexity"),
            "dyn_prob": row.get("mean_dynamic_probability"),
            "confidence": row.get("mean_label_confidence"),
        })
    _write_csv(result_dir / "debug_compact.csv", compact)
    _write_csv(result_dir / "outer_compact.csv", outer_summaries)

    warnings = []
    if outer_summaries:
        worse = [row for row in outer_summaries if float(row.get("rigid_minus_raw_render", 0.0)) > 0.0]
        if worse:
            has_gate_key = any("rigid_feedback_next_enabled" in row for row in outer_summaries)
            action = "feedback was gated off" if has_gate_key else "feedback would now be gated off"
            warnings.append(f"Rigid projection rendered worse than raw in {len(worse)}/{len(outer_summaries)} outer iterations; {action} for those iterations.")
        proj0, proj1, projd = _phase_delta(outer_summaries, "projection_position_error_mean")
        if projd is not None and projd > 0.0:
            warnings.append(f"Rigid projection drift increased: projection_position_error_mean {proj0:.6g} -> {proj1:.6g}.")
    raw_total = _phase_delta(raw_rows, "loss_total")
    label_total = _phase_delta(label_rows, "loss_total")
    if len(raw_rows) < 2:
        warnings.append(f"Raw-motion trend has only {len(raw_rows)} row(s); smoke runs cannot show loss evolution.")
    if len(label_rows) < 2:
        warnings.append(f"Label trend has only {len(label_rows)} row(s); smoke runs cannot show loss evolution.")
    rep_frac = [float(row.get("frac_graph_repulsive", 0.0)) for row in label_rows if isinstance(row.get("frac_graph_repulsive"), (int, float))]
    if rep_frac and max(rep_frac) > 0.5:
        warnings.append(f"Graph repulsion dominated label loss before/after normalization; max weighted fraction={max(rep_frac):.3f}.")

    lines = [
        "# Exact closed-loop rigid-part debug",
        "",
        "## Verdict",
    ]
    if warnings:
        lines.extend(f"- WARNING: {item}" for item in warnings)
    else:
        lines.append("- No automatic red flags from compact diagnostics.")
    lines.extend([
        "",
        "## Loss deltas",
        "",
        "| phase | metric | first | last | delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for phase, rows, keys in (
        ("raw", raw_rows, ("loss_total", "loss_render", "loss_observed_velocity", "loss_killing", "loss_temporal", "loss_rigid_feedback")),
        ("labels", label_rows, ("loss_total", "loss_trajectory", "loss_twist", "loss_graph_attractive", "loss_graph_repulsive", "loss_prior", "loss_moving_complexity")),
        ("outer", outer_summaries, ("raw_render_rgb", "rigid_render_rgb", "rigid_minus_raw_render", "projection_position_error_mean", "mean_dynamic_probability", "mean_label_confidence")),
    ):
        for key in keys:
            first, last, delta = _phase_delta(rows, key)
            if first is not None:
                lines.append(f"| {phase} | {key} | {first:.6g} | {last:.6g} | {delta:.6g} |")
    lines.extend([
        "",
        "## Files",
        "",
        "- `debug_compact.csv`: readable per-step losses.",
        "- `outer_compact.csv`: one row per outer iteration.",
        "- `debug_dashboard.html`: grouped plots; no all-column soup.",
        "- `history.json`: full raw dump if needed.",
    ])
    (result_dir / "debug_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "raw": raw_rows,
        "labels": label_rows,
        "outer": outer_summaries,
    }
    (result_dir / "debug_dashboard.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Exact closed-loop debug dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<h2>Raw motion losses</h2><div id="raw" style="height:32vh"></div>
<h2>Label losses</h2><div id="labels" style="height:32vh"></div>
<h2>Outer projection/render</h2><div id="outer" style="height:32vh"></div>
<script>
const p = """
        + json.dumps(_jsonable(payload))
        + """;
function trace(rows, key, name) { return {x: rows.map((_,i)=>i), y: rows.map(r=>r[key] ?? null), mode:'lines+markers', name}; }
Plotly.newPlot('raw', [
 trace(p.raw, 'loss_render', 'render'),
 trace(p.raw, 'loss_observed_velocity', 'observed velocity'),
 trace(p.raw, 'loss_killing', 'killing'),
 trace(p.raw, 'loss_temporal', 'temporal'),
 trace(p.raw, 'loss_rigid_feedback', 'rigid feedback')
], {margin:{l:50,r:20,t:10,b:35}});
Plotly.newPlot('labels', [
 trace(p.labels, 'loss_trajectory', 'trajectory'),
 trace(p.labels, 'loss_twist', 'twist'),
 trace(p.labels, 'loss_graph_attractive', 'graph attractive'),
 trace(p.labels, 'loss_graph_repulsive', 'graph repulsive'),
 trace(p.labels, 'loss_prior', 'prior'),
 trace(p.labels, 'loss_moving_complexity', 'moving complexity')
], {margin:{l:50,r:20,t:10,b:35}});
Plotly.newPlot('outer', [
 trace(p.outer, 'raw_render_rgb', 'raw rgb'),
 trace(p.outer, 'rigid_render_rgb', 'rigid rgb'),
 trace(p.outer, 'rigid_minus_raw_render', 'rigid - raw render'),
 trace(p.outer, 'projection_position_error_mean', 'projection position error'),
 trace(p.outer, 'mean_dynamic_probability', 'mean dynamic probability')
], {margin:{l:50,r:20,t:10,b:35}});
</script>""",
        encoding="utf-8",
    )


def _rigid_projected_tracks(
    canonical_xyz: torch.Tensor,
    canonical_rotation: torch.Tensor,
    state,
    labels: torch.Tensor,
    raw_times: torch.Tensor,
) -> dict:
    device = canonical_xyz.device
    labels = torch.as_tensor(labels, device=device).long()
    eye_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=canonical_xyz.dtype, device=device)
    eye_r = torch.eye(3, dtype=canonical_xyz.dtype, device=device)
    zero_t = torch.zeros(3, dtype=canonical_xyz.dtype, device=device)
    frame_count = int(raw_times.numel())
    part_count = int(state.part_prob.shape[1])
    dt = torch.diff(raw_times.to(device=device, dtype=canonical_xyz.dtype).reshape(-1)).clamp_min(1.0e-8)

    current_r = eye_r[None].expand(part_count, -1, -1).clone()
    current_t = zero_t[None].expand(part_count, -1).clone()
    current_q = eye_q[None].expand(part_count, -1).clone()
    part_r_frames = [current_r.clone()]
    part_t_frames_full = [current_t.clone()]
    part_q_frames_full = [current_q.clone()]
    for interval in range(max(0, frame_count - 1)):
        omega = state.omega[interval].to(device=device, dtype=canonical_xyz.dtype)
        linear = state.b[interval].to(device=device, dtype=canonical_xyz.dtype)
        r_inc, p_inc, q_inc = se3_exp_spatial(omega, linear, dt[interval])
        if part_count:
            r_inc = r_inc.clone()
            p_inc = p_inc.clone()
            q_inc = q_inc.clone()
            r_inc[0] = eye_r
            p_inc[0] = zero_t
            q_inc[0] = eye_q
        current_r = r_inc @ current_r
        current_t = (r_inc @ current_t[..., None])[..., 0] + p_inc
        current_q = normalize_quaternion(quaternion_multiply(q_inc, current_q))
        part_r_frames.append(current_r.clone())
        part_t_frames_full.append(current_t.clone())
        part_q_frames_full.append(current_q.clone())
    part_r = torch.stack(part_r_frames)
    part_t = torch.stack(part_t_frames_full)
    part_q = torch.stack(part_q_frames_full)

    xyz_frames, rot_frames, vel_frames = [], [], []
    part_q_frames, part_t_frames = [], []
    for frame in range(frame_count):
        candidates, velocity_candidates, rotation_candidates = [], [], []
        twist_frame = min(frame, int(state.omega.shape[0]) - 1)
        for part in range(part_count):
            r = part_r[frame, part]
            t = part_t[frame, part]
            moved = canonical_xyz @ r.transpose(0, 1) + t
            q = part_q[frame, part]
            candidates.append(moved)
            if part == 0:
                velocity_candidates.append(torch.zeros_like(canonical_xyz))
            else:
                velocity_candidates.append(torch.cross(state.omega[twist_frame, part][None].expand_as(moved), moved, dim=-1) + state.b[twist_frame, part])
            rotation_candidates.append(apply_rotation_to_gaussian_quaternions(canonical_rotation, q))
        stacked_xyz = torch.stack(candidates, dim=1)
        stacked_vel = torch.stack(velocity_candidates, dim=1)
        stacked_rot = torch.stack(rotation_candidates, dim=1)
        pick = labels[:, None, None].expand(-1, 1, 3)
        xyz_frames.append(stacked_xyz.gather(1, pick).squeeze(1).detach().cpu())
        vel_frames.append(stacked_vel.gather(1, pick).squeeze(1).detach().cpu())
        rot_frames.append(stacked_rot.gather(1, labels[:, None, None].expand(-1, 1, 4)).squeeze(1).detach().cpu())
        if part_count > 1:
            part_q_frames.append(part_q[frame, 1:].detach().cpu())
            part_t_frames.append(part_t[frame, 1:].detach().cpu())
        else:
            part_q_frames.append(eye_q[None].detach().cpu()[:0])
            part_t_frames.append(torch.empty((0, 3), dtype=canonical_xyz.dtype).cpu())
    return {
        "times": raw_times.detach().cpu().float(),
        "positions": torch.stack(xyz_frames),
        "rotations": torch.stack(rot_frames),
        "velocities": torch.stack(vel_frames),
        "canonical_xyz": canonical_xyz.detach().cpu(),
        "part_labels": labels.detach().cpu(),
        "part_rotation_quaternions": torch.stack(part_q_frames),
        "part_translations": torch.stack(part_t_frames),
        "schema_version": 4,
        "source_of_truth": "spatial_twist_se3_exponential_rigid_projection",
        "se3_integration_convention": "spatial/world-frame left action: T_next = Exp(dt xi_hat) @ T, x_dot = omega cross x + b",
    }


def run_corrected_rigid_part_discovery_from_tracks(
    *,
    canonical_xyz: torch.Tensor,
    canonical_rotation: torch.Tensor,
    raw_tracks: torch.Tensor,
    raw_velocities: torch.Tensor,
    raw_times: torch.Tensor,
    init_prob: torch.Tensor,
    args,
    output: Path,
    final_xyz: torch.Tensor | None = None,
    log_fn=None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xyz0 = torch.as_tensor(canonical_xyz).detach().float().to(device)
    rot0 = torch.as_tensor(canonical_rotation).detach().float().to(device)
    all_pos = torch.as_tensor(raw_tracks).detach().float()
    all_vel = torch.as_tensor(raw_velocities).detach().float()
    all_times = torch.as_tensor(raw_times).detach().float()
    select = _selected_indices(int(all_pos.shape[0]), int(getattr(args, "rigid_part_max_frames", 20)))
    pos = all_pos.index_select(0, select).to(device)
    vel = all_vel.index_select(0, select).to(device)
    times = all_times.index_select(0, select).to(device)
    labels = RigidPartLabels(
        xyz0.shape[0],
        int(torch.as_tensor(init_prob).shape[1]) - 1,
        init_logits=torch.log(_soften(init_prob).clamp_min(1.0e-8)),
        temperature=float(getattr(args, "rigid_part_label_temperature", 1.0)),
    ).to(device)
    before_prob = labels.probabilities(float(getattr(args, "rigid_part_label_temperature", 1.0))).detach()
    before_hard = before_prob.argmax(-1).detach().cpu()
    edge_src, edge_dst, edge_d0 = build_knn_graph(xyz0.detach().cpu(), k=int(getattr(args, "rigid_part_knn", 16)))
    edge_src, edge_dst, edge_d0 = edge_src.to(device), edge_dst.to(device), edge_d0.to(device)
    affinity = raw_graph_affinity(pos.detach(), vel.detach(), edge_src, edge_dst, edge_d0).detach()
    final = None if final_xyz is None else torch.as_tensor(final_xyz).detach().float().to(device)

    optimizer = torch.optim.Adam(labels.parameters(), lr=float(getattr(args, "rigid_part_lr", 5.0e-2)))
    steps = max(1, int(getattr(args, "rigid_part_refine_iters", 25)))
    history: list[dict[str, object]] = []
    progress = tqdm(range(steps), desc="Corrected raw-track rigid discovery", unit="iter")
    previous_hard = before_hard.to(device)
    previous_prob = before_prob.to(device)
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        temp = _temperature(args, step, steps)
        prob = labels.probabilities(temp)
        state = build_sequence_state(xyz0, pos, vel, prob, times)
        traj_e = trajectory_unary(xyz0, pos, state)
        twist_e = twist_unary(pos, vel, state)
        endpoint_e = (
            compute_counterfactual_endpoint_unary(
                xyz0,
                state.rotation[-1],
                state.translation[-1],
                final,
                chunk_size=int(getattr(args, "rigid_part_endpoint_chunk", 2048)),
            )
            if final is not None
            else torch.zeros_like(traj_e)
        )
        losses = {
            "trajectory": trajectory_unary_loss(traj_e, prob),
            "twist": twist_unary_loss(twist_e, prob),
            "graph": graph_label_coherence_loss(prob, edge_src, edge_dst, affinity),
            "pair": pairwise_distance_loss(pos, edge_src, edge_dst, edge_d0, prob),
            "distance_rate": distance_rate_loss(pos, vel, edge_src, edge_dst, prob),
            "com": com_transport_loss(state),
            "inertia": inertia_transport_loss(state),
            "inertia_invariants": inertia_invariant_loss(state),
            "endpoint": (prob * endpoint_e).sum() / prob.sum().clamp_min(1.0e-8),
            "occupancy": occupancy_loss(prob, float(getattr(args, "rigid_part_min_fraction", 0.01))),
            "entropy": RigidPartLabels.entropy(prob),
        }
        total = (
            float(getattr(args, "lambda_rigid_trajectory", 1.0)) * losses["trajectory"]
            + float(getattr(args, "lambda_rigid_twist", 1.0)) * losses["twist"]
            + float(getattr(args, "lambda_rigid_graph", getattr(args, "lambda_rigid_pair", 1.0))) * losses["graph"]
            + float(getattr(args, "lambda_rigid_pair", 1.0)) * losses["pair"]
            + float(getattr(args, "lambda_rigid_distance_rate", 1.0)) * losses["distance_rate"]
            + float(getattr(args, "lambda_rigid_com", 0.1)) * losses["com"]
            + float(getattr(args, "lambda_rigid_inertia", 0.1)) * losses["inertia"]
            + float(getattr(args, "lambda_rigid_inertia_invariants", 0.05)) * losses["inertia_invariants"]
            + float(getattr(args, "lambda_endpoint_geometry", 0.1)) * losses["endpoint"]
            + float(getattr(args, "lambda_rigid_occupancy", 0.01)) * losses["occupancy"]
            + float(getattr(args, "lambda_rigid_entropy", 0.001)) * losses["entropy"]
        )
        grad = {f"grad_label_{name}": _grad_norm(value, labels.part_logits) for name, value in losses.items()}
        total.backward()
        optimizer.step()

        with torch.no_grad():
            hard = prob.argmax(-1)
            confidence = prob.max(-1).values
            entropy_each = -(prob.clamp_min(1.0e-8) * prob.clamp_min(1.0e-8).log()).sum(-1)
            unary = (
                float(getattr(args, "lambda_rigid_trajectory", 1.0)) * traj_e
                + float(getattr(args, "lambda_rigid_twist", 1.0)) * twist_e
                + float(getattr(args, "lambda_endpoint_geometry", 0.1)) * endpoint_e
            )
            preferred = unary.argmin(-1)
            row = {
                "iter": step,
                "temperature": temp,
                "loss": float(total.detach().cpu()),
                **{name: float(value.detach().cpu()) for name, value in losses.items()},
                **grad,
                "num_label_changes": int((hard != previous_hard).sum().detach().cpu()),
                "static_to_moving": int(((previous_hard == 0) & (hard > 0)).sum().detach().cpu()),
                "moving_to_static": int(((previous_hard > 0) & (hard == 0)).sum().detach().cpu()),
                "soft_label_l1_delta": float((prob - previous_prob).abs().sum(-1).mean().detach().cpu()),
                "mean_label_confidence": float(confidence.mean().detach().cpu()),
                "mean_label_entropy": float(entropy_each.mean().detach().cpu()),
                "candidate_static_points_that_prefer_any_moving_model": int(((hard == 0) & (preferred > 0)).sum().detach().cpu()),
                "candidate_moving_points_that_prefer_static_model": int(((hard > 0) & (preferred == 0)).sum().detach().cpu()),
            }
            for part in range(int(prob.shape[1])):
                row[f"num_part_{part}"] = int((hard == part).sum().detach().cpu())
                row[f"traj_unary_mean_part_{part}"] = float(traj_e[:, part].mean().detach().cpu())
                row[f"twist_unary_mean_part_{part}"] = float(twist_e[:, part].mean().detach().cpu())
                row[f"endpoint_unary_mean_part_{part}"] = float(endpoint_e[:, part].mean().detach().cpu())
            if row["num_label_changes"] == 0 and row["mean_label_confidence"] > 0.995 and step < max(3, steps // 5):
                row["warning"] = "labels_saturated_too_early"
                print("[Corrected rigid discovery] WARNING labels saturated too early")
            history.append(row)
            previous_hard = hard.detach()
            previous_prob = prob.detach()
        progress.set_postfix(loss=f"{history[-1]['loss']:.4g}", traj=f"{history[-1]['trajectory']:.3g}", end=f"{history[-1]['endpoint']:.3g}")
        if log_fn is not None and step % max(1, int(getattr(args, "wandb_log_interval", 10))) == 0:
            log_fn("corrected_rigid_parts", step, {f"corrected_rigid_parts/{k}": v for k, v in history[-1].items() if isinstance(v, (int, float))})

    with torch.no_grad():
        final_prob = labels.probabilities(_temperature(args, steps - 1, steps)).detach()
        final_hard = final_prob.argmax(-1)
        final_conf = final_prob.max(-1).values
        final_state = build_sequence_state(xyz0, pos, vel, final_prob, times)
        projected = _rigid_projected_tracks(xyz0, rot0, final_state, final_hard, times)
        raw = {
            "times": times.detach().cpu().float(),
            "positions": pos.detach().cpu(),
            "velocities": vel.detach().cpu(),
            "canonical_xyz": xyz0.detach().cpu(),
            "schema_version": 1,
            "source_of_truth": "label_independent_observed_velocity_integration",
        }
        summary = {
            "schema_version": 1,
            "algorithm": "corrected_raw_track_rigid_part_discovery",
            "uses_raw_tracks_for_discovery": True,
            "uses_rigid_replay_for_discovery": False,
            "static_leftover_frozen": False,
            "selected_frame_indices": select.cpu().tolist(),
            "selected_times": times.detach().cpu().tolist(),
            "refine_iters": steps,
            "class_counts_before": [int((before_hard == part).sum()) for part in range(int(final_prob.shape[1]))],
            "class_counts_after": [int((final_hard.cpu() == part).sum()) for part in range(int(final_prob.shape[1]))],
            "num_changed_from_initial": int((final_hard.cpu() != before_hard).sum()),
            "mean_label_confidence": float(final_conf.mean().cpu()),
            "mean_label_entropy": float(RigidPartLabels.entropy(final_prob.cpu())),
            "latest": history[-1] if history else {},
            "ground_truth_used_as_algorithm_input": False,
        }

    result_dir = output / "corrected_rigid_part_discovery"
    result_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(raw, result_dir / "raw_free_gaussian_tracks.pt")
    _atomic_torch_save(projected, result_dir / "rigid_projected_gaussian_tracks.pt")
    _atomic_torch_save({"labels": before_hard, "probabilities": before_prob.cpu()}, result_dir / "part_labels_before.pt")
    _atomic_torch_save({"labels": final_hard.cpu(), "confidence": final_conf.cpu()}, result_dir / "part_labels_after.pt")
    _atomic_torch_save({"probabilities": final_prob.cpu()}, result_dir / "part_probabilities_after.pt")
    _atomic_torch_save({"summary": summary, "history": history}, result_dir / "corrected_rigid_part_discovery_state.pt")
    (result_dir / "corrected_rigid_part_discovery_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, allow_nan=False), encoding="utf-8")
    (result_dir / "corrected_rigid_part_discovery_history.json").write_text(json.dumps(_jsonable(history), indent=2, allow_nan=False), encoding="utf-8")
    _write_csv(result_dir / "corrected_rigid_part_discovery_history.csv", history)
    _write_label_html(result_dir / "corrected_rigid_part_labels.html", xyz0.detach().cpu(), final_hard.detach().cpu(), final_conf.detach().cpu(), int(getattr(args, "transported_html_max_points", 5000)))
    return {
        "summary": summary,
        "responsibilities": final_prob.cpu(),
        "labels": final_hard.cpu(),
        "confidence": final_conf.cpu(),
        "raw_tracks": raw,
        "rigid_tracks": projected,
        "state": {
            "rotation": final_state.rotation.detach().cpu(),
            "translation": final_state.translation.detach().cpu(),
            "omega": final_state.omega.detach().cpu(),
            "b": final_state.b.detach().cpu(),
            "times": times.detach().cpu(),
        },
    }


def run_exact_closed_loop_rigid_part_discovery(
    stage1: dict,
    scene,
    gaussians,
    pipe,
    background: torch.Tensor,
    dataset,
    args,
    output: Path,
    *,
    log_fn=None,
) -> dict:
    """Exact closed-loop raw-motion/rigid-part/projection discovery.

    This path intentionally treats the old velocity-field K-part result as an
    observation source and weak initialization only.  It optimizes a trainable
    raw/free Gaussian velocity field, refits soft rigid parts from the raw
    tracks, renders the rigid projection, and feeds the projection back into the
    next raw-motion update.
    """
    if "raw_tracks" not in stage1 or "raw_velocities" not in stage1:
        raise RuntimeError("Exact closed-loop rigid discovery requires raw observed tracks/velocities")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = output / "exact_closed_loop_rigid_parts"
    result_dir.mkdir(parents=True, exist_ok=True)

    xyz0 = torch.as_tensor(stage1["raw_tracks"]["canonical_xyz"]).detach().float().to(device)
    rot0 = torch.as_tensor(stage1["coarse_tracks"]["rotations"][0]).detach().float().to(device)
    raw_times = torch.as_tensor(stage1["raw_tracks"]["times"]).detach().float().to(device)
    raw_velocity_init = torch.as_tensor(stage1["raw_velocities"]).detach().float()
    if raw_velocity_init.shape[0] == raw_times.numel():
        raw_velocity_init = raw_velocity_init[:-1]
    observed = _observed_velocity_state(stage1, raw_times.detach().cpu(), raw_velocity_init.detach().cpu())
    init_interval = raw_velocity_init.to(device)
    if init_interval.shape != observed.velocities.shape:
        init_interval = observed.velocities.to(device)
    interval_omega = torch.nn.Parameter(torch.zeros_like(init_interval))
    interval_b = torch.nn.Parameter(init_interval.clone())

    init_prob = _initial_prob(stage1, int(getattr(args, "num_moving_parts", 1)))
    labels = RigidPartLabels(
        xyz0.shape[0],
        int(init_prob.shape[1]) - 1,
        init_logits=torch.log(_soften(init_prob).clamp_min(1.0e-8)),
        temperature=float(getattr(args, "rigid_part_label_temperature", 1.0)),
    ).to(device)
    prior = _dynamic_prior_payload(stage1, int(xyz0.shape[0]))

    final_ply = _resolve_final_ply(str(getattr(args, "final_3dgs_path", "")))
    final_xyz = load_ply_xyz(final_ply).to(device) if final_ply is not None else None

    edge_src, edge_dst, edge_d0 = build_knn_graph(xyz0.detach().cpu(), k=int(getattr(args, "rigid_part_knn", 16)))
    edge_src, edge_dst, edge_d0 = edge_src.to(device), edge_dst.to(device), edge_d0.to(device)
    render_views = _selected_render_views(
        scene,
        stage1,
        max_frames=int(getattr(args, "rigid_part_render_frames", 4)),
        max_cameras=int(getattr(args, "rigid_part_render_cameras", 2)),
    )
    frame_select = _selected_indices(int(raw_times.numel()), int(getattr(args, "rigid_part_max_frames", 20))).to(device)

    raw_optimizer = torch.optim.Adam([interval_omega, interval_b], lr=float(getattr(args, "rigid_part_raw_lr", getattr(args, "rigid_part_lr", 5.0e-2))))
    label_optimizer = torch.optim.Adam(labels.parameters(), lr=float(getattr(args, "rigid_part_lr", 5.0e-2)))
    outer_iters = max(1, int(getattr(args, "rigid_part_outer_iters", 3)))
    local_steps = max(0, int(getattr(args, "rigid_part_local_motion_steps", 5)))
    label_steps = max(1, int(getattr(args, "rigid_part_label_steps", getattr(args, "rigid_part_refine_iters", 25))))

    history: list[dict[str, object]] = []
    outer_summaries: list[dict[str, object]] = []
    previous_projection: dict | None = None
    previous_prob: torch.Tensor | None = None
    previous_hard: torch.Tensor | None = None
    previous_feedback_enabled = False

    observed_payload = {
        "schema_version": 1,
        "velocities": observed.velocities,
        "confidence": observed.confidence,
        "valid": observed.valid,
        "source_of_truth": "rasterized_multiview_observed_velocity_measurement_only",
        "ground_truth_used_as_algorithm_input": False,
    }
    _atomic_torch_save(observed_payload, result_dir / "observed_motion_state.pt")

    progress = tqdm(range(outer_iters), desc="Exact closed-loop rigid parts", unit="outer")
    final_state = None
    final_prob = None
    final_hard = None
    final_conf = None
    final_raw = None
    final_projection = None
    for outer in progress:
        # A. Local/free motion update against render, observed velocity, Killing
        # regularity, temporal smoothness, and previous rigid projection.
        raw_progress = tqdm(range(local_steps), desc=f"outer {outer:03d} raw/free motion", unit="step", leave=False)
        for step in raw_progress:
            raw_optimizer.zero_grad(set_to_none=True)
            raw_state = _integrate_local_killing_twist(xyz0, rot0, interval_omega, interval_b, raw_times)
            render_loss, render_stats = _render_state_loss(
                views=render_views,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                dataset=dataset,
                args=args,
                positions=raw_state.tracks,
                rotations=raw_state.rotations,
                debug_dir=None,
                prefix="raw",
            )
            raw_losses = {
                "render": render_loss,
                "observed_velocity": _observed_velocity_loss(raw_state.interval_velocities, observed),
                "killing": _raw_killing_loss(raw_state.tracks, raw_state.velocities, edge_src, edge_dst),
                "temporal": _raw_temporal_loss(raw_state.interval_velocities) + _raw_twist_temporal_loss(interval_omega, interval_b),
                "rigid_feedback": _rigid_feedback_loss(
                    raw_state,
                    previous_projection if previous_feedback_enabled else None,
                    previous_prob if previous_feedback_enabled else None,
                ),
            }
            raw_weights = {
                "render": 1.0,
                "observed_velocity": float(getattr(args, "lambda_raw_observed_velocity", 1.0)),
                "killing": float(getattr(args, "lambda_raw_killing", 0.1)),
                "temporal": float(getattr(args, "lambda_raw_temporal", 0.01)),
                "rigid_feedback": float(getattr(args, "lambda_raw_rigid_feedback", 0.1)),
            }
            total, weighted = _weighted_sum(raw_losses, raw_weights)
            grad = {f"grad_motion_{name}": _grad_norm_many(value, [interval_omega, interval_b]) for name, value in raw_losses.items()}
            grad.update({
                "grad_motion_total_omega": _grad_norm(total, interval_omega),
                "grad_motion_total_b": _grad_norm(total, interval_b),
            })
            total.backward()
            raw_optimizer.step()
            row = {
                "phase": "raw_motion",
                "outer": outer,
                "step": step,
                "loss_total": float(total.detach().cpu()),
                **{f"loss_{k}": float(v.detach().cpu()) for k, v in raw_losses.items()},
                **{f"weighted_{k}": float(v.detach().cpu()) for k, v in weighted.items()},
                **_loss_fractions(weighted),
                **grad,
                "render_rgb": render_stats["rgb"],
                "render_depth": render_stats["depth"],
                "render_views": render_stats["views"],
                "rigid_feedback_enabled": int(previous_feedback_enabled),
            }
            history.append(row)
            raw_progress.set_postfix(
                total=f"{row['loss_total']:.3g}",
                render=f"{row['loss_render']:.3g}",
                obs=f"{row['loss_observed_velocity']:.3g}",
                kill=f"{row['loss_killing']:.2g}",
                fb=f"{row['loss_rigid_feedback']:.2g}",
                gate=int(previous_feedback_enabled),
            )
            if log_fn is not None:
                log_fn("exact_closed_loop/raw", outer * max(1, local_steps) + step, {f"exact_closed_loop/raw/{k}": v for k, v in row.items() if isinstance(v, (int, float))})

        # B. Soft label/part update from current raw trajectories.
        with torch.no_grad():
            raw_state = _integrate_local_killing_twist(xyz0, rot0, interval_omega, interval_b, raw_times)
        pos = raw_state.tracks.index_select(0, frame_select).detach()
        vel = raw_state.velocities.index_select(0, frame_select).detach()
        times = raw_times.index_select(0, frame_select)
        affinity = raw_graph_affinity(pos, vel, edge_src, edge_dst, edge_d0).detach()
        label_progress = tqdm(range(label_steps), desc=f"outer {outer:03d} soft rigid labels", unit="step", leave=False)
        for step in label_progress:
            label_optimizer.zero_grad(set_to_none=True)
            global_step = outer * label_steps + step
            temp = _temperature(args, global_step, max(1, outer_iters * label_steps))
            prob = labels.probabilities(temp)
            state = build_sequence_state(xyz0, pos, vel, prob, times)
            traj_e = trajectory_unary(xyz0, pos, state)
            twist_e = twist_unary(pos, vel, state)
            endpoint_e = (
                compute_counterfactual_endpoint_unary(
                    xyz0,
                    state.rotation[-1],
                    state.translation[-1],
                    final_xyz,
                    chunk_size=int(getattr(args, "rigid_part_endpoint_chunk", 2048)),
                )
                if final_xyz is not None
                else torch.zeros_like(traj_e)
            )
            label_losses = {
                "trajectory": trajectory_unary_loss(traj_e, prob),
                "twist": twist_unary_loss(twist_e, prob),
                "graph_attractive": graph_label_coherence_loss(prob, edge_src, edge_dst, affinity),
                "graph_repulsive": _graph_repulsive_loss(prob, edge_src, edge_dst, affinity),
                "pair": pairwise_distance_loss(pos, edge_src, edge_dst, edge_d0, prob),
                "distance_rate": distance_rate_loss(pos, vel, edge_src, edge_dst, prob),
                "com": com_transport_loss(state),
                "inertia": inertia_transport_loss(state),
                "inertia_invariants": inertia_invariant_loss(state),
                "endpoint": (prob * endpoint_e).sum() / prob.sum().clamp_min(1.0e-8),
                "prior": _dynamic_prior_loss(prob, prior),
                "moving_complexity": _moving_complexity_loss(prob),
                "occupancy": occupancy_loss(prob, float(getattr(args, "rigid_part_min_fraction", 0.01))),
                "entropy": RigidPartLabels.entropy(prob),
            }
            label_weights = {
                "trajectory": float(getattr(args, "lambda_rigid_trajectory", 1.0)),
                "twist": float(getattr(args, "lambda_rigid_twist", 1.0)),
                "graph_attractive": float(getattr(args, "lambda_rigid_graph", 1.0)),
                "graph_repulsive": float(getattr(args, "lambda_rigid_graph_repulsive", 0.5)),
                "pair": float(getattr(args, "lambda_rigid_pair", 1.0)),
                "distance_rate": float(getattr(args, "lambda_rigid_distance_rate", 1.0)),
                "com": float(getattr(args, "lambda_rigid_com", 0.1)),
                "inertia": float(getattr(args, "lambda_rigid_inertia", 0.1)),
                "inertia_invariants": float(getattr(args, "lambda_rigid_inertia_invariants", 0.05)),
                "endpoint": float(getattr(args, "lambda_rigid_endpoint", getattr(args, "lambda_endpoint_geometry", 0.1))),
                "prior": float(getattr(args, "lambda_rigid_prior", 0.1)),
                "moving_complexity": float(getattr(args, "lambda_rigid_moving_complexity", 0.01)),
                "occupancy": float(getattr(args, "lambda_rigid_occupancy", 0.01)),
                "entropy": float(getattr(args, "lambda_rigid_entropy", 0.001)),
            }
            total, weighted = _weighted_sum(label_losses, label_weights)
            grad = {f"grad_label_{name}": _grad_norm(value, labels.part_logits) for name, value in label_losses.items()}
            total.backward()
            label_optimizer.step()
            with torch.no_grad():
                hard = prob.argmax(-1)
                conf = prob.max(-1).values
                changed = int((hard != previous_hard).sum().detach().cpu()) if previous_hard is not None else int(hard.numel())
                row = {
                    "phase": "labels",
                    "outer": outer,
                    "step": step,
                    "temperature": temp,
                    "loss_total": float(total.detach().cpu()),
                    **{f"loss_{k}": float(v.detach().cpu()) for k, v in label_losses.items()},
                    **{f"weighted_{k}": float(v.detach().cpu()) for k, v in weighted.items()},
                    **_loss_fractions(weighted),
                    **grad,
                    "label_changes_from_previous_outer": changed,
                    "mean_label_confidence": float(conf.mean().detach().cpu()),
                    "mean_dynamic_probability": float((1.0 - prob[:, 0]).mean().detach().cpu()),
                }
                for part in range(int(prob.shape[1])):
                    row[f"num_part_{part}"] = int((hard == part).sum().detach().cpu())
                history.append(row)
                label_progress.set_postfix(
                    total=f"{row['loss_total']:.3g}",
                    traj=f"{row['loss_trajectory']:.2g}",
                    twist=f"{row['loss_twist']:.2g}",
                    rep=f"{row['loss_graph_repulsive']:.2g}",
                    prior=f"{row['loss_prior']:.2g}",
                    dyn=f"{row['mean_dynamic_probability']:.3f}",
                    conf=f"{row['mean_label_confidence']:.3f}",
                )
                if log_fn is not None:
                    log_fn("exact_closed_loop/labels", outer * label_steps + step, {f"exact_closed_loop/labels/{k}": v for k, v in row.items() if isinstance(v, (int, float))})

        # C. Refit rigid state from current labels, project, render, and store
        # feedback for the next outer loop.
        with torch.no_grad():
            raw_state = _integrate_local_killing_twist(xyz0, rot0, interval_omega, interval_b, raw_times)
            final_prob = labels.probabilities(_temperature(args, (outer + 1) * label_steps - 1, max(1, outer_iters * label_steps))).detach()
            final_hard = final_prob.argmax(-1)
            final_conf = final_prob.max(-1).values
            full_state = build_sequence_state(xyz0, raw_state.tracks.detach(), raw_state.velocities.detach(), final_prob, raw_times)
            projected = _rigid_projected_tracks(xyz0, rot0, full_state, final_hard, raw_times)
            rigid_pos = projected["positions"].to(device)
            rigid_rot = projected["rotations"].to(device)
            raw_render_loss, raw_render_stats = _render_state_loss(
                views=render_views,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                dataset=dataset,
                args=args,
                positions=raw_state.tracks.detach(),
                rotations=raw_state.rotations,
                debug_dir=result_dir / f"outer_{outer:04d}" / "raw_render",
                prefix="raw",
            )
            rigid_render_loss, rigid_render_stats = _render_state_loss(
                views=render_views,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                dataset=dataset,
                args=args,
                positions=rigid_pos,
                rotations=rigid_rot,
                debug_dir=result_dir / f"outer_{outer:04d}" / "rigid_render",
                prefix="rigid",
            )
            projection_position_error = torch.linalg.norm(raw_state.tracks.detach() - rigid_pos, dim=-1)
            projection_velocity_error = torch.linalg.norm(raw_state.velocities.detach() - projected["velocities"].to(device), dim=-1)
            label_changes = int((final_hard != previous_hard).sum().detach().cpu()) if previous_hard is not None else int(final_hard.numel())
            summary = {
                "outer": outer,
                "raw_render_rgb": raw_render_stats["rgb"],
                "raw_render_depth": raw_render_stats["depth"],
                "rigid_render_rgb": rigid_render_stats["rgb"],
                "rigid_render_depth": rigid_render_stats["depth"],
                "rigid_minus_raw_render": float((rigid_render_loss - raw_render_loss).detach().cpu()),
                "rigid_feedback_next_enabled": bool(rigid_render_loss <= raw_render_loss),
                "projection_position_error_mean": float(projection_position_error.mean().detach().cpu()),
                "projection_position_error_p90": float(torch.quantile(projection_position_error.flatten(), 0.9).detach().cpu()),
                "projection_velocity_error_mean": float(projection_velocity_error.mean().detach().cpu()),
                "projection_velocity_error_p90": float(torch.quantile(projection_velocity_error.flatten(), 0.9).detach().cpu()),
                "label_changes": label_changes,
                "mean_label_confidence": float(final_conf.mean().detach().cpu()),
                "mean_dynamic_probability": float((1.0 - final_prob[:, 0]).mean().detach().cpu()),
            }
            for part in range(int(final_prob.shape[1])):
                summary[f"num_part_{part}"] = int((final_hard == part).sum().detach().cpu())
            outer_summaries.append(summary)
            history.append({"phase": "outer_summary", **summary})
            (result_dir / f"outer_{outer:04d}" / "outer_summary.json").parent.mkdir(parents=True, exist_ok=True)
            (result_dir / f"outer_{outer:04d}" / "outer_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, allow_nan=False), encoding="utf-8")
            previous_projection = projected
            previous_prob = final_prob.detach()
            previous_hard = final_hard.detach()
            previous_feedback_enabled = bool(summary["rigid_feedback_next_enabled"])
            final_state = full_state
            final_raw = raw_state
            final_projection = projected
        progress.set_postfix(
            dyn=f"{outer_summaries[-1]['mean_dynamic_probability']:.3f}",
            proj=f"{outer_summaries[-1]['projection_position_error_mean']:.3g}",
            dR=f"{outer_summaries[-1]['rigid_minus_raw_render']:.3g}",
            fb=int(outer_summaries[-1]["rigid_feedback_next_enabled"]),
        )

    assert final_state is not None and final_prob is not None and final_hard is not None and final_conf is not None and final_raw is not None and final_projection is not None
    raw = {
        "schema_version": 3,
        "source_of_truth": "render_optimized_local_killing_twist_closed_loop",
        "times": raw_times.detach().cpu().float(),
        "positions": final_raw.tracks.detach().cpu(),
        "rotations": final_raw.rotations.detach().cpu(),
        "velocities": final_raw.velocities.detach().cpu(),
        "interval_velocities": final_raw.interval_velocities.detach().cpu(),
        "interval_omega": final_raw.interval_omega.detach().cpu(),
        "interval_b": final_raw.interval_b.detach().cpu(),
        "canonical_xyz": xyz0.detach().cpu(),
        "se3_integration_convention": "per-Gaussian spatial/world-frame twist: x_dot = omega cross x + b; q_next = q_inc ⊗ q",
        "ground_truth_used_as_algorithm_input": False,
    }
    rigid_state = {
        "schema_version": 1,
        "probabilities": final_prob.detach().cpu(),
        "labels": final_hard.detach().cpu(),
        "confidence": final_conf.detach().cpu(),
        "rotation": final_state.rotation.detach().cpu(),
        "translation": final_state.translation.detach().cpu(),
        "omega": final_state.omega.detach().cpu(),
        "b": final_state.b.detach().cpu(),
        "com": final_state.com.detach().cpu(),
        "inertia": final_state.inertia.detach().cpu(),
        "ground_truth_used_as_algorithm_input": False,
    }
    final_projection.update({
        "schema_version": 4,
        "source_of_truth": "exact_closed_loop_rigid_projection",
        "motion_membership": final_prob[:, 1:].sum(-1).detach().cpu(),
        "part_probabilities": final_prob.detach().cpu(),
        "fixed_cameras": stage1["coarse_tracks"].get("fixed_cameras", []),
    })
    summary = {
        "schema_version": 1,
        "algorithm": "exact_closed_loop_rigid_part_architecture",
        "outer_iters": outer_iters,
        "local_motion_steps": local_steps,
        "label_steps": label_steps,
        "render_views": len(render_views),
        "frame_indices_for_label_fit": frame_select.detach().cpu().tolist(),
        "class_counts_after": [int((final_hard == part).sum().detach().cpu()) for part in range(int(final_prob.shape[1]))],
        "mean_label_confidence": float(final_conf.mean().detach().cpu()),
        "mean_dynamic_probability": float((1.0 - final_prob[:, 0]).mean().detach().cpu()),
        "latest_outer": outer_summaries[-1] if outer_summaries else {},
        "uses_3dgs_cuda_render_for_raw_motion": True,
        "uses_3dgs_cuda_render_for_rigid_projection": True,
        "raw_motion_parameterization": "per-Gaussian local Killing spatial twists integrated by exact SE(3) exponential",
        "rigid_projection_parameterization": "shared part Killing spatial twists integrated by exact SE(3) exponential",
        "gaussian_orientation_transport": "q_next = q(Exp_SO3(dt omega)) ⊗ q_current",
        "euler_position_integration_used": False,
        "legacy_kpart_used_only_for_observations_and_initialization": True,
        "observed_velocity_is_measurement_not_final_motion": True,
        "ground_truth_used_as_algorithm_input": False,
    }

    _atomic_torch_save(raw, result_dir / "raw_motion_state.pt")
    _atomic_torch_save(rigid_state, result_dir / "rigid_part_state.pt")
    _atomic_torch_save(final_projection, result_dir / "rigid_projection_state.pt")
    _atomic_torch_save({"summary": summary, "history": history, "outer_summaries": outer_summaries}, result_dir / "exact_closed_loop_state.pt")
    _atomic_torch_save(raw, result_dir / "raw_free_gaussian_tracks.pt")
    _atomic_torch_save(final_projection, result_dir / "rigid_projected_gaussian_tracks.pt")
    (result_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, allow_nan=False), encoding="utf-8")
    (result_dir / "history.json").write_text(json.dumps(_jsonable(history), indent=2, allow_nan=False), encoding="utf-8")
    (result_dir / "outer_summaries.json").write_text(json.dumps(_jsonable(outer_summaries), indent=2, allow_nan=False), encoding="utf-8")
    _write_csv(result_dir / "history.csv", history)
    _write_csv(result_dir / "outer_summaries.csv", outer_summaries)
    _write_loss_curves_html(result_dir / "loss_curves.html", history)
    _write_compact_debug(result_dir, history, outer_summaries)
    _write_label_html(
        result_dir / "final_labels.html",
        xyz0.detach().cpu(),
        final_hard.detach().cpu(),
        final_conf.detach().cpu(),
        int(getattr(args, "transported_html_max_points", 5000)),
    )
    return {
        "summary": summary,
        "responsibilities": final_prob.detach().cpu(),
        "labels": final_hard.detach().cpu(),
        "confidence": final_conf.detach().cpu(),
        "raw_tracks": raw,
        "rigid_tracks": final_projection,
        "state": {
            "rotation": final_state.rotation.detach().cpu(),
            "translation": final_state.translation.detach().cpu(),
            "omega": final_state.omega.detach().cpu(),
            "b": final_state.b.detach().cpu(),
            "com": final_state.com.detach().cpu(),
            "inertia": final_state.inertia.detach().cpu(),
            "times": raw_times.detach().cpu(),
        },
        "stage2_legacy_modes_are_stale": True,
    }


def run_rigid_part_discovery(stage1: dict, gaussians, args, output: Path, *, log_fn=None) -> dict:
    if "raw_tracks" not in stage1 or "raw_velocities" not in stage1:
        raise RuntimeError("Corrected rigid-part discovery requires stage1['raw_tracks'] and stage1['raw_velocities']; refusing to use rigid replay as discovery evidence")
    final_ply = _resolve_final_ply(str(getattr(args, "final_3dgs_path", "")))
    final_xyz = load_ply_xyz(final_ply) if final_ply is not None else None
    init_prob = _initial_prob(stage1, int(getattr(args, "num_moving_parts", 1)))
    return run_corrected_rigid_part_discovery_from_tracks(
        canonical_xyz=torch.as_tensor(stage1["raw_tracks"]["canonical_xyz"]),
        canonical_rotation=torch.as_tensor(stage1["coarse_tracks"]["rotations"][0]),
        raw_tracks=torch.as_tensor(stage1["raw_tracks"]["positions"]),
        raw_velocities=torch.as_tensor(stage1["raw_velocities"]),
        raw_times=torch.as_tensor(stage1["raw_tracks"]["times"]),
        init_prob=init_prob,
        args=args,
        output=output,
        final_xyz=final_xyz,
        log_fn=log_fn,
    )


def run_rigid_part_killing_inertia_refinement(stage1: dict, gaussians, args, output: Path, *, log_fn=None) -> dict:
    """Compatibility wrapper: now runs corrected raw-track discovery."""
    return run_rigid_part_discovery(stage1, gaussians, args, output, log_fn=log_fn)
