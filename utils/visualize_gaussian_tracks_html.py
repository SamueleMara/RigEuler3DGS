"""Interactive Plotly HTML visualization for exported Gaussian trajectories.

This version keeps all geometry in the same real world coordinate system:
  - Gaussian positions are displayed exactly as stored in gaussian_tracks.pt;
  - camera centers are displayed exactly as stored/derived from cameras.json;
  - frustum corners are computed from real intrinsics/extrinsics with a user
    chosen frustum depth in world units, like the older joint/depth HTML tools;
  - camera/object coordinates are never rescaled; aspectmode is configurable.
    The default is cube, matching the older debug visualizers, because metric
    data aspect makes shallow scenes look visually flattened when cameras are far away.
"""

from __future__ import annotations

import json
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

CAMERA_COLORS = [
    "#f94144", "#277da1", "#90be6d", "#f8961e",
    "#577590", "#43aa8b", "#f3722c", "#4d908e",
]


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _frame_ids(tracks: Dict) -> np.ndarray:
    positions = tracks["positions"]
    if "frame_ids" in tracks and tracks["frame_ids"] is not None:
        return _to_numpy(tracks["frame_ids"]).astype(np.int64)
    return np.arange(int(positions.shape[0]), dtype=np.int64)


def _max_speed(tracks: Dict) -> np.ndarray:
    n = int(tracks["positions"].shape[1])
    if "velocities" in tracks and tracks["velocities"] is not None:
        vel = _to_numpy(tracks["velocities"]).astype(np.float32)
        if vel.ndim == 3 and vel.shape[1] == n:
            return np.linalg.norm(vel, axis=-1).max(axis=0)
    pos = _to_numpy(tracks["positions"]).astype(np.float32)
    if pos.shape[0] <= 1:
        return np.zeros((n,), dtype=np.float32)
    return np.linalg.norm(np.diff(pos, axis=0), axis=-1).max(axis=0)


def _speed_labels(tracks: Dict, dynamic_threshold: float = 0.5, speed_percentile: float = 80.0) -> Tuple[np.ndarray, Dict]:
    speed = _max_speed(tracks)
    positive = speed[speed > 1e-12]
    if positive.size == 0:
        labels = np.zeros_like(speed, dtype=bool)
        return labels, {"source": "speed", "speed_threshold": 0.0, "speed_percentile": float(speed_percentile)}
    p = float(speed_percentile)
    if 0.0 <= p <= 100.0:
        thr = float(np.percentile(positive, p))
        labels = speed >= thr
    else:
        denom = float(np.percentile(positive, 95.0) + 1e-8)
        labels = np.clip(speed / denom, 0.0, 1.0) >= float(dynamic_threshold)
        thr = float(dynamic_threshold * denom)
    return labels, {"source": "speed", "speed_threshold": thr, "speed_percentile": float(speed_percentile)}


def _prob_labels(tracks: Dict, threshold: float) -> Optional[Tuple[np.ndarray, Dict]]:
    n = int(tracks["positions"].shape[1])
    if "dynamic_prob" not in tracks or tracks["dynamic_prob"] is None:
        return None
    prob = _to_numpy(tracks["dynamic_prob"]).reshape(-1)
    if prob.shape[0] != n:
        return None
    labels = prob >= float(threshold)
    return labels, {
        "source": "dynamic_prob",
        "prob_threshold": float(threshold),
        "prob_min": float(np.nanmin(prob)),
        "prob_max": float(np.nanmax(prob)),
        "prob_mean": float(np.nanmean(prob)),
    }


def _compute_dynamic_labels(tracks: Dict, threshold: float, label_mode: str = "auto", speed_percentile: float = 80.0) -> Tuple[np.ndarray, Dict]:
    label_mode = str(label_mode).lower()
    speed_labels, speed_meta = _speed_labels(tracks, threshold, speed_percentile)
    prob_result = _prob_labels(tracks, threshold)
    if label_mode == "speed" or prob_result is None:
        return speed_labels, speed_meta
    prob_labels, prob_meta = prob_result
    n_dyn = int(prob_labels.sum())
    n = int(prob_labels.shape[0])
    if label_mode == "prob":
        return prob_labels, prob_meta
    if label_mode == "combined":
        labels = np.logical_or(prob_labels, speed_labels)
        return labels, {"source": "combined", "prob": prob_meta, "speed": speed_meta}
    if 0 < n_dyn < n:
        return prob_labels, {**prob_meta, "source": "dynamic_prob_auto"}
    return speed_labels, {"source": "speed_auto_fallback", "prob": prob_meta, "speed": speed_meta}


def _sample_indices(labels: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    n = labels.shape[0]
    if n <= int(max_points):
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    dyn_idx = np.flatnonzero(labels)
    sta_idx = np.flatnonzero(~labels)
    dyn_budget = min(len(dyn_idx), int(max_points) // 2 if len(sta_idx) else int(max_points))
    sta_budget = int(max_points) - dyn_budget
    if len(dyn_idx) > dyn_budget:
        dyn_idx = rng.choice(dyn_idx, size=dyn_budget, replace=False)
    if len(sta_idx) > sta_budget:
        sta_idx = rng.choice(sta_idx, size=sta_budget, replace=False)
    sample = np.concatenate([sta_idx, dyn_idx])
    rng.shuffle(sample)
    return sample.astype(np.int64)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def _camera_lookat_to_r_cw(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return camera-to-world rotation, matching the older visualization scripts."""
    z_axis = _normalize(target - position)
    x_axis = _normalize(np.cross(z_axis, up))
    y_up = _normalize(np.cross(x_axis, z_axis))
    y_axis = -y_up
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float64)


def _as_float(v, default=None):
    if v is None:
        return default
    try:
        out = float(v)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _camera_name(cam: Dict, i: int) -> str:
    raw = str(cam.get("name", cam.get("img_name", cam.get("image_name", cam.get("file_path", f"camera_{i}")))))
    parts = raw.replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("cam_"):
            return part[:7] if len(part) >= 7 else part
    if "cam_" in raw:
        j = raw.find("cam_")
        return raw[j:j + 7]
    return raw


def _parse_camera_entry(cam: Dict, i: int) -> Optional[Dict]:
    """Parse GraphDeco cameras.json or metadata/cameras.json into metric camera parameters."""
    name = _camera_name(cam, i)
    source_name = str(cam.get("source_camera_name", name))
    width = int(cam.get("width", cam.get("w", cam.get("image_width", 0))) or 0)
    height = int(cam.get("height", cam.get("h", cam.get("image_height", 0))) or 0)

    fx = _as_float(cam.get("fx"), None)
    fy = _as_float(cam.get("fy"), None)
    cx = _as_float(cam.get("cx"), None)
    cy = _as_float(cam.get("cy"), None)
    if (fx is None or fy is None) and width > 0 and height > 0:
        fov_y = _as_float(cam.get("fov_y_deg", cam.get("FoVy", cam.get("fov_y"))), None)
        fov_x = _as_float(cam.get("fov_x_deg", cam.get("FoVx", cam.get("fov_x"))), None)
        if fov_y is not None:
            if fov_y > math.pi:  # degrees
                fov_y = math.radians(fov_y)
            fy = (0.5 * float(height)) / max(math.tan(0.5 * fov_y), 1e-8)
        if fov_x is not None:
            if fov_x > math.pi:
                fov_x = math.radians(fov_x)
            fx = (0.5 * float(width)) / max(math.tan(0.5 * fov_x), 1e-8)
    if fx is None and fy is not None:
        fx = fy
    if fy is None and fx is not None:
        fy = fx
    if cx is None and width > 0:
        cx = 0.5 * (float(width) - 1.0)
    if cy is None and height > 0:
        cy = 0.5 * (float(height) - 1.0)

    center = None
    r_cw = None
    if "transform_matrix" in cam:
        mat = np.asarray(cam["transform_matrix"], dtype=np.float64).reshape(4, 4)
        center = mat[:3, 3]
        r_cw = mat[:3, :3]
    elif cam.get("position", None) is not None:
        center = np.asarray(cam["position"], dtype=np.float64).reshape(3)
        target = cam.get("target", cam.get("look_at", None))
        up = cam.get("up", None)
        if target is not None and up is not None:
            try:
                r_cw = _camera_lookat_to_r_cw(center, np.asarray(target, dtype=np.float64).reshape(3), np.asarray(up, dtype=np.float64).reshape(3))
            except Exception:
                r_cw = None
        if r_cw is None and cam.get("rotation", None) is not None:
            # GraphDeco camera_to_JSON writes camera-to-world rotation with position.
            r_cw = np.asarray(cam["rotation"], dtype=np.float64).reshape(3, 3)
    elif cam.get("R", None) is not None and cam.get("T", cam.get("translation", None)) is not None:
        r_wc = np.asarray(cam["R"], dtype=np.float64).reshape(3, 3)
        t_wc = np.asarray(cam.get("T", cam.get("translation")), dtype=np.float64).reshape(3)
        center = -r_wc.T @ t_wc
        r_cw = r_wc.T

    if center is None:
        return None
    if r_cw is None:
        r_cw = np.eye(3, dtype=np.float64)
    if width <= 0:
        width = 640
    if height <= 0:
        height = 480
    if fx is None:
        fx = float(width)
    if fy is None:
        fy = float(height)
    if cx is None:
        cx = 0.5 * (float(width) - 1.0)
    if cy is None:
        cy = 0.5 * (float(height) - 1.0)

    return {
        "name": name,
        "source_camera_name": source_name,
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "center": np.asarray(center, dtype=np.float64).reshape(3),
        "r_cw": np.asarray(r_cw, dtype=np.float64).reshape(3, 3),
    }


def _deduplicate_camera_params(params: List[Dict], decimals: int = 6) -> List[Dict]:
    seen = set()
    out = []
    for cp in params:
        key = (
            cp["name"],
            tuple(np.round(cp["center"].astype(np.float64), decimals).tolist()),
            tuple(np.round(cp["r_cw"].reshape(-1).astype(np.float64), 4).tolist()),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(cp)
    return out


def _compute_frustum_world_points(cp: Dict, frustum_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute frustum geometry in real world coordinates, no rescaling/recentering."""
    z = float(frustum_scale)
    corners_px = np.array(
        [[0.0, 0.0], [float(cp["width"] - 1), 0.0], [float(cp["width"] - 1), float(cp["height"] - 1)], [0.0, float(cp["height"] - 1)]],
        dtype=np.float64,
    )
    corners_cam = np.stack(
        [
            (corners_px[:, 0] - float(cp["cx"])) / max(float(cp["fx"]), 1e-8) * z,
            (corners_px[:, 1] - float(cp["cy"])) / max(float(cp["fy"]), 1e-8) * z,
            np.full((4,), z, dtype=np.float64),
        ],
        axis=1,
    )
    center = np.asarray(cp["center"], dtype=np.float64).reshape(3)
    corners_world = (np.asarray(cp["r_cw"], dtype=np.float64) @ corners_cam.T).T + center[None, :]
    return center, corners_world


def _frustum_line_array(center: np.ndarray, corners_world: np.ndarray) -> np.ndarray:
    pts: List[np.ndarray] = []
    nan = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        pts.extend([corners_world[a], corners_world[b], nan])
    for q in corners_world:
        pts.extend([center, q, nan])
    return np.asarray(pts, dtype=np.float32)


def load_camera_frustums(
    cameras_json: str,
    frustum_scale: float = 0.12,
    deduplicate: bool = True,
) -> Tuple[List[np.ndarray], List[str], Dict, np.ndarray]:
    """Load camera frustums from JSON in true world coordinates.

    This mirrors the older visualization scripts: parse metadata/cameras.json or
    GraphDeco cameras.json, compute corners from intrinsics/extrinsics, and do
    not move/rescale camera centers relative to the object.
    """
    meta = {
        "cameras_json": str(cameras_json) if cameras_json else "",
        "camera_coordinate_mode": "world",
        "frustum_scale_world_units": float(frustum_scale),
        "camera_positions_rescaled": False,
        "num_camera_records": 0,
        "num_camera_frustums": 0,
    }
    empty_centers = np.zeros((0, 3), dtype=np.float32)
    if not cameras_json:
        return [], [], meta, empty_centers
    path = Path(cameras_json).expanduser()
    if not path.exists():
        meta["error"] = f"missing_camera_json: {path}"
        return [], [], meta, empty_centers
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        meta["error"] = f"failed_to_read_json: {exc}"
        return [], [], meta, empty_centers
    cams = data.get("cameras", data.get("frames", [])) if isinstance(data, dict) else data
    if not isinstance(cams, list):
        meta["error"] = "camera_json_not_list"
        return [], [], meta, empty_centers
    params = []
    for i, cam in enumerate(cams):
        if isinstance(cam, dict):
            parsed = _parse_camera_entry(cam, i)
            if parsed is not None:
                params.append(parsed)
    meta["num_camera_records"] = int(len(params))
    if deduplicate:
        params = _deduplicate_camera_params(params)
    if not params:
        meta["error"] = "no_parseable_camera_frustums"
        return [], [], meta, empty_centers

    frustums: List[np.ndarray] = []
    names: List[str] = []
    centers: List[np.ndarray] = []
    per_camera = []
    for cp in params:
        center, corners = _compute_frustum_world_points(cp, frustum_scale=float(frustum_scale))
        frustums.append(_frustum_line_array(center, corners))
        names.append(str(cp["name"]))
        centers.append(center.astype(np.float32))
        per_camera.append({
            "name": str(cp["name"]),
            "source_camera_name": str(cp["source_camera_name"]),
            "position": center.astype(float).tolist(),
            "width": int(cp["width"]),
            "height": int(cp["height"]),
            "fx": float(cp["fx"]),
            "fy": float(cp["fy"]),
        })
    centers_arr = np.asarray(centers, dtype=np.float32)
    meta.update({
        "num_camera_frustums": int(len(frustums)),
        "camera_names": names,
        "camera_centers_bbox_min": centers_arr.min(axis=0).astype(float).tolist(),
        "camera_centers_bbox_max": centers_arr.max(axis=0).astype(float).tolist(),
        "per_camera": per_camera,
    })
    return frustums, names, meta, centers_arr


def _axis_limits(points: np.ndarray, extra_lines: Optional[List[np.ndarray]] = None, extra_points: Optional[np.ndarray] = None, scope: str = "all") -> Tuple[np.ndarray, float, Dict]:
    chunks = []
    flat = points.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(axis=1)]
    scope = str(scope).lower()
    if scope not in {"object", "cameras", "all"}:
        scope = "all"
    if flat.size and scope in {"object", "all"}:
        chunks.append(flat)
    if extra_lines and scope in {"cameras", "all"}:
        extra = np.vstack(extra_lines).reshape(-1, 3)
        extra = extra[np.isfinite(extra).all(axis=1)]
        if extra.size:
            chunks.append(extra)
    if extra_points is not None and np.asarray(extra_points).size and scope in {"cameras", "all"}:
        ep = np.asarray(extra_points, dtype=np.float64).reshape(-1, 3)
        ep = ep[np.isfinite(ep).all(axis=1)]
        if ep.size:
            chunks.append(ep)
    if not chunks:
        center = np.zeros(3, dtype=np.float64)
        return center, 1.0, {"axis_bbox_min": center.tolist(), "axis_bbox_max": center.tolist(), "axis_radius": 1.0}
    allp = np.vstack(chunks).astype(np.float64)
    lo = np.min(allp, axis=0)
    hi = np.max(allp, axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.max(hi - lo) * 0.525 + 1e-6)
    return center, radius, {"axis_bbox_min": lo.tolist(), "axis_bbox_max": hi.tolist(), "axis_radius": radius}


def _point_trace(pos_frame: np.ndarray, labels: np.ndarray, want_dynamic: bool, name: str, color: str, opacity: float):
    import plotly.graph_objects as go
    mask = labels if want_dynamic else ~labels
    pts = pos_frame[mask]
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker={"size": 2.2 if want_dynamic else 1.8, "color": color, "opacity": opacity},
        name=name,
    )


def write_track_html(
    tracks: Dict,
    out_html: str,
    cameras_json: str = "",
    max_points: int = 30000,
    dynamic_threshold: float = 0.5,
    frame_stride: int = 1,
    title: str = "Dynamic Gaussian Tracks",
    seed: int = 0,
    label_mode: str = "auto",
    speed_percentile: float = 80.0,
    frustum_scale: float = 0.12,
    camera_deduplicate: bool = True,
    aspectmode: str = "cube",
    axis_scope: str = "all",
) -> Dict:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise ImportError("Plotly is required for track HTML. Install with: pip install plotly") from exc

    out_html = Path(out_html).expanduser()
    out_html.parent.mkdir(parents=True, exist_ok=True)

    positions_all = _to_numpy(tracks["positions"]).astype(np.float32)
    frame_ids_all = _frame_ids(tracks)
    labels_all, label_meta = _compute_dynamic_labels(tracks, dynamic_threshold, label_mode, speed_percentile)
    sample_idx = _sample_indices(labels_all, max_points=max_points, seed=seed)
    labels = labels_all[sample_idx]
    positions = positions_all[:, sample_idx, :]

    frame_stride = max(1, int(frame_stride))
    frame_sel = np.arange(0, positions.shape[0], frame_stride, dtype=np.int64)
    if frame_sel.size == 0:
        frame_sel = np.array([0], dtype=np.int64)

    frustums, cam_names, camera_meta, cam_centers = load_camera_frustums(
        cameras_json, frustum_scale=float(frustum_scale), deduplicate=bool(camera_deduplicate)
    )
    aspectmode = str(aspectmode).lower()
    if aspectmode not in {"auto", "cube", "data"}:
        aspectmode = "cube"
    axis_scope = str(axis_scope).lower()
    if axis_scope not in {"object", "cameras", "all"}:
        axis_scope = "all"
    center, radius, axis_meta = _axis_limits(positions, frustums, cam_centers, scope=axis_scope)

    first = int(frame_sel[0])
    traces = [
        _point_trace(positions[first], labels, False, "static / low-motion Gaussians", "#9ca3af", 0.55),
        _point_trace(positions[first], labels, True, "dynamic / high-motion Gaussians", "#ff3b30", 0.9),
    ]

    for i, (line, name) in enumerate(zip(frustums, cam_names)):
        color = CAMERA_COLORS[i % len(CAMERA_COLORS)]
        traces.append(go.Scatter3d(
            x=line[:, 0], y=line[:, 1], z=line[:, 2],
            mode="lines",
            line={"width": 4, "color": color},
            opacity=0.65,
            name=f"{name} frustum",
            hoverinfo="skip",
        ))
        c = cam_centers[i]
        traces.append(go.Scatter3d(
            x=[float(c[0])], y=[float(c[1])], z=[float(c[2])],
            mode="markers+text",
            marker={"size": 5, "color": color},
            text=[str(name)],
            textposition="top center",
            name=f"{name} center",
        ))

    frames = []
    for fi in frame_sel:
        fid = int(frame_ids_all[int(fi)])
        frames.append(go.Frame(
            name=str(fid),
            data=[
                _point_trace(positions[int(fi)], labels, False, "static / low-motion Gaussians", "#9ca3af", 0.55),
                _point_trace(positions[int(fi)], labels, True, "dynamic / high-motion Gaussians", "#ff3b30", 0.9),
            ],
            traces=[0, 1],
        ))

    slider_steps = [{
        "args": [[str(int(frame_ids_all[int(fi)]))], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
        "label": str(int(frame_ids_all[int(fi)])),
        "method": "animate",
    } for fi in frame_sel]

    layout = go.Layout(
        title=f"{title}<br><sup>{len(sample_idx)} / {labels_all.shape[0]} Gaussians shown. Labels: {label_meta.get('source')}. Cameras: real world coordinates.</sup>",
        scene=dict(
            xaxis=dict(title="x [world]", range=[center[0] - radius, center[0] + radius]),
            yaxis=dict(title="y [world]", range=[center[1] - radius, center[1] + radius]),
            zaxis=dict(title="z [world]", range=[center[2] - radius, center[2] + radius]),
            aspectmode=aspectmode,
        ),
        legend=dict(itemsizing="constant"),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{"steps": slider_steps, "currentvalue": {"prefix": "frame "}}],
        margin=dict(l=0, r=0, b=0, t=70),
    )
    fig = go.Figure(data=traces, layout=layout, frames=frames)
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)

    summary = {
        "html": str(out_html),
        "num_frames": int(positions_all.shape[0]),
        "num_gaussians_total": int(labels_all.shape[0]),
        "num_gaussians_shown": int(len(sample_idx)),
        "num_dynamic_shown": int(labels.sum()),
        "num_static_shown": int((~labels).sum()),
        "frame_stride": int(frame_stride),
        "label_meta": label_meta,
        "used_cameras_json": bool(cameras_json),
        "num_camera_frustums": int(len(frustums)),
        "camera_meta": camera_meta,
        "axis_meta": axis_meta,
        "metric_scene": True,
        "aspectmode": aspectmode,
        "axis_scope": axis_scope,
    }
    with open(out_html.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Dynamic 3DGS] Wrote track HTML: {out_html}")
    return summary


def write_track_html_from_file(
    tracks_path: str,
    out_html: str,
    cameras_json: str = "",
    max_points: int = 30000,
    dynamic_threshold: float = 0.5,
    frame_stride: int = 1,
    title: str = "Dynamic Gaussian Tracks",
    seed: int = 0,
    label_mode: str = "auto",
    speed_percentile: float = 80.0,
    frustum_scale: float = 0.12,
    camera_deduplicate: bool = True,
    aspectmode: str = "cube",
    axis_scope: str = "all",
) -> Dict:
    tracks = torch.load(str(Path(tracks_path).expanduser()), map_location="cpu")
    return write_track_html(
        tracks=tracks,
        out_html=out_html,
        cameras_json=cameras_json,
        max_points=max_points,
        dynamic_threshold=dynamic_threshold,
        frame_stride=frame_stride,
        title=title,
        seed=seed,
        label_mode=label_mode,
        speed_percentile=speed_percentile,
        frustum_scale=frustum_scale,
        camera_deduplicate=camera_deduplicate,
        aspectmode=aspectmode,
        axis_scope=axis_scope,
    )


def main() -> None:
    parser = ArgumentParser(description="Create metric Plotly HTML from gaussian_tracks.pt")
    parser.add_argument("--tracks_path", required=True, type=str)
    parser.add_argument("--out_html", required=True, type=str)
    parser.add_argument("--cameras_json", default="", type=str)
    parser.add_argument("--max_points", default=30000, type=int)
    parser.add_argument("--dynamic_threshold", default=0.5, type=float)
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--title", default="Dynamic Gaussian Tracks", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--label_mode", default="auto", choices=["auto", "prob", "speed", "combined"])
    parser.add_argument("--speed_percentile", default=80.0, type=float)
    parser.add_argument("--frustum_scale", default=0.12, type=float, help="Camera frustum depth in world units/meters. Camera centers are not rescaled.")
    parser.add_argument("--aspectmode", default="cube", choices=["cube", "data", "auto"], help="Plotly 3D aspect mode. cube matches the older debug HTML; data preserves metric visual aspect but shallow scenes look flat.")
    parser.add_argument("--axis_scope", default="all", choices=["object", "cameras", "all"], help="Which coordinates define the initial axis range. Coordinates are never rescaled.")
    parser.add_argument("--no_camera_deduplicate", action="store_true", default=False)
    args = parser.parse_args()
    write_track_html_from_file(
        tracks_path=args.tracks_path,
        out_html=args.out_html,
        cameras_json=args.cameras_json,
        max_points=args.max_points,
        dynamic_threshold=args.dynamic_threshold,
        frame_stride=args.frame_stride,
        title=args.title,
        seed=args.seed,
        label_mode=args.label_mode,
        speed_percentile=args.speed_percentile,
        frustum_scale=args.frustum_scale,
        camera_deduplicate=not bool(args.no_camera_deduplicate),
        aspectmode=args.aspectmode,
        axis_scope=args.axis_scope,
    )


if __name__ == "__main__":
    main()
