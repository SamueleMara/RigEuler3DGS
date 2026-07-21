"""Metric Plotly HTML visualization of exported Gaussian velocity field.

Everything is drawn in real world coordinates:
  - positions from gaussian_tracks.pt are unchanged;
  - camera centers are unchanged;
  - frustum depth is the only user-chosen display length, in world units;
  - velocity arrows represent displacement over one exported time step by default.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from utils.visualize_gaussian_tracks_html import load_camera_frustums, CAMERA_COLORS
except Exception:  # standalone copy fallback
    from visualize_gaussian_tracks_html import load_camera_frustums, CAMERA_COLORS


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _frame_ids(tracks: Dict) -> np.ndarray:
    if "frame_ids" in tracks and tracks["frame_ids"] is not None:
        return _to_numpy(tracks["frame_ids"]).astype(np.int64)
    return np.arange(int(tracks["positions"].shape[0]), dtype=np.int64)


def _times(tracks: Dict) -> np.ndarray | None:
    if "times" in tracks and tracks["times"] is not None:
        t = _to_numpy(tracks["times"]).astype(np.float64).reshape(-1)
        if t.size == int(tracks["positions"].shape[0]):
            return t
    return None


def _axis_limits(points: np.ndarray, extra_lines: List[np.ndarray] | None = None, extra_points: np.ndarray | None = None, scope: str = "all") -> Tuple[np.ndarray, float, Dict]:
    chunks = []
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if p.size:
        chunks.append(p)
    if extra_lines:
        e = np.vstack(extra_lines).reshape(-1, 3)
        e = e[np.isfinite(e).all(axis=1)]
        if e.size:
            chunks.append(e)
    if extra_points is not None and np.asarray(extra_points).size:
        c = np.asarray(extra_points, dtype=np.float64).reshape(-1, 3)
        c = c[np.isfinite(c).all(axis=1)]
        if c.size:
            chunks.append(c)
    if not chunks:
        return np.zeros(3), 1.0, {"axis_bbox_min": [0, 0, 0], "axis_bbox_max": [0, 0, 0], "axis_radius": 1.0}
    allp = np.vstack(chunks)
    lo = np.min(allp, axis=0)
    hi = np.max(allp, axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.max(hi - lo) * 0.525 + 1e-6)
    return center, radius, {"axis_bbox_min": lo.tolist(), "axis_bbox_max": hi.tolist(), "axis_radius": radius}


def _sample_by_speed(speed_max: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    n = int(speed_max.shape[0])
    if n <= int(max_points):
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    fast_budget = int(max_points) // 2
    ctx_budget = int(max_points) - fast_budget
    fast = np.argsort(speed_max)[-fast_budget:]
    rest = np.setdiff1d(np.arange(n), fast, assume_unique=False)
    if rest.size > ctx_budget:
        rest = rng.choice(rest, size=ctx_budget, replace=False)
    idx = np.concatenate([fast, rest])
    rng.shuffle(idx)
    return idx.astype(np.int64)


def _default_arrow_dt(tracks: Dict) -> float:
    t = _times(tracks)
    if t is not None and t.size > 1:
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (np.abs(dt) > 1e-12)]
        if dt.size:
            return float(np.median(np.abs(dt)))
    return 1.0


def _arrow_segments(pos: np.ndarray, vel: np.ndarray, speed: np.ndarray, max_arrows: int, arrow_dt: float) -> np.ndarray:
    n = int(pos.shape[0])
    if n == 0 or int(max_arrows) <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    idx = np.argsort(speed)[-int(max_arrows):] if n > int(max_arrows) else np.arange(n)
    p = pos[idx]
    v = vel[idx]
    end = p + v * float(arrow_dt)
    nan = np.full((len(idx), 1, 3), np.nan, dtype=np.float32)
    return np.concatenate([p[:, None, :], end[:, None, :], nan], axis=1).reshape(-1, 3).astype(np.float32)


def write_velocity_field_html(
    tracks: Dict,
    out_html: str,
    cameras_json: str = "",
    max_points: int = 30000,
    max_arrows: int = 3000,
    frame_stride: int = 1,
    speed_clip_percentile: float = 98.0,
    arrow_scale: float = 0.0,
    title: str = "Dynamic Eulerian 3DGS Velocity Field",
    seed: int = 0,
    frustum_scale: float = 0.12,
    camera_deduplicate: bool = True,
    aspectmode: str = "cube",
    axis_scope: str = "all",
) -> Dict:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise ImportError("Plotly is required for velocity-field HTML. Install with: pip install plotly") from exc

    out_html = Path(out_html).expanduser()
    out_html.parent.mkdir(parents=True, exist_ok=True)

    positions_all = _to_numpy(tracks["positions"]).astype(np.float32)
    velocities_all = _to_numpy(tracks.get("velocities", np.zeros_like(positions_all))).astype(np.float32)
    if velocities_all.shape != positions_all.shape:
        velocities_all = np.zeros_like(positions_all)
    frame_ids_all = _frame_ids(tracks)

    speed_all = np.linalg.norm(velocities_all, axis=-1)
    speed_max = speed_all.max(axis=0)
    sample_idx = _sample_by_speed(speed_max, max_points=max_points, seed=seed)
    positions = positions_all[:, sample_idx, :]
    velocities = velocities_all[:, sample_idx, :]
    speed = speed_all[:, sample_idx]

    frame_stride = max(1, int(frame_stride))
    frame_sel = np.arange(0, positions.shape[0], frame_stride, dtype=np.int64)
    if frame_sel.size == 0:
        frame_sel = np.array([0], dtype=np.int64)

    clip = np.percentile(speed[np.isfinite(speed)], float(speed_clip_percentile)) if np.isfinite(speed).any() else 1.0
    if not np.isfinite(clip) or clip <= 0:
        clip = 1.0
    arrow_dt = float(arrow_scale) if float(arrow_scale) > 0 else _default_arrow_dt(tracks)

    frustums, cam_names, camera_meta, cam_centers = load_camera_frustums(
        cameras_json, frustum_scale=float(frustum_scale), deduplicate=bool(camera_deduplicate)
    )
    # Include arrow endpoints from first frame in limits as well as camera geometry.
    first = int(frame_sel[0])
    arrows0 = _arrow_segments(positions[first], velocities[first], speed[first], max_arrows=max_arrows, arrow_dt=arrow_dt)
    extra_lines = list(frustums)
    if arrows0.size:
        extra_lines.append(arrows0)
    aspectmode = str(aspectmode).lower()
    if aspectmode not in {"auto", "cube", "data"}:
        aspectmode = "cube"
    axis_scope = str(axis_scope).lower()
    if axis_scope not in {"object", "cameras", "all"}:
        axis_scope = "all"
    center, radius, axis_meta = _axis_limits(positions, extra_lines, cam_centers, scope=axis_scope)

    traces = [
        go.Scatter3d(
            x=positions[first, :, 0], y=positions[first, :, 1], z=positions[first, :, 2],
            mode="markers",
            marker={
                "size": 2.2,
                "color": np.clip(speed[first], 0.0, clip),
                "colorscale": "Turbo",
                "cmin": 0.0,
                "cmax": float(clip),
                "colorbar": {"title": "speed"},
                "opacity": 0.85,
            },
            name="Gaussian speed heatmap",
        ),
        go.Scatter3d(
            x=arrows0[:, 0] if arrows0.size else [],
            y=arrows0[:, 1] if arrows0.size else [],
            z=arrows0[:, 2] if arrows0.size else [],
            mode="lines",
            line={"width": 2, "color": "#111827"},
            opacity=0.7,
            name=f"velocity vectors over dt={arrow_dt:.4g}",
            hoverinfo="skip",
        ),
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
        arr = _arrow_segments(positions[int(fi)], velocities[int(fi)], speed[int(fi)], max_arrows=max_arrows, arrow_dt=arrow_dt)
        frames.append(go.Frame(
            name=str(fid),
            data=[
                go.Scatter3d(
                    x=positions[int(fi), :, 0], y=positions[int(fi), :, 1], z=positions[int(fi), :, 2],
                    mode="markers",
                    marker={"size": 2.2, "color": np.clip(speed[int(fi)], 0.0, clip), "colorscale": "Turbo", "cmin": 0.0, "cmax": float(clip), "opacity": 0.85},
                    name="Gaussian speed heatmap",
                ),
                go.Scatter3d(
                    x=arr[:, 0] if arr.size else [], y=arr[:, 1] if arr.size else [], z=arr[:, 2] if arr.size else [],
                    mode="lines", line={"width": 2, "color": "#111827"}, opacity=0.7, name=f"velocity vectors over dt={arrow_dt:.4g}", hoverinfo="skip",
                ),
            ],
            traces=[0, 1],
        ))

    slider_steps = [{
        "args": [[str(int(frame_ids_all[int(fi)]))], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
        "label": str(int(frame_ids_all[int(fi)])),
        "method": "animate",
    } for fi in frame_sel]

    layout = go.Layout(
        title=f"{title}<br><sup>{len(sample_idx)} / {positions_all.shape[1]} Gaussians sampled. Heatmap = speed. Cameras: real world coordinates.</sup>",
        scene=dict(
            xaxis=dict(title="x [world]", range=[center[0] - radius, center[0] + radius]),
            yaxis=dict(title="y [world]", range=[center[1] - radius, center[1] + radius]),
            zaxis=dict(title="z [world]", range=[center[2] - radius, center[2] + radius]),
            aspectmode=aspectmode,
        ),
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
        "num_gaussians_total": int(positions_all.shape[1]),
        "num_gaussians_shown": int(len(sample_idx)),
        "frame_stride": int(frame_stride),
        "speed_clip_percentile": float(speed_clip_percentile),
        "speed_clip_value": float(clip),
        "arrow_dt_world_time": float(arrow_dt),
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
    print(f"[Dynamic 3DGS] Wrote velocity-field HTML: {out_html}")
    return summary


def write_velocity_field_html_from_file(
    tracks_path: str,
    out_html: str,
    cameras_json: str = "",
    max_points: int = 30000,
    max_arrows: int = 3000,
    frame_stride: int = 1,
    speed_clip_percentile: float = 98.0,
    arrow_scale: float = 0.0,
    title: str = "Dynamic Eulerian 3DGS Velocity Field",
    seed: int = 0,
    frustum_scale: float = 0.12,
    camera_deduplicate: bool = True,
    aspectmode: str = "cube",
    axis_scope: str = "all",
) -> Dict:
    tracks = torch.load(str(Path(tracks_path).expanduser()), map_location="cpu")
    return write_velocity_field_html(
        tracks=tracks,
        out_html=out_html,
        cameras_json=cameras_json,
        max_points=max_points,
        max_arrows=max_arrows,
        frame_stride=frame_stride,
        speed_clip_percentile=speed_clip_percentile,
        arrow_scale=arrow_scale,
        title=title,
        seed=seed,
        frustum_scale=frustum_scale,
        camera_deduplicate=camera_deduplicate,
        aspectmode=aspectmode,
        axis_scope=axis_scope,
    )


def main() -> None:
    parser = ArgumentParser(description="Create metric 3D velocity-field Plotly HTML from gaussian_tracks.pt")
    parser.add_argument("--tracks_path", required=True, type=str)
    parser.add_argument("--out_html", required=True, type=str)
    parser.add_argument("--cameras_json", default="", type=str)
    parser.add_argument("--max_points", default=30000, type=int)
    parser.add_argument("--max_arrows", default=3000, type=int)
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--speed_clip_percentile", default=98.0, type=float)
    parser.add_argument("--arrow_scale", default=0.0, type=float, help="Velocity arrow duration. 0 = median exported time step.")
    parser.add_argument("--title", default="Dynamic Eulerian 3DGS Velocity Field", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--frustum_scale", default=0.12, type=float, help="Camera frustum depth in world units/meters. Camera centers are not rescaled.")
    parser.add_argument("--aspectmode", default="cube", choices=["cube", "data", "auto"], help="Plotly 3D aspect mode. cube matches the older debug HTML; data preserves metric visual aspect but shallow scenes look flat.")
    parser.add_argument("--axis_scope", default="all", choices=["object", "cameras", "all"], help="Which coordinates define the initial axis range. Coordinates are never rescaled.")
    parser.add_argument("--no_camera_deduplicate", action="store_true", default=False)
    args = parser.parse_args()
    write_velocity_field_html_from_file(
        tracks_path=args.tracks_path,
        out_html=args.out_html,
        cameras_json=args.cameras_json,
        max_points=args.max_points,
        max_arrows=args.max_arrows,
        frame_stride=args.frame_stride,
        speed_clip_percentile=args.speed_clip_percentile,
        arrow_scale=args.arrow_scale,
        title=args.title,
        seed=args.seed,
        frustum_scale=args.frustum_scale,
        camera_deduplicate=not bool(args.no_camera_deduplicate),
        aspectmode=args.aspectmode,
        axis_scope=args.axis_scope,
    )


if __name__ == "__main__":
    main()
