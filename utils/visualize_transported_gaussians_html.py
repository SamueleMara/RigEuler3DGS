"""Animated transported-Gaussian diagnostic for free and exact-joint phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from utils.html_plot_layout import (
    fixed_scene_bounds,
    left_colorbar,
    plotly_3d_config,
    plotly_3d_layout,
)


_PART_COLORS = (
    "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#17becf", "#bcbd22",
)


def _label_name(label):
    if int(label) < 0:
        return "ambiguous/background identity (-1)"
    if int(label) == 0:
        return "foreground static / identity (0)"
    return f"moving material part {int(label)}"


def _label_color(label):
    if int(label) < 0:
        return "#a0a0a0"
    if int(label) == 0:
        return "#377eb8"
    return _PART_COLORS[(int(label) - 1) % len(_PART_COLORS)]


_CAMERA_COLORS = (
    "#ef5350", "#2a80a5", "#7cb65a", "#f39c25",
    "#557590", "#41a689", "#f26b2b", "#4e9291",
)


def _sample_ids(labels, max_points):
    count = labels.numel()
    if count <= int(max_points):
        return torch.arange(count)
    moving = torch.nonzero(labels > 0, as_tuple=False).flatten()
    remaining = torch.nonzero(labels <= 0, as_tuple=False).flatten()
    if moving.numel() >= int(max_points):
        select = torch.linspace(0, moving.numel() - 1, int(max_points)).round().long().unique()
        return moving[select]
    budget = int(max_points) - moving.numel()
    select = torch.linspace(0, remaining.numel() - 1, budget).round().long().unique()
    return torch.cat((moving, remaining[select])).sort().values


def _transport_lines(position, velocity, ids, scale):
    if ids.numel() == 0:
        return [], [], []
    start = position[ids]
    end = start + float(scale) * velocity[ids]
    x, y, z = [], [], []
    for source, destination in zip(start.tolist(), end.tolist()):
        x.extend((source[0], destination[0], None))
        y.extend((source[1], destination[1], None))
        z.extend((source[2], destination[2], None))
    return x, y, z


def _camera_traces(camera, color, depth):
    center = np.asarray(camera["center"], dtype=np.float64)
    rotation = np.asarray(camera["r_cw"], dtype=np.float64)
    corners_pixel = (
        (0.0, 0.0), (float(camera["width"]), 0.0),
        (float(camera["width"]), float(camera["height"])),
        (0.0, float(camera["height"])),
    )
    corners_camera = np.asarray(
        [
            [
                (u - float(camera["cx"])) / float(camera["fx"]) * depth,
                (v - float(camera["cy"])) / float(camera["fy"]) * depth,
                depth,
            ]
            for u, v in corners_pixel
        ],
        dtype=np.float64,
    )
    corners = corners_camera @ rotation.T + center[None, :]
    segments = []
    for corner in corners:
        segments.append((center, corner))
    for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
        segments.append((corners[first], corners[second]))
    coordinates = [[], [], []]
    for start, end in segments:
        for axis in range(3):
            coordinates[axis].extend((float(start[axis]), float(end[axis]), None))
    name = str(camera["name"])
    traces = [
        {
            "type": "scatter3d", "mode": "lines",
            "x": coordinates[0], "y": coordinates[1], "z": coordinates[2],
            "line": {"width": 4, "color": color},
            "name": f"{name} frustum", "showlegend": True, "hoverinfo": "skip",
        },
        {
            "type": "scatter3d", "mode": "markers+text",
            "x": [float(center[0])], "y": [float(center[1])], "z": [float(center[2])],
            "text": [name], "textposition": "top center",
            "marker": {"size": 5, "color": color},
            "name": f"{name} center", "showlegend": True,
        },
    ]
    return traces, np.concatenate((center[None, :], corners), axis=0)


@torch.no_grad()
def write_transported_gaussians_html(
    tracks,
    out_html,
    *,
    title="Transported Gaussians",
    semantic_threshold=0.5,
    max_points=1000,
    max_vectors=250,
):
    positions = tracks["positions"].detach().cpu().float()
    velocities = tracks["velocities"].detach().cpu().float()
    rotations = tracks.get("rotations")
    rotations = None if rotations is None else rotations.detach().cpu().float()
    times = tracks["times"].detach().cpu().float()
    canonical = tracks.get("canonical_xyz", positions[0]).detach().cpu().float()
    membership_state = tracks.get("motion_membership")
    membership = None if membership_state is None else membership_state.detach().cpu().float().reshape(-1)
    part_labels = tracks.get("part_labels")
    if part_labels is not None:
        labels = part_labels.detach().cpu().long().reshape(-1)
    else:
        dynamic = (
            tracks["initial_dynamic_mask"].detach().cpu().bool().reshape(-1)
            if membership is None else membership >= 0.5
        )
        semantic = tracks.get("semantic_probability")
        foreground = (
            torch.ones_like(dynamic)
            if semantic is None
            else semantic.detach().cpu().reshape(-1) > float(semantic_threshold)
        )
        labels = torch.full_like(dynamic, -1, dtype=torch.long)
        labels[foreground] = 0
        labels[dynamic & foreground] = 1
    if labels.numel() != positions.shape[1]:
        raise ValueError("Transport labels must match the Gaussian count")

    ids = _sample_ids(labels, max_points=int(max_points))
    positions = positions[:, ids]
    velocities = velocities[:, ids]
    if rotations is not None:
        rotations = torch.nn.functional.normalize(rotations[:, ids], dim=-1)
    canonical = canonical[ids]
    labels = labels[ids]
    if membership is None:
        membership = (labels > 0).float()
    else:
        membership = membership[ids]
    gaussian_ids = ids.tolist()
    speed = torch.linalg.norm(velocities, dim=-1)
    displacement = torch.linalg.norm(positions - canonical[None], dim=-1)
    orientation_delta = None
    if rotations is not None:
        quaternion_alignment = torch.abs(
            (rotations * rotations[:1]).sum(dim=-1)
        ).clamp(0.0, 1.0)
        orientation_delta = 2.0 * torch.acos(quaternion_alignment) * (180.0 / np.pi)
    positive_speed = speed[speed > 0]
    speed_reference = (
        float(torch.quantile(positive_speed, 0.95)) if positive_speed.numel() else 1.0
    )
    extent = float((canonical.amax(dim=0) - canonical.amin(dim=0)).norm().clamp_min(1.0e-6))
    arrow_scale = 0.12 * extent / max(speed_reference, 1.0e-8)
    moving_ids = torch.nonzero(labels > 0, as_tuple=False).flatten()
    if moving_ids.numel() > int(max_vectors):
        score = speed.mean(dim=0)[moving_ids]
        moving_ids = moving_ids[torch.topk(score, k=int(max_vectors), largest=True).indices]
    label_values = sorted(int(value) for value in labels.unique().tolist())

    frames = []
    for frame_index, time in enumerate(times.tolist()):
        xyz = positions[frame_index]
        vector_x, vector_y, vector_z = _transport_lines(
            xyz, velocities[frame_index], moving_ids, arrow_scale
        )
        hover = [
            (
                f"gaussian={gaussian_ids[index]}<br>{_label_name(int(labels[index]))}"
                f"<br>frame={frame_index}<br>time={time:.6f}"
                f"<br>|v|={float(speed[frame_index,index]):.6g}"
                f"<br>soft pi={float(membership[index]):.6g}"
                f"<br>|x(t)-x(0)|={float(displacement[frame_index,index]):.6g}"
                + (
                    "" if orientation_delta is None
                    else f"<br>orientation delta={float(orientation_delta[frame_index,index]):.6g} deg"
                )
            )
            for index in range(labels.numel())
        ]
        frame_traces = [
            {
                "type": "scatter3d", "mode": "markers",
                "x": xyz[:, 0].tolist(), "y": xyz[:, 1].tolist(), "z": xyz[:, 2].tolist(),
                "text": hover, "hoverinfo": "text",
                "marker": {
                    "size": 2.4,
                    "color": speed[frame_index].tolist(),
                    "colorscale": "Turbo", "cmin": 0.0, "cmax": speed_reference,
                    "colorbar": left_colorbar("speed"), "opacity": 0.82,
                },
                "name": "transported Gaussian speed", "showlegend": True,
            },
            {
                "type": "scatter3d", "mode": "lines",
                "x": vector_x, "y": vector_y, "z": vector_z,
                "line": {"width": 3, "color": "rgba(35,35,35,0.52)"},
                "name": "sampled velocity vectors", "showlegend": True,
                "hoverinfo": "skip",
            },
            {
                "type": "scatter3d", "mode": "markers", "visible": "legendonly",
                "x": xyz[:, 0].tolist(), "y": xyz[:, 1].tolist(), "z": xyz[:, 2].tolist(),
                "text": hover, "hoverinfo": "text",
                "marker": {
                    "size": 2.6, "color": membership.tolist(), "colorscale": "Viridis",
                    "cmin": 0.0, "cmax": 1.0,
                    "colorbar": left_colorbar("soft motion pi"), "opacity": 0.86,
                },
                "name": "soft motion responsibility pi", "showlegend": True,
            },
        ]
        xyz_np = xyz.numpy()
        labels_np = labels.numpy()
        for label in label_values:
            mask = labels_np == label
            frame_traces.append(
                {
                    "type": "scatter3d", "mode": "markers",
                    "x": xyz_np[mask, 0].tolist(), "y": xyz_np[mask, 1].tolist(),
                    "z": xyz_np[mask, 2].tolist(),
                    "marker": {"size": 2.8, "color": _label_color(label), "opacity": 0.88},
                    "name": _label_name(label), "showlegend": True,
                }
            )
        frames.append(
            {
                "name": str(frame_index),
                "data": frame_traces,
                "traces": list(range(len(frame_traces))),
            }
        )

    initial = [dict(trace) for trace in frames[0]["data"]] if frames else []
    for trace in initial[3:]:
        trace["visible"] = "legendonly"
    canonical_trace = {
        "type": "scatter3d", "mode": "markers", "visible": "legendonly",
        "x": canonical[:, 0].tolist(), "y": canonical[:, 1].tolist(), "z": canonical[:, 2].tolist(),
        "marker": {"size": 2.0, "color": "rgba(25,25,25,0.42)"},
        "name": "canonical x(0)", "showlegend": True,
    }
    fixed_cameras = tracks.get("fixed_cameras", []) or []
    camera_traces = []
    camera_world_points = []
    frustum_depth = max(0.18 * extent, 1.0e-4)
    for camera_index, camera in enumerate(fixed_cameras):
        traces, points = _camera_traces(
            camera,
            _CAMERA_COLORS[camera_index % len(_CAMERA_COLORS)],
            frustum_depth,
        )
        camera_traces.extend(traces)
        camera_world_points.append(points)
    scene_bounds = fixed_scene_bounds(
        positions,
        canonical,
        None if not camera_world_points else np.concatenate(camera_world_points, axis=0),
    )
    steps = [
        {
            "label": str(index), "method": "animate",
            "args": [[str(index)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
        }
        for index in range(len(frames))
    ]
    sliders = [{"active": 0, "currentvalue": {"prefix": "frame: "}, "steps": steps}]
    buttons = [{"type": "buttons", "showactive": False, "buttons": [
        {"label": "Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 180, "redraw": True}, "transition": {"duration": 0}}]},
        {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0}}]},
    ]}]
    subtitle = (
        f"{title}<br><sup>{positions.shape[1]} / {tracks['positions'].shape[1]} Gaussians; "
        f"soft pi shown continuously; part labels are hard only in Phase 2; "
        f"{len(fixed_cameras)} fixed camera frustums; arrows are display-scaled</sup>"
    )
    payload = {
        "data": [*initial, canonical_trace, *camera_traces],
        "frames": frames,
        "layout": plotly_3d_layout(
            subtitle, sliders=sliders, buttons=buttons,
            scene_bounds=scene_bounds, aspectmode="cube",
        ),
    }
    output = Path(out_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head>"
        "<body style='margin:0'><div id='plot' style='width:100vw;height:100vh'></div>"
        f"<script>const p={json.dumps(payload)};const config={json.dumps(plotly_3d_config())};"
        "Plotly.newPlot('plot',p.data,p.layout,config).then(()=>Plotly.addFrames('plot',p.frames));"
        "</script></body></html>",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Transported Gaussians")
    parser.add_argument("--semantic_threshold", type=float, default=0.5)
    parser.add_argument("--max_points", type=int, default=1000)
    parser.add_argument("--max_vectors", type=int, default=250)
    cli = parser.parse_args()
    write_transported_gaussians_html(
        torch.load(cli.tracks, map_location="cpu", weights_only=False),
        cli.output,
        title=cli.title,
        semantic_threshold=cli.semantic_threshold,
        max_points=cli.max_points,
        max_vectors=cli.max_vectors,
    )
