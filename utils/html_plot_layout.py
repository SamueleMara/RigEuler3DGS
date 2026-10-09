"""Shared Plotly layout for dynamic 3D diagnostics."""

from __future__ import annotations

import numpy as np


def fixed_scene_bounds(*point_sets, padding=0.05, minimum_radius=1.0e-4):
    """One immutable cubic world-space range computed from every animation frame."""
    finite_chunks = []
    for values in point_sets:
        if values is None:
            continue
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            continue
        array = array.reshape(-1, 3)
        array = array[np.isfinite(array).all(axis=1)]
        if array.size:
            finite_chunks.append(array)
    if not finite_chunks:
        center = np.zeros(3, dtype=np.float64)
        radius = 1.0
    else:
        points = np.concatenate(finite_chunks, axis=0)
        lower, upper = points.min(axis=0), points.max(axis=0)
        center = 0.5 * (lower + upper)
        radius = max(
            0.5 * float(np.max(upper - lower)) * (1.0 + float(padding)),
            float(minimum_radius),
        )
    return {
        "center": center.tolist(),
        "radius": float(radius),
        "x": [float(center[0] - radius), float(center[0] + radius)],
        "y": [float(center[1] - radius), float(center[1] + radius)],
        "z": [float(center[2] - radius), float(center[2] + radius)],
    }


def left_colorbar(title):
    """Color scale in the reserved strip left of the 3D scene."""
    return {
        "title": str(title),
        "x": 0.075,
        "xanchor": "right",
        "y": 0.53,
        "len": 0.68,
        "thickness": 18,
    }


def plotly_3d_layout(
    title, *, sliders=None, buttons=None, scene_bounds=None, aspectmode="cube"
):
    """Stable orbitable scene: fixed world bounds, left scale, right legend."""
    bounds = scene_bounds or fixed_scene_bounds()
    layout = {
        "title": {"text": str(title), "x": 0.05, "xanchor": "left"},
        "uirevision": "fixed-world-camera-v1",
        "scene": {
            "domain": {"x": [0.12, 0.78], "y": [0.08, 1.0]},
            "uirevision": "fixed-world-camera-v1",
            "dragmode": "orbit",
            "aspectmode": str(aspectmode),
            "xaxis": {"title": "x [world]", "range": bounds["x"], "autorange": False},
            "yaxis": {"title": "y [world]", "range": bounds["y"], "autorange": False},
            "zaxis": {"title": "z [world]", "range": bounds["z"], "autorange": False},
        },
        "margin": {"l": 70, "r": 310, "b": 80, "t": 70},
        "legend": {
            "x": 0.82,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": "rgba(80,80,80,0.35)",
            "borderwidth": 1,
        },
    }
    if sliders:
        for slider in sliders:
            slider.setdefault("x", 0.08)
            slider.setdefault("len", 0.70)
            slider.setdefault("y", 0.02)
        layout["sliders"] = sliders
    if buttons:
        for menu in buttons:
            menu.setdefault("x", 0.01)
            menu.setdefault("y", 0.98)
        layout["updatemenus"] = buttons
    return layout


def plotly_3d_config():
    """Explicitly expose mouse orbit, wheel zoom and the standard 3D modebar."""
    return {
        "responsive": True,
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
    }
