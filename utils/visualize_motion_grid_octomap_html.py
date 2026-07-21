"""Interactive octomap-style HTML visualization for the adaptive motion grid.

This module is intentionally independent from training losses. It visualizes the
current adaptive grid topology and the learned per-cell feature vectors saved in
`motion_grid_state.pt` or passed directly as a state dict.

Color convention:
    default `color_mode='feature_pca'`: RGB = PCA projection of motion-grid
    feature vectors. This gives an immediate view of feature-field structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_state_dict(state_or_path: Any) -> Dict[str, Any]:
    if isinstance(state_or_path, (str, Path)):
        return torch.load(str(state_or_path), map_location="cpu")
    if not isinstance(state_or_path, dict):
        raise TypeError(f"Expected motion grid state dict/path, got {type(state_or_path)!r}")
    return state_or_path


def _normalize01(values: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)
    if finite.sum() == 0:
        return out
    lo = np.percentile(values[finite], lo_pct)
    hi = np.percentile(values[finite], hi_pct)
    denom = max(float(hi - lo), 1e-8)
    out[finite] = np.clip((values[finite] - lo) / denom, 0.0, 1.0)
    return out


def _pca_rgb(features: np.ndarray) -> np.ndarray:
    """Project feature vectors to RGB via robust PCA normalization."""
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    X = X - X.mean(axis=0, keepdims=True)
    if X.shape[1] >= 3 and X.shape[0] >= 3:
        try:
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            comps = X @ vt[:3].T
        except np.linalg.LinAlgError:
            comps = X[:, :3]
    else:
        comps = X[:, : min(3, X.shape[1])]
    if comps.shape[1] < 3:
        comps = np.pad(comps, ((0, 0), (0, 3 - comps.shape[1])), mode="constant")
    rgb = np.stack([_normalize01(comps[:, c]) for c in range(3)], axis=1)
    return (255.0 * rgb).astype(np.uint8)


def _scalar_rgb(values: np.ndarray) -> np.ndarray:
    """Simple blue->cyan->yellow->red colormap without matplotlib dependency."""
    t = _normalize01(values)
    r = np.clip(1.5 * t - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * t - 1.0) * 1.5, 0.0, 1.0)
    b = np.clip(1.25 - 1.5 * t, 0.0, 1.0)
    return (255.0 * np.stack([r, g, b], axis=1)).astype(np.uint8)


def _get_debug_array(debug: Dict[str, Any], name: str, n_total: int, default: float = np.nan) -> np.ndarray:
    arr = _to_numpy(debug.get(name)) if isinstance(debug, dict) else None
    if arr is None:
        return np.full(n_total, default, dtype=np.float32)
    arr = np.asarray(arr).reshape(-1)
    if arr.shape[0] < n_total:
        pad = np.full(n_total - arr.shape[0], default, dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:n_total]


def _make_cube_mesh(centers: np.ndarray, sizes: np.ndarray, colors_rgb: np.ndarray, alpha: int):
    verts = []
    ii, jj, kk = [], [], []
    vertexcolor = []
    tri = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    for center, size, color in zip(centers, sizes, colors_rgb):
        cx, cy, cz = [float(v) for v in center]
        h = float(size) * 0.5
        base = len(verts)
        cube = np.array(
            [
                [cx - h, cy - h, cz - h],
                [cx + h, cy - h, cz - h],
                [cx + h, cy + h, cz - h],
                [cx - h, cy + h, cz - h],
                [cx - h, cy - h, cz + h],
                [cx + h, cy - h, cz + h],
                [cx + h, cy + h, cz + h],
                [cx - h, cy + h, cz + h],
            ],
            dtype=np.float32,
        )
        verts.extend(cube.tolist())
        rgba = [int(color[0]), int(color[1]), int(color[2]), int(alpha)]
        vertexcolor.extend([rgba] * 8)
        tri_off = tri + base
        ii.extend(tri_off[:, 0].tolist())
        jj.extend(tri_off[:, 1].tolist())
        kk.extend(tri_off[:, 2].tolist())
    return np.asarray(verts, dtype=np.float32), np.asarray(ii), np.asarray(jj), np.asarray(kk), vertexcolor


def write_motion_grid_octomap_html(
    grid_state: Dict[str, Any],
    out_html: str | Path,
    *,
    title: str = "Adaptive Motion Grid",
    color_mode: str = "feature_pca",
    max_cells: int = 5000,
    active_only: bool = True,
    min_count: int = 1,
    opacity: float = 0.24,
    include_plotlyjs: str = "cdn",
) -> Dict[str, Any]:
    """Write octomap-style adaptive-grid visualization.

    Args:
        grid_state: state dict produced by `_build_grid_debug_from_tracks`.
        out_html: output HTML path.
        color_mode: one of feature_pca, feature_norm, dynamic_prob, semantic_prob,
            mean_speed, max_speed, max_displacement, score_ema, level, count.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    centers = _to_numpy(grid_state["cell_centers"]).astype(np.float32)
    levels = _to_numpy(grid_state["cell_level"]).astype(np.int64)
    active_mask = _to_numpy(grid_state.get("active_mask"))
    if active_mask is None:
        active_mask = np.ones(len(centers), dtype=bool)
    active_mask = active_mask.astype(bool)
    features = _to_numpy(grid_state["features"]).astype(np.float32)
    n_total = int(len(centers))
    voxel_size = float(grid_state.get("voxel_size", 1.0))
    debug = grid_state.get("per_cell_debug", {}) or {}

    counts = _to_numpy(grid_state.get("per_cell_count"))
    if counts is None:
        counts = _get_debug_array(debug, "count", n_total, default=1.0)
    counts = counts.reshape(-1)
    if counts.shape[0] < n_total:
        counts = np.pad(counts, (0, n_total - counts.shape[0]), constant_values=0)
    counts = counts[:n_total]

    keep = np.ones(n_total, dtype=bool)
    if active_only:
        keep &= active_mask
    keep &= counts >= int(min_count)
    ids = np.where(keep)[0]

    score_ema = _to_numpy(grid_state.get("score_ema"))
    if score_ema is None or score_ema.shape[0] < n_total:
        score_for_downsample = counts.astype(np.float32)
    else:
        score_for_downsample = score_ema[:n_total]
    if ids.shape[0] > int(max_cells):
        order = np.argsort(score_for_downsample[ids])[::-1]
        ids = ids[order[: int(max_cells)]]

    centers_s = centers[ids]
    levels_s = levels[ids]
    features_s = features[ids]
    sizes = voxel_size / np.power(2.0, levels_s.astype(np.float32))

    dyn = _get_debug_array(debug, "dynamic_prob_mean", n_total)[ids]
    sem = _get_debug_array(debug, "semantic_prob_mean", n_total)[ids]
    fg = _get_debug_array(debug, "foreground_fraction", n_total)[ids]
    mean_speed = _get_debug_array(debug, "mean_speed_mean", n_total)[ids]
    max_speed = _get_debug_array(debug, "max_speed_mean", n_total)[ids]
    max_disp = _get_debug_array(debug, "max_displacement_mean", n_total)[ids]
    feat_norm = np.linalg.norm(features_s, axis=1)
    counts_s = counts[ids]
    score_s = score_for_downsample[ids]

    cmode = str(color_mode).lower()
    scalar_by_mode = {
        "feature_norm": feat_norm,
        "dynamic_prob": dyn,
        "semantic_prob": sem,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "max_displacement": max_disp,
        "score_ema": score_s,
        "level": levels_s.astype(np.float32),
        "count": counts_s.astype(np.float32),
    }
    if cmode == "feature_pca":
        colors = _pca_rgb(features_s)
        color_note = "RGB = PCA(feature vector)"
    elif cmode in scalar_by_mode:
        colors = _scalar_rgb(scalar_by_mode[cmode])
        color_note = f"RGB = scalar colormap({cmode})"
    else:
        raise ValueError(f"Unknown motion-grid octomap color_mode '{color_mode}'")

    alpha = int(np.clip(float(opacity), 0.02, 1.0) * 255.0)
    verts, ii, jj, kk, vertexcolor = _make_cube_mesh(centers_s, sizes, colors, alpha=alpha)

    hover = []
    for n, cell_id in enumerate(ids.tolist()):
        f_preview = ", ".join(f"{v:.3f}" for v in features_s[n][: min(8, features_s.shape[1])])
        hover.append(
            f"cell_id={cell_id}<br>level={int(levels_s[n])}<br>size={sizes[n]:.6f}"
            f"<br>count={int(counts_s[n])}"
            f"<br>dynamic_prob_mean={dyn[n]:.5f}<br>semantic_prob_mean={sem[n]:.5f}"
            f"<br>foreground_fraction={fg[n]:.5f}"
            f"<br>mean_speed={mean_speed[n]:.5f}<br>max_speed={max_speed[n]:.5f}"
            f"<br>max_displacement={max_disp[n]:.5f}"
            f"<br>score={float(score_s[n]):.5f}<br>feature_norm={feat_norm[n]:.5f}"
            f"<br>feature[:8]=[{f_preview}]"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=ii, j=jj, k=kk,
            flatshading=True,
            vertexcolor=vertexcolor,
            opacity=float(np.clip(opacity, 0.02, 1.0)),
            name="octomap cells",
            hoverinfo="skip",
            showscale=False,
        )
    )
    # Center markers carry hover text and remain visible even with translucent cubes.
    fig.add_trace(
        go.Scatter3d(
            x=centers_s[:, 0], y=centers_s[:, 1], z=centers_s[:, 2],
            mode="markers",
            marker=dict(
                size=np.clip(2.0 + 1.75 * np.log1p(np.maximum(counts_s, 1)), 2.5, 8.0),
                color=[f"rgb({r},{g},{b})" for r, g, b in colors],
                opacity=0.95,
                line=dict(width=0.4, color="rgba(30,30,30,0.45)"),
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            name="cell feature samples",
        )
    )

    fig.update_layout(
        title=(
            f"{title}<br><sup>octomap-style adaptive grid · {color_note} · "
            f"shown {len(ids)} / {n_total} cells · voxel_size={voxel_size:.6f}</sup>"
        ),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(x=0.01, y=0.99),
    )

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(out_html), include_plotlyjs=include_plotlyjs, full_html=True, auto_open=False)

    summary = {
        "html": str(out_html),
        "num_cells_total": n_total,
        "num_cells_shown": int(len(ids)),
        "active_only": bool(active_only),
        "min_count": int(min_count),
        "max_cells": int(max_cells),
        "feature_dim": int(features.shape[1]) if features.ndim == 2 else 0,
        "voxel_size": voxel_size,
        "color_mode": cmode,
        "color_note": color_note,
        "opacity": float(opacity),
        "cell_level_min": int(levels.min()) if len(levels) else 0,
        "cell_level_max": int(levels.max()) if len(levels) else 0,
    }
    with open(out_html.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_motion_grid_octomap_html_from_file(
    state_path: str | Path,
    out_html: str | Path,
    **kwargs: Any,
) -> Dict[str, Any]:
    state = _as_state_dict(state_path)
    return write_motion_grid_octomap_html(state, out_html, **kwargs)
