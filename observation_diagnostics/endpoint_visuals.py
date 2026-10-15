"""Figures and pages for the endpoint diagnosis: legible without the CSVs.

Reuses observation_diagnostics.visualize for colour keys, depth/residual maps
and PNG writing; adds the endpoint grid, case-study sheets, the 3-D pages
(with the estimated joint axis) and overview.html.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from . import visualize as vz

WARN = ("CENTER_ONLY_FIXED_ORIENTATION: the historical Phase-1 runs saved centres only. "
        "Transported states (B0, FT) use each Gaussian's TRAINED orientation at every frame; "
        "anisotropic footprints of rotating parts are wrong away from their anchor, which can "
        "create or hide apparent mismatches.")


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def images_for(view, gt, depth_range):
    rgb = vz._to_u8(_np(view.lin.rgb).transpose(1, 2, 0))
    a = _np(view.lin.alpha)
    d = vz.depth_rgb(_np(view.lin.depth), a > 0.05, depth_range)
    e = np.where(_np(gt["valid"]), _np(view.lin.depth) - _np(gt["depth"]), 0)
    res = vz.residual_rgb(e, _np(gt["valid"]) & (a >= 0.5))
    return rgb, d, res


def endpoint_grid(path: Path, scene, rows, depth_range):
    """rows: [(label, [(title, img), ...])], one row per (frame, camera)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.6 * ncol, 2.2 * len(rows)), squeeze=False)
    for i, (lab, panels) in enumerate(rows):
        for j in range(ncol):
            ax = axes[i][j]
            ax.axis("off")
            if j < len(panels):
                t, img = panels[j]
                ax.imshow(img)
                ax.set_title(t, fontsize=7)
        axes[i][0].text(-0.05, 0.5, lab, transform=axes[i][0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=7)
    fig.suptitle(f"{scene}: endpoint contrasts, each source rendered alone.  depth turbo "
                 f"[{depth_range[0]:.2f}, {depth_range[1]:.2f}] m, residual RdBu "
                 f"+-{vz.RESIDUAL_RANGE_M} m (red = source behind the observation)\n{WARN}",
                 fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def case_sheet(path: Path, title, rows, note=""):
    """rows: [(label, [(title, img), ...])] -- crops, same box per row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.1 * ncol, 2.2 * len(rows)), squeeze=False)
    for i, (lab, panels) in enumerate(rows):
        for j in range(ncol):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j < len(panels):
                t, img = panels[j]
                ax.imshow(img)
                if i == 0 or t.startswith("*"):
                    ax.set_title(t.lstrip("*"), fontsize=7)
            else:
                ax.axis("off")
        axes[i][0].set_ylabel(lab, fontsize=7)
    fig.suptitle(title + ("\n" + note if note else ""), fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=85)
    plt.close(fig)


def overlay(rgb_u8, mass, colour=(255, 255, 255)):
    base = rgb_u8.astype(np.float32) * 0.45
    m = np.clip(mass / max(float(mass.max()), 1e-9), 0, 1)[..., None]
    return np.clip(base + m * np.asarray(colour, np.float32), 0, 255).astype(np.uint8)


def crop(img, box, margin=24):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(x0 - margin, 0), max(y0 - margin, 0)
    x1, y1 = min(x1 + margin, W - 1), min(y1 + margin, H - 1)
    if x1 - x0 < 80:
        c = (x0 + x1) // 2
        x0, x1 = max(c - 40, 0), min(c + 40, W - 1)
    if y1 - y0 < 80:
        c = (y0 + y1) // 2
        y0, y1 = max(c - 40, 0), min(c + 40, H - 1)
    return img[y0:y1 + 1, x0:x1 + 1]


def plot3d(path: Path, title: str, layers: list, arrows: list = (), note: str = ""):
    """layers: (name, xyz[n,3], colour, size); arrows: (name, origin, direction, length, colour)."""
    traces = []
    for name, xyz, col, size in layers:
        if len(xyz) == 0:
            continue
        traces.append({"type": "scatter3d", "mode": "markers", "name": name,
                       "x": np.round(xyz[:, 0], 4).tolist(), "y": np.round(xyz[:, 1], 4).tolist(),
                       "z": np.round(xyz[:, 2], 4).tolist(),
                       "marker": {"size": size, "color": col, "opacity": 0.85}})
    for name, o, d, L, col in arrows:
        o, d = np.asarray(o, float), np.asarray(d, float) / np.linalg.norm(d)
        a, b = o - d * L / 2, o + d * L / 2
        traces.append({"type": "scatter3d", "mode": "lines", "name": name,
                       "x": [a[0], b[0]], "y": [a[1], b[1]], "z": [a[2], b[2]],
                       "line": {"width": 8, "color": col}})
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>body{{font-family:sans-serif;margin:12px;background:#fff;color:#111}}
.warn{{background:#fff3cd;border:1px solid #e0b000;padding:6px;font-size:13px}}</style></head><body>
<h3>{html.escape(title)}</h3><div class="warn">{html.escape(WARN)}</div>
<p style="font-size:13px">{html.escape(note)}</p>
<div id="p" style="width:100%;height:82vh"></div>
<script>Plotly.newPlot('p',{json.dumps(traces)},{{scene:{{aspectmode:'data'}},legend:{{x:0,y:1}}}});</script>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)


def overview(path: Path, scene: str, summary: dict, tables: dict, images: list):
    """tables: {title: [row dicts]}; images: [(title, relative path)]."""
    def tab(rows):
        if not rows:
            return "<p><i>none</i></p>"
        keys = list(rows[0])
        head = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(r.get(k, '')))}</td>" for k in keys)
                       + "</tr>" for r in rows)
        return f"<table><tr>{head}</tr>{body}</table>"
    secs = "".join(f"<h3>{html.escape(t)}</h3>{tab(r)}" for t, r in tables.items())
    imgs = "".join(f"<h3>{html.escape(t)}</h3><img src='{html.escape(p)}' style='max-width:100%'>"
                   for t, p in images)
    flags = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
                    for k, v in summary.get("flags", {}).items())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Endpoint evidence: {html.escape(scene)}</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fff;color:#111;max-width:1500px}}
table{{border-collapse:collapse;font-size:12px;margin-bottom:12px}} td,th{{border:1px solid #ccc;padding:2px 6px}}
.warn{{background:#fff3cd;border:1px solid #e0b000;padding:8px}}</style></head><body>
<h2>{html.escape(scene)}: which observed surfaces are shared, complementary, or misplaced by transport</h2>
<div class="warn"><b>{html.escape(WARN)}</b><br>Read-only diagnostic. No GT joints, poses, part masks,
labels or meshes were read (firewall report in provenance.json). Estimated joints are inferred
quantities from the same RGB-D, used as consistency tests, not as measurements.</div>
<h3>Report flags</h3><table>{flags}</table>
<p>Pages: <a href="static_correspondence_3d.html">static correspondence 3-D</a>,
<a href="complementary_patches_3d.html">complementary patches 3-D</a>,
<a href="../endpoint_summary.md">endpoint_summary.md</a>.</p>
{secs}{imgs}</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
