"""Maps, timelines, panels and galleries with fixed, documented colour keys.

Every categorical map uses the colours in LEGENDS; every depth map uses one
range per scene (DEPTH_RANGE, from the GT); every residual map uses
+-RESIDUAL_RANGE_M. The keys are written into index.html and onto every
matplotlib panel.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

RESIDUAL_RANGE_M = 0.10

LEGENDS = {
    "coverage_categories": [
        ("NEITHER_COVERS", (60, 60, 60)), ("F_ONLY_COVER", (40, 110, 230)),
        ("B_ONLY_COVER", (245, 140, 20)), ("BOTH_COVER", (40, 180, 70)),
        ("(any of the above on a GT-invalid pixel: same hue, darkened)", (25, 25, 25))],
    "support": [
        ("NEITHER_VALID_SUPPORT", (170, 30, 30)), ("F_SUPPORTED_B_UNEXPLAINED", (40, 110, 230)),
        ("B_SUPPORTED_F_UNEXPLAINED", (245, 140, 20)), ("BOTH_VALID_SUPPORT", (40, 180, 70)),
        ("GT_INVALID", (0, 0, 0))],
    "same_surface_candidates": [
        ("NOT_BOTH_COVER (GT valid)", (70, 70, 70)),
        ("BOTH_SAME_OBSERVED_SURFACE_CANDIDATE", (40, 200, 70)),
        ("BOTH_COVER_DIFFERENT_DEPTH", (220, 40, 200)),
        ("BOTH_COVER_AGREE_OFF_OBSERVED_SURFACE", (230, 220, 40)),
        ("BOTH_COVER_GT_INVALID", (150, 150, 150)), ("GT_INVALID", (0, 0, 0))],
    "B_explains_F_misses": [("B explains GT depth, F does not", (245, 140, 20)),
                            ("other GT-valid pixel", (70, 70, 70)), ("GT_INVALID", (0, 0, 0))],
    "F_explains_B_misses": [("F explains GT depth, B does not", (40, 110, 230)),
                            ("other GT-valid pixel", (70, 70, 70)), ("GT_INVALID", (0, 0, 0))],
}
_COVER = np.array([c for _n, c in LEGENDS["coverage_categories"][:4]], np.uint8)
_SUPPORT = np.array([c for _n, c in LEGENDS["support"]], np.uint8)
_LAYER = np.array([c for _n, c in LEGENDS["same_surface_candidates"]], np.uint8)


def _cmap(name):
    import matplotlib
    return matplotlib.colormaps[name]


def _to_u8(rgb_float):
    return (np.clip(rgb_float, 0, 1) * 255 + 0.5).astype(np.uint8)


def depth_rgb(depth, mask, rng):
    lo, hi = rng
    x = np.clip((depth - lo) / max(hi - lo, 1e-9), 0, 1)
    img = _to_u8(_cmap("turbo")(x)[..., :3])
    img[~mask] = 0
    return img


def residual_rgb(res, mask, r=RESIDUAL_RANGE_M):
    x = np.clip(res / r * 0.5 + 0.5, 0, 1)
    img = _to_u8(_cmap("RdBu_r")(x)[..., :3])
    img[~mask] = 0
    return img


def gray(a):
    return np.repeat(_to_u8(a)[..., None], 3, axis=2)


def save_png(path: Path, img: np.ndarray):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, optimize=False, compress_level=3)


def view_maps(fields_np: dict, F: dict, B: dict, gt: dict, depth_range) -> dict:
    """The per-(frame, camera) images, as uint8 RGB arrays."""
    valid = gt["valid"]
    cover = _COVER[fields_np["cover"]].copy()
    cover[~valid] = (cover[~valid] * 0.35).astype(np.uint8)
    support = _SUPPORT[fields_np["support"]]
    layer_code = fields_np["layer"].copy()
    lay = np.where(valid, layer_code, np.where(layer_code == 4, 4, 5))
    layer = _LAYER[lay]
    disagree = depth_rgb(fields_np["dFB"], fields_np["layer"] == 2, (0.0, 0.3))

    def binary(mask, col):
        img = np.full(valid.shape + (3,), 70, np.uint8)
        img[mask] = col
        img[~valid] = 0
        return img

    gate = 0.05
    return {
        "F_alpha": gray(F["alpha"]), "B_alpha": gray(B["alpha"]),
        "F_depth": depth_rgb(F["depth"], F["alpha"] > gate, depth_range),
        "B_depth": depth_rgb(B["depth"], B["alpha"] > gate, depth_range),
        "GT_depth": depth_rgb(gt["depth"], valid, depth_range),
        "F_depth_residual": residual_rgb(fields_np["eF"], valid & fields_np["covF"]),
        "B_depth_residual": residual_rgb(fields_np["eB"], valid & fields_np["covB"]),
        "coverage_categories": cover, "support_categories": support,
        "same_surface_candidates": layer,
        "B_explains_F_misses": binary(fields_np["W_Bonly"], (245, 140, 20)),
        "F_explains_B_misses": binary(fields_np["W_Fonly"], (40, 110, 230)),
        "depth_layer_disagreement": disagree,
    }


def legend_html() -> str:
    parts = []
    for name, items in LEGENDS.items():
        rows = "".join(
            f'<span class="sw" style="background:rgb{c}"></span>{html.escape(n)}<br>'
            for n, c in items)
        parts.append(f"<div class='leg'><b>{name}</b><br>{rows}</div>")
    parts.append("<div class='leg'><b>depth maps</b><br>turbo colormap over the scene's "
                 "GT depth range DEPTH_RANGE; black = alpha &le; 0.05 or GT invalid</div>")
    parts.append(f"<div class='leg'><b>residual maps</b><br>D_source - D_gt, RdBu, "
                 f"&plusmn;{RESIDUAL_RANGE_M} m (red = source behind the observation, "
                 "blue = in front); black = not covered or GT invalid</div>")
    parts.append("<div class='leg'><b>depth_layer_disagreement</b><br>|D_F - D_B| on "
                 "BOTH_COVER_DIFFERENT_DEPTH pixels, turbo 0..0.3 m</div>")
    return "\n".join(parts)


MAP_ORDER = ("GT_RGB", "F_RGB", "B_RGB", "GT_depth", "F_depth", "B_depth", "F_alpha",
             "B_alpha", "F_depth_residual", "B_depth_residual", "coverage_categories",
             "support_categories", "same_surface_candidates", "B_explains_F_misses",
             "F_explains_B_misses", "depth_layer_disagreement")


def write_index(out: Path, scene: str, frames, cams, depth_range, summary: dict):
    """index.html: frame/camera navigation over pixel_maps/, with the keys."""
    data = json.dumps({"frames": list(frames), "cams": list(cams),
                       "maps": list(MAP_ORDER)})
    rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
                   for k, v in summary.items())
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>F/B overlap: {html.escape(scene)}</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fff;color:#111}}
img{{width:320px;image-rendering:pixelated;border:1px solid #ccc}}
.grid{{display:flex;flex-wrap:wrap;gap:6px}} .cell{{font-size:12px}}
.sw{{display:inline-block;width:12px;height:12px;margin-right:4px;border:1px solid #888}}
.leg{{display:inline-block;vertical-align:top;margin:6px 18px 6px 0;font-size:12px}}
table{{border-collapse:collapse;font-size:12px}} td{{border:1px solid #ccc;padding:2px 6px}}</style>
</head><body>
<h2>{html.escape(scene)}: forward vs backward, each source rendered alone</h2>
<p>DEPTH_RANGE = [{depth_range[0]:.3f}, {depth_range[1]:.3f}] m. Sources rendered
CENTER_ONLY_FIXED_ORIENTATION unless provenance.json says otherwise.
See ../report/RESULTS.md for the numbers and their caveats.</p>
<div>{legend_html()}</div>
<p>frame <select id="f"></select> camera <select id="c"></select>
<button onclick="step(-1)">&larr;</button><button onclick="step(1)">&rarr;</button></p>
<div class="grid" id="g"></div>
<h3>summary</h3><table>{rows}</table>
<p>Other pages: <a href="coverage_timeline.png">coverage timeline</a>,
<a href="depth_error_timeline.png">depth-error timeline</a>,
<a href="gaussian_3d_diagnostics.html">3-D candidates</a>,
<a href="source_comparison_panels/">comparison panels</a>,
<a href="missing_surface_case_studies/">missing-surface case studies</a>,
<a href="pair_case_studies/">pair case studies</a>.</p>
<script>
const D={data};
const f=document.getElementById('f'),c=document.getElementById('c'),g=document.getElementById('g');
D.frames.forEach(x=>f.add(new Option(x,x)));D.cams.forEach(x=>c.add(new Option(x,x)));
function pad(n){{return String(n).padStart(4,'0')}}
function show(){{g.innerHTML='';D.maps.forEach(m=>{{const d=document.createElement('div');d.className='cell';
d.innerHTML=m+'<br><img loading="lazy" src="../pixel_maps/frame_'+pad(f.value)+'/'+c.value+'/'+m+'.png">';g.appendChild(d)}})}}
function step(k){{f.selectedIndex=Math.min(Math.max(f.selectedIndex+k,0),D.frames.length-1);show()}}
f.onchange=show;c.onchange=show;show();
</script></body></html>"""
    (out / "index.html").write_text(page)


def timeline_plots(out: Path, per_frame, scene: str):
    """coverage_timeline.png and depth_error_timeline.png from per_frame rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = [r["frame"] for r in per_frame]

    def pct(key):
        return [100.0 * r[key] / max(r["gt_valid"], 1) for r in per_frame]

    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for key, col, lab in (("BOTH_COVER__on_gt_valid", "tab:green", "both cover"),
                          ("F_ONLY_COVER__on_gt_valid", "tab:blue", "F only covers"),
                          ("B_ONLY_COVER__on_gt_valid", "tab:orange", "B only covers"),
                          ("NEITHER_COVERS__on_gt_valid", "0.4", "neither covers")):
        ax[0].plot(fr, pct(key), color=col, label=lab)
    ax[0].set_ylabel("% of GT-valid pixels")
    ax[0].set_title(f"{scene}: COVERAGE (alpha >= alpha_cover), all cameras pooled")
    ax[0].legend(fontsize=8, ncol=4)
    for key, col, lab in (("BOTH_VALID_SUPPORT", "tab:green", "both explain depth"),
                          ("F_SUPPORTED_B_UNEXPLAINED", "tab:blue", "F explains, B does not"),
                          ("B_SUPPORTED_F_UNEXPLAINED", "tab:orange", "B explains, F does not"),
                          ("NEITHER_VALID_SUPPORT", "tab:red", "neither explains"),
                          ("BOTH_SAME_OBSERVED_SURFACE_CANDIDATE", "k", "same-surface candidate"),
                          ("BOTH_COVER_DIFFERENT_DEPTH", "m", "both cover, different depth")):
        ax[1].plot(fr, pct(key), color=col, label=lab)
    ax[1].set_ylabel("% of GT-valid pixels")
    ax[1].set_xlabel("original frame")
    ax[1].set_title("EXPLANATION (|D - D_gt| <= depth_tol on covered GT-valid pixels)")
    ax[1].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "coverage_timeline.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for key, col in (("F", "tab:blue"), ("B", "tab:orange")):
        ax[0].plot(fr, [r.get(f"{key}_depth_mae_m__covered") for r in per_frame], color=col,
                   label=f"{key}: depth MAE on its covered GT-valid pixels")
        ax[0].plot(fr, [r.get(f"{key}_depth_mae_m__production_gate") for r in per_frame],
                   color=col, ls="--", label=f"{key}: depth MAE, alpha > 0.05 gate")
        ax[1].plot(fr, [r.get(f"{key}_rgb_rmse__all_valid") for r in per_frame], color=col,
                   label=f"{key}: RGB RMSE on all GT-valid pixels")
    ax[0].set_ylabel("m")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{scene}: observed residuals per original frame (supports differ "
                    "between sources and are NOT equalised)")
    ax[1].legend(fontsize=8)
    ax[1].set_xlabel("original frame")
    fig.tight_layout()
    fig.savefig(out / "depth_error_timeline.png", dpi=110)
    plt.close(fig)


def comparison_panel(path: Path, title: str, imgs: dict, depth_range):
    """GT | F | B rows plus the category maps, with keys, for one view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    order = [["GT_RGB", "F_RGB", "B_RGB", "coverage_categories"],
             ["GT_depth", "F_depth", "B_depth", "support_categories"],
             ["F_alpha", "F_depth_residual", "B_depth_residual", "same_surface_candidates"]]
    fig, axes = plt.subplots(3, 4, figsize=(17, 10.5))
    for r, row in enumerate(order):
        for c, name in enumerate(row):
            a = axes[r][c]
            a.imshow(imgs[name])
            a.set_title(name, fontsize=9)
            a.axis("off")
            key = {"coverage_categories": "coverage_categories",
                   "support_categories": "support",
                   "same_surface_candidates": "same_surface_candidates"}.get(name)
            if key:
                a.legend(handles=[Patch(color=np.array(col) / 255, label=n)
                                  for n, col in LEGENDS[key]],
                         fontsize=6, loc="lower left", framealpha=0.85)
    fig.suptitle(f"{title}\ndepth turbo [{depth_range[0]:.2f}, {depth_range[1]:.2f}] m; "
                 f"residual RdBu +-{RESIDUAL_RANGE_M} m (red: source behind GT)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def case_study(path: Path, title: str, rows: list, depth_range):
    """rows: one per frame: dict of crops (uint8) + 'label'. Columns fixed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["GT_RGB", "GT_depth", "F_RGB", "F_depth", "B_RGB", "B_depth",
            "F_depth_residual", "B_depth_residual", "contribution"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.0 * len(cols), 2.1 * len(rows)),
                             squeeze=False)
    for r, row in enumerate(rows):
        for c, name in enumerate(cols):
            a = axes[r][c]
            a.imshow(row[name])
            a.set_xticks([])
            a.set_yticks([])
            if r == 0:
                a.set_title(name, fontsize=8)
            if c == 0:
                a.set_ylabel(row["label"], fontsize=7)
    fig.suptitle(title + f"\ndepth turbo [{depth_range[0]:.2f}, {depth_range[1]:.2f}] m; residual "
                 f"RdBu +-{RESIDUAL_RANGE_M} m; 'contribution' = sum of q over the nominated "
                 "Gaussians (white) on the GT RGB", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=85)
    plt.close(fig)


def curve_plot(path: Path, title: str, series: dict, ylabel: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    for lab, (x, y) in series.items():
        ax.plot(x, y, label=lab)
    ax.set_xlabel("original frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def gaussian_3d_html(path: Path, scene: str, frame: int, layers: list):
    """layers: [(name, xyz[n,3], colour, size)]. Plotly from a CDN."""
    traces = []
    for name, xyz, col, size in layers:
        traces.append({"type": "scatter3d", "mode": "markers", "name": name,
                       "x": xyz[:, 0].round(4).tolist(), "y": xyz[:, 1].round(4).tolist(),
                       "z": xyz[:, 2].round(4).tolist(),
                       "marker": {"size": size, "color": col, "opacity": 0.8}})
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>3-D candidates: {html.escape(scene)}</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>body{{font-family:sans-serif;margin:12px;background:#fff}}</style></head><body>
<h3>{html.escape(scene)}: source centres at original frame {frame} (Phase-1 tracks)</h3>
<p>Background layers are a uniform display subsample (stated in each layer's name);
the candidate layers are complete.</p>
<div id="p" style="width:100%;height:85vh"></div>
<script>Plotly.newPlot('p',{json.dumps(traces)},{{scene:{{aspectmode:'data'}},legend:{{x:0,y:1}}}});</script>
</body></html>"""
    path.write_text(page)
