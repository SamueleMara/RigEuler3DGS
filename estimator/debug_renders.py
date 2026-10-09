"""Sections 23, 24: the chosen camera, every frame, every method.

CODEX_COMPLETE_PHASE2_DEBUG_AND_PARALLEL_PR_MODELS.md.

Five methods -- GT, Phase 1, generic screw, exact P, exact R -- in RGB and
depth, plus RGB and depth residuals against GT.

Section 23 is strict about scaling, and for a good reason: depth normalized
per method makes every method look equally good. One global depth range is
computed over the WHOLE sequence and every image in the run uses it, and the
residual colour range is likewise shared, so two cells can be compared by eye.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

METHODS = ("gt", "phase1", "screw", "prismatic", "revolute")
METHOD_TITLES = {"gt": "GT", "phase1": "Phase 1 (free)",
                 "screw": "Phase 2 screw", "prismatic": "Phase 2 P",
                 "revolute": "Phase 2 R"}


class DepthScale:
    """One depth range for the entire run (section 23)."""

    def __init__(self):
        self.lo, self.hi = None, None

    def observe(self, d: torch.Tensor) -> None:
        a = d.detach().float().cpu()
        f = a[torch.isfinite(a) & (a > 0)]
        if not f.numel():
            return
        lo, hi = float(f.min()), float(f.max())
        self.lo = lo if self.lo is None else min(self.lo, lo)
        self.hi = hi if self.hi is None else max(self.hi, hi)

    def to_dict(self) -> dict:
        return {"depth_min": self.lo, "depth_max": self.hi,
                "normalization": "global over the whole sequence and all methods",
                "per_method_normalization": False}


def _save_rgb(path: Path, img: torch.Tensor) -> None:
    from PIL import Image
    a = img.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((a * 255).astype(np.uint8), mode="RGB").save(path)


def _save_depth(path: Path, d: torch.Tensor, scale: DepthScale) -> None:
    from PIL import Image
    a = d.detach().float().cpu()
    lo = 0.0 if scale.lo is None else scale.lo
    hi = 1.0 if scale.hi is None else max(scale.hi, lo + 1e-9)
    a = ((a - lo) / (hi - lo)).clamp(0, 1)
    Image.fromarray((a.numpy() * 255).astype(np.uint8), mode="L").save(path)


def _save_residual(path: Path, r: torch.Tensor, vmax: float) -> None:
    """Signed residual as blue-black-red, on a SHARED symmetric range."""
    from PIL import Image
    a = r.detach().float().cpu()
    if a.ndim == 3:
        a = a.abs().mean(0)
    v = max(float(vmax), 1e-12)
    t = (a / v).clamp(-1, 1).numpy()
    h, w = t.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(t, 0, 1)
    rgb[..., 2] = np.clip(-t, 0, 1)
    Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").save(path)


def render_chosen_camera(out_dir, camera_id: int, frames_gt, states,
                         truth, cameras, cfg, build_linearization,
                         verbose: bool = True, method_dirs=None) -> dict:
    """Render every frame of ONE camera for every method (section 23).

    `states` maps a method name to a callable `t -> (mu, Rg)`; a method that
    has no state (R was unobservable, say) is simply absent and its row is
    marked, never filled with another method's image.
    """
    from utils.progress import pbar

    from .phase2_shared import render_state

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    cam = [cameras[camera_id]]
    T = len(frames_gt)

    # ---- pass 1: the shared scales (section 23) -------------------------
    scale = DepthScale()
    rgb_vmax = depth_vmax = 0.0
    cache = {}
    for t in pbar(range(T), "renders: scan scales", total=T):
        ob = frames_gt[t][camera_id]
        if ob is not None:
            scale.observe(ob[1])
        for name, fn in states.items():
            mu, Rg = fn(t)
            lin = render_state(truth, mu, Rg, cam, cfg, build_linearization)
            v = lin.views[0]
            scale.observe(v.depth)
            if ob is not None:
                rr = (ob[0].to(v.rgb.device) - v.rgb).abs().mean(0)
                rd = (ob[1].to(v.depth.device) - v.depth).abs()
                rgb_vmax = max(rgb_vmax, float(rr.max()))
                depth_vmax = max(depth_vmax, float(rd.max()))
            cache[(t, name)] = (v.rgb.cpu(), v.depth.cpu())
            del lin

    # ---- pass 2: write, with every image on the same scales -------------
    # Section 22's tree puts each method's images under its OWN directory, so
    # `method_dirs` maps a method to where its PNGs belong. The HTML stores
    # paths relative to `root`, so the two stay in step however the caller
    # arranges the tree.
    files = []
    sub = f"camera_{camera_id:03d}"

    def _dir(name):
        d = (Path(method_dirs[name]) if method_dirs and name in method_dirs
             else root / name) / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _rel(d, fname):
        try:
            return str((d / fname).resolve().relative_to(root.resolve()))
        except ValueError:
            return str((d / fname).resolve())

    for t in pbar(range(T), "renders: write PNGs", total=T):
        ob = frames_gt[t][camera_id]
        if ob is not None:
            d = _dir("gt")
            _save_rgb(d / f"frame_{t:04d}_rgb.png", ob[0])
            _save_depth(d / f"frame_{t:04d}_depth.png", ob[1], scale)
            files.append([t, "gt", _rel(d, f"frame_{t:04d}_rgb.png"),
                          _rel(d, f"frame_{t:04d}_depth.png"), None, None])
        for name in METHODS:
            if name == "gt" or name not in states:
                continue
            d = _dir(name)
            rgb, dep = cache[(t, name)]
            _save_rgb(d / f"frame_{t:04d}_rgb.png", rgb)
            _save_depth(d / f"frame_{t:04d}_depth.png", dep, scale)
            rr = rd = None
            if ob is not None:
                _save_residual(d / f"frame_{t:04d}_rgb_residual.png",
                               (ob[0].cpu() - rgb), rgb_vmax)
                _save_residual(d / f"frame_{t:04d}_depth_residual.png",
                               (ob[1].cpu() - dep), depth_vmax)
                rr = _rel(d, f"frame_{t:04d}_rgb_residual.png")
                rd = _rel(d, f"frame_{t:04d}_depth_residual.png")
            files.append([t, name, _rel(d, f"frame_{t:04d}_rgb.png"),
                          _rel(d, f"frame_{t:04d}_depth.png"), rr, rd])
        if verbose and (t % 10 == 0 or t == T - 1):
            print(f"      chosen-camera frame {t + 1}/{T}")
    return {"camera": camera_id, "frames": T, "files": files,
            "methods": [m for m in METHODS if m == "gt" or m in states],
            "depth_scale": scale.to_dict(),
            "rgb_residual_vmax": rgb_vmax, "depth_residual_vmax": depth_vmax}


def write_chosen_camera_html(path, index: dict, per_frame: dict,
                             rel_root: str) -> Path:
    """Section 24: a frame slider over the chosen camera, with the numbers."""
    import json as _json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by = {}
    for (t, m, rgb, dep, rr, rd) in index["files"]:
        by.setdefault(t, {})[m] = [rgb, dep, rr, rd]
    methods = index["methods"]
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Chosen camera &mdash; all frames</title><style>
 body{{font:13px/1.5 system-ui,sans-serif;margin:20px;background:#111;color:#eee}}
 table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #333;
 padding:4px;text-align:center;vertical-align:top}} th{{background:#1b1b1b}}
 img{{width:100%;max-width:260px;display:block;margin:2px auto;
 image-rendering:pixelated;background:#000}}
 .lbl{{font-size:11px;color:#9cf}} #bar{{position:sticky;top:0;background:#111;
 padding:10px 0;z-index:9;border-bottom:1px solid #333}}
 input[type=range]{{width:60%}} pre{{background:#1b1b1b;padding:10px;
 border:1px solid #333;overflow:auto;font-size:12px}}
 .num{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
 gap:6px;margin:8px 0}} .num div{{background:#1b1b1b;padding:6px 9px;
 border:1px solid #333}} .k{{color:#9cf}}
</style></head><body>
<h1>Chosen camera {index['camera']} &mdash; all {index['frames']} frames</h1>
<p>Section 23: one global depth range over the whole sequence and all methods
({_json.dumps(index['depth_scale'])}), and one shared symmetric residual range.
No method is normalized on its own.</p>
<div id="bar">
 <input type="range" id="sl" min="0" max="{index['frames'] - 1}" value="0">
 <span id="lbl">frame 0</span>
</div>
<div class="num" id="nums"></div>
<table id="grid"><thead><tr><th>row</th>
{''.join(f'<th>{METHOD_TITLES[m]}</th>' for m in methods)}</tr></thead>
<tbody></tbody></table>
<script>
const BY = {_json.dumps(by)};
const PF = {_json.dumps(per_frame)};
const M  = {_json.dumps(methods)};
const TT = {_json.dumps(METHOD_TITLES)};
const ROOT = {_json.dumps(rel_root)};
const ROWS = [["RGB",0],["depth",1],["RGB residual",2],["depth residual",3]];
function draw(t){{
  document.getElementById('lbl').textContent = 'frame ' + t;
  const tb = document.querySelector('#grid tbody');
  tb.innerHTML = ROWS.map(([name,i]) => '<tr><th>'+name+'</th>' + M.map(m => {{
     const e = (BY[t]||{{}})[m];
     if(!e || !e[i]) return '<td class="lbl">&mdash;</td>';
     return '<td><img loading="lazy" src="'+ROOT+'/'+e[i]+'"></td>';
  }}).join('') + '</tr>').join('');
  const p = PF[t] || {{}};
  document.getElementById('nums').innerHTML = Object.keys(p).map(k =>
     '<div><span class="k">'+k+'</span><br>'+p[k]+'</div>').join('');
}}
document.getElementById('sl').addEventListener('input', e => draw(+e.target.value));
draw(0);
</script></body></html>"""
    path.write_text(doc)
    return path
