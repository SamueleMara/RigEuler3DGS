"""Section 12: the Stage-A fixed-camera sequences and the HTML index.

CODEX_VALIDATE_GENERIC_TWIST_THEN_PARALLEL_PRISMATIC_REVOLUTE_PHASE2.md.

Six rows per camera, every one of them produced by the SAME renderer and the
same measurement model (section 6):

    GT                      the dataset RGB-D
    Static                  the canonical Gaussians, no motion
    Phase1                  the Phase-1 per-Gaussian solution
    Phase2 translation      the COM translation of the generic screw alone
    Phase2 rotation         its rotation about the COM alone
    Phase2 full rigid       both

Images are written as PNGs under `renders/`, one directory per camera and per
frame; the HTML is an index over those files and holds no image data itself.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from .rigid_motion_audit import STATES, build_state, com_decomposition

ROWS = ("gt", "static", "phase1", "translation", "rotation", "full")
ROW_TITLES = {"gt": "GT (dataset RGB-D)", "static": "Static (canonical)",
              "phase1": "Phase 1 (per-Gaussian)",
              "translation": "Phase 2 translation-only",
              "rotation": "Phase 2 rotation-only",
              "full": "Phase 2 full rigid"}


def _png(path: Path, img: torch.Tensor, depth: bool = False) -> None:
    """Write one [3,H,W] RGB or [H,W] depth tensor as an 8-bit PNG."""
    import numpy as np
    from PIL import Image

    a = img.detach().float().cpu()
    if depth:
        finite = a[torch.isfinite(a) & (a > 0)]
        if finite.numel():
            lo, hi = float(finite.min()), float(finite.max())
        else:
            lo, hi = 0.0, 1.0
        a = ((a - lo) / max(hi - lo, 1e-9)).clamp(0, 1)
        arr = (a.numpy() * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(path)
    else:
        arr = (a.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr, mode="RGB").save(path)


def render_sequences(out_dir, part, mu0, R0, observations, truth, cameras,
                     build_lin, phase1_positions=None, phase1_rotations=None,
                     frame_stride: int = 1, verbose: bool = True) -> dict:
    """Write the six rows for every camera and every sampled frame."""
    root = Path(out_dir) / "renders"
    root.mkdir(parents=True, exist_ok=True)
    T = len(observations)
    frames = list(range(0, T, max(1, int(frame_stride))))
    index = {"frames": frames, "cameras": len(cameras), "rows": list(ROWS),
             "files": []}

    for t in frames:
        dec = com_decomposition(part, t, mu0)
        states = {}
        for kind in STATES:
            states[kind] = build_state(kind, part, t, mu0, R0, dec)
        if phase1_positions is not None:
            mu_p1 = torch.as_tensor(phase1_positions[t]).double().to(mu0.device)
            Rg_p1 = (None if phase1_rotations is None
                     else torch.as_tensor(phase1_rotations[t]).double())
            states["phase1"] = (mu_p1, Rg_p1)
        for row in ROWS:
            if row == "gt":
                continue
            if row not in states:
                continue
            mu, Rg = states[row]
            lin = build_lin(mu, Rg)
            for ci, v in enumerate(lin.views):
                d = root / f"camera_{ci:03d}" / f"frame_{t:04d}"
                d.mkdir(parents=True, exist_ok=True)
                _png(d / f"{row}_rgb.png", v.rgb)
                _png(d / f"{row}_depth.png", v.depth, depth=True)
                index["files"].append(
                    [t, ci, row, str((d / f"{row}_rgb.png").relative_to(root)),
                     str((d / f"{row}_depth.png").relative_to(root))])
        for ci, ob in enumerate(observations[t]):
            if ob is None:
                continue
            d = root / f"camera_{ci:03d}" / f"frame_{t:04d}"
            d.mkdir(parents=True, exist_ok=True)
            _png(d / "gt_rgb.png", ob[0])
            _png(d / "gt_depth.png", ob[1], depth=True)
            index["files"].append(
                [t, ci, "gt", str((d / "gt_rgb.png").relative_to(root)),
                 str((d / "gt_depth.png").relative_to(root))])
        if verbose:
            print(f"    rendered frame {t}")
    return index


def write_html(out_dir, index: dict, summary: dict, title: str) -> Path:
    """An index over the PNGs on disk. No image bytes are embedded."""
    out = Path(out_dir)
    by = {}
    for (t, ci, row, rgb, dep) in index["files"]:
        by.setdefault((ci, t), {})[row] = (rgb, dep)
    rows_html = []
    for (ci, t) in sorted(by):
        cells = []
        for row in ROWS:
            got = by[(ci, t)].get(row)
            if not got:
                cells.append(f'<td class="missing">{ROW_TITLES[row]}<br>'
                             f'not rendered</td>')
                continue
            rgb, dep = got
            cells.append(
                f'<td><div class="t">{ROW_TITLES[row]}</div>'
                f'<img src="renders/{rgb}" loading="lazy">'
                f'<img src="renders/{dep}" loading="lazy" class="d"></td>')
        rows_html.append(f'<tr><th>cam {ci}<br>frame {t}</th>'
                         + "".join(cells) + "</tr>")
    import json as _json
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body{{font:13px/1.5 system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
 table{{border-collapse:collapse}} td,th{{border:1px solid #333;padding:4px;
 vertical-align:top;text-align:center}} th{{background:#1b1b1b;white-space:nowrap}}
 img{{width:180px;display:block;margin:2px auto;image-rendering:pixelated}}
 img.d{{filter:none;opacity:.95}} .t{{font-size:11px;color:#9cf;margin-bottom:2px}}
 .missing{{color:#666;font-size:11px}}
 pre{{background:#1b1b1b;padding:12px;overflow:auto;border:1px solid #333}}
</style></head><body>
<h1>{title}</h1>
<p>Section 12 Stage-A fixed-camera sequences. Each cell is RGB above,
normalised depth below. Every row is rendered by the same rasterizer under the
same measurement model; the only thing that changes is the state.</p>
<h2>Stage-A summary</h2>
<pre>{_json.dumps(summary, indent=1, default=str)}</pre>
<h2>Sequences</h2>
<table>{''.join(rows_html)}</table>
</body></html>"""
    p = out / "stage_a_sequences.html"
    p.write_text(html)
    return p
