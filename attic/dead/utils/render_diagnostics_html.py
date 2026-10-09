"""§46 -- the mandatory render/refinement diagnostics page.

Every number this pipeline produces is an aggregate over stages that can each
fail silently.  The page exists so a dead stage is visible at a glance instead
of requiring someone to read a JSON and notice an absent key: if the refined
render looks identical to the initial one, refinement did nothing, and no
summary statistic says that as directly as the two pictures side by side.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import torch


def _png(tensor: torch.Tensor, *, cmap: str | None = None) -> str:
    """A tensor as a base64 PNG data URI, so the page is a single file."""
    import numpy as np
    from PIL import Image

    array = torch.as_tensor(tensor).detach().float().cpu()
    if array.ndim == 3 and int(array.shape[0]) in (1, 3):
        array = array.permute(1, 2, 0)
    if array.ndim == 3 and int(array.shape[-1]) == 1:
        array = array[..., 0]
    array = array.numpy()

    if array.ndim == 2:
        low, high = float(np.nanmin(array)), float(np.nanmax(array))
        normalised = (array - low) / max(high - low, 1e-8)
        if cmap == "diverging":
            # Signed quantities (flow, residual) get a two-sided map so the
            # sign is readable rather than folded into brightness.
            limit = max(abs(low), abs(high), 1e-8)
            scaled = (array / limit + 1.0) / 2.0
            rgb = np.stack([scaled, np.zeros_like(scaled), 1.0 - scaled], axis=-1)
        else:
            rgb = np.stack([normalised] * 3, axis=-1)
    else:
        rgb = np.clip(array, 0.0, 1.0)

    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8"))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def write_render_diagnostics_html(path: Path, payload: dict[str, Any]) -> None:
    """One page per run: per-hypothesis panels plus the scene-level maps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload)
    path.write_text(
        """<!doctype html><meta charset="utf-8"><title>Render refinement diagnostics</title>
<style>
body{font:13px system-ui;margin:0;background:#111;color:#ddd}
h2{font-size:15px;margin:18px 12px 6px;color:#9cf}
#tabs{padding:8px;background:#1b1b1b;position:sticky;top:0;z-index:9}
button{font:12px system-ui;margin:2px;padding:5px 9px;background:#2a2a2a;color:#ddd;
border:1px solid #444;border-radius:3px;cursor:pointer}
button.on{background:#3a5a8a;border-color:#6a9ada}
.grid{display:flex;flex-wrap:wrap;gap:10px;padding:8px 12px}
.cell{background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:6px}
.cell img{display:block;max-width:300px;image-rendering:pixelated;border-radius:2px}
.cap{font-size:11px;color:#9ab;padding-top:4px}
table{border-collapse:collapse;margin:8px 12px;font-size:12px}
td,th{border:1px solid #333;padding:4px 8px;text-align:right}
th{background:#1b1b1b;color:#9cf}td:first-child,th:first-child{text-align:left}
.good{color:#6d6}.bad{color:#d66}.note{padding:6px 12px;color:#9ab;font-size:12px}
</style>
<div id="tabs"></div><div id="view"></div>
<script>
const D = """ + body + """;
const tabs=document.getElementById('tabs'), view=document.getElementById('view');
function cell(src,cap){return src?`<div class="cell"><img src="${src}"><div class="cap">${cap}</div></div>`:'';}
function table(rows,cols){
  let h='<table><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr>';
  for(const r of rows){h+='<tr>'+cols.map(c=>{
    let v=r[c]; if(typeof v==='number') v=Math.abs(v)>1e4?v.toExponential(2):v.toFixed(4);
    const cls=(c.includes('gain')||c.includes('LLR'))&&typeof r[c]==='number'?(r[c]>0?'good':'bad'):'';
    return `<td class="${cls}">${v===undefined?'-':v}</td>`;}).join('')+'</tr>';}
  return h+'</table>';
}
function draw(name){
  [...tabs.children].forEach(b=>b.className=b.textContent===name?'on':'');
  const v=D.views[name]; let html='';
  if(v.note) html+=`<div class="note">${v.note}</div>`;
  if(v.table) html+=table(v.table.rows,v.table.cols);
  if(v.panels) for(const p of v.panels){
    html+=`<h2>${p.title}</h2><div class="grid">`+p.images.map(i=>cell(i.src,i.cap)).join('')+'</div>';
    if(p.table) html+=table(p.table.rows,p.table.cols);
  }
  view.innerHTML=html;
}
Object.keys(D.views).forEach(n=>{const b=document.createElement('button');
  b.textContent=n;b.onclick=()=>draw(n);tabs.appendChild(b);});
draw(Object.keys(D.views)[0]);
</script>""",
        encoding="utf-8",
    )


def image_entry(tensor, caption: str, *, cmap: str | None = None) -> dict[str, str]:
    return {"src": _png(tensor, cmap=cmap), "cap": caption}
