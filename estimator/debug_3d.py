"""Sections 25-29: the definitive 3D debug page.

CODEX_COMPLETE_PHASE2_DEBUG_AND_PARALLEL_PR_MODELS.md.

One shared world frame, one frame slider, and the ACTUAL Gaussian centers at
the selected frame for each of:

    canonical / frame 0      static context
    Phase 1                  free per-Gaussian motion
    Phase 2 generic screw    one rigid screw
    Phase 2 exact P          pure translation
    Phase 2 exact R          rotation about a finite axis

Section 26: ONE deterministic subset of the frozen moving support is used for
every layer, and its indices are saved. Resampling per model would mean the
layers are not the same Gaussians and nothing could be compared point to point.

Section 29: the Phase-1 layer is the real free-motion centers. Applying one
rigid transform to the canonical centers and calling it Phase 1 would defeat
the entire purpose, which is to see how far the free motion departs from one
rigid screw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

LAYERS = ("static_context", "canonical", "phase1", "screw", "prismatic",
          "revolute")
LAYER_COLORS = {"static_context": "#555b66", "canonical": "#8899aa",
                "phase1": "#ffd166", "screw": "#4cc9f0",
                "prismatic": "#90e07a", "revolute": "#f06595"}


def select_indices(members: torch.Tensor, max_points: int = 5000,
                   seed: int = 0) -> torch.Tensor:
    """Section 26: one deterministic subset, reused by every layer."""
    m = torch.as_tensor(members).long().reshape(-1).cpu()
    if m.numel() <= max_points:
        return m
    g = torch.Generator().manual_seed(int(seed))
    return m[torch.randperm(m.numel(), generator=g)[:max_points]].sort().values


def _frustum(cam) -> dict | None:
    """Camera centre and the four corner rays, in world coordinates."""
    try:
        V = torch.as_tensor(cam.viewmatrix).double().cpu()
        # repo convention: viewmatrix rows 0..2 hold R_cw columns, row 3 is t
        R = V[:3, :3].T
        t = V[3, :3]
        C = -(R.T @ t)
        tx, ty = float(cam.tanfovx), float(cam.tanfovy)
        corners = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            d_cam = torch.tensor([sx * tx, sy * ty, 1.0], dtype=torch.float64)
            corners.append((C + (R.T @ d_cam) * 0.35).tolist())
        return {"center": C.tolist(), "corners": corners}
    except Exception:                                   # noqa: BLE001
        return None


def build_payload(idx, mu0, layers_fn, static_idx=None, overlays=None,
                  cameras=None, frame_labels=None, max_frames=None) -> dict:
    """Collect every layer's centers at every frame into one JSON payload.

    `layers_fn` maps a layer name to `t -> [N,3]` centers for the FULL
    Gaussian array; this function does the subsetting so every layer is
    guaranteed to be the same indices (section 26).
    """
    idx = torch.as_tensor(idx).long().reshape(-1).cpu()
    T = None
    data = {}
    for name, fn in layers_fn.items():
        pts, t = [], 0
        while True:
            if max_frames is not None and t >= max_frames:
                break
            try:
                p = fn(t)
            except (IndexError, StopIteration):
                break
            if p is None:
                break
            pts.append(torch.as_tensor(p).double().cpu()[idx].numpy()
                       .astype(np.float32).round(5).tolist())
            t += 1
        if pts:
            data[name] = pts
            T = len(pts) if T is None else min(T, len(pts))
    T = T or 0
    for k in data:
        data[k] = data[k][:T]
    c0 = mu0.double().cpu()[idx]
    payload = {
        "frames": T,
        "indices": idx.tolist(),
        "layers": data,
        "canonical": c0.numpy().astype(np.float32).round(5).tolist(),
        "colors": LAYER_COLORS,
        "frame_labels": (frame_labels or [{"frame": t} for t in range(T)])[:T],
        "overlays": overlays or {},
    }
    if static_idx is not None and len(static_idx):
        s = torch.as_tensor(static_idx).long().reshape(-1).cpu()
        payload["static_context"] = (mu0.double().cpu()[s].numpy()
                                     .astype(np.float32).round(4).tolist())
    if cameras:
        fr = [_frustum(c) for c in cameras]
        payload["cameras"] = [f for f in fr if f]
    return payload


def write_3d_html(path, payload: dict, title: str) -> Path:
    """Section 25/27/28: the page itself. Three.js from the CDN, data inline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = """<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title><style>
 body{margin:0;font:13px/1.5 system-ui,sans-serif;background:#0b0d10;color:#eee;
 overflow:hidden}
 #c{position:fixed;inset:0}
 #ui{position:fixed;left:12px;top:12px;background:#15181ddd;padding:12px 14px;
 border:1px solid #2a2f37;max-width:330px;max-height:94vh;overflow:auto}
 h1{font-size:15px;margin:0 0 8px} label{display:block;margin:3px 0;cursor:pointer}
 .sw{display:inline-block;width:11px;height:11px;margin-right:6px;
 vertical-align:middle;border:1px solid #000}
 input[type=range]{width:100%} #lbl{font-family:ui-monospace,monospace;
 font-size:12px;color:#9cf;white-space:pre-wrap}
 hr{border:0;border-top:1px solid #2a2f37;margin:9px 0}
 .note{color:#8a93a0;font-size:11px}
</style></head><body>
<canvas id="c"></canvas>
<div id="ui">
 <h1>__TITLE__</h1>
 <input type="range" id="sl" min="0" max="0" value="0">
 <div id="lbl"></div>
 <hr><div id="toggles"></div>
 <hr><div id="otoggles"></div>
 <p class="note">Drag to orbit, wheel to zoom, right-drag to pan.
 All layers share one world frame and the same Gaussian indices.</p>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.min.js"></script>
<script>
const D = __PAYLOAD__;
const S = new THREE.Scene();
const cv = document.getElementById('c');
const RE = new THREE.WebGLRenderer({canvas:cv, antialias:true});
RE.setPixelRatio(devicePixelRatio); RE.setSize(innerWidth, innerHeight);
S.background = new THREE.Color(0x0b0d10);
const CAM = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.01, 2000);

// centre the view on the canonical cloud
let ctr = [0,0,0], n = D.canonical.length;
for (const p of D.canonical){ctr[0]+=p[0];ctr[1]+=p[1];ctr[2]+=p[2];}
ctr = ctr.map(v=>v/Math.max(n,1));
let rad = 0;
for (const p of D.canonical){
  rad = Math.max(rad, Math.hypot(p[0]-ctr[0],p[1]-ctr[1],p[2]-ctr[2]));}
rad = Math.max(rad, 1e-3);

let th=0.9, ph=1.1, dist=rad*3.4;
const tgt = new THREE.Vector3(...ctr);
function place(){
  CAM.position.set(tgt.x+dist*Math.sin(ph)*Math.cos(th),
                   tgt.y+dist*Math.cos(ph),
                   tgt.z+dist*Math.sin(ph)*Math.sin(th));
  CAM.lookAt(tgt);}
place();
let drag=null;
cv.addEventListener('mousedown',e=>drag={x:e.clientX,y:e.clientY,b:e.button});
addEventListener('mouseup',()=>drag=null);
addEventListener('mousemove',e=>{ if(!drag) return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  if(drag.b===0){ th+=dx*0.006; ph=Math.max(0.05,Math.min(3.09,ph+dy*0.006)); }
  else { const s=dist*0.0016;
         const r=new THREE.Vector3().subVectors(CAM.position,tgt).normalize();
         const u=new THREE.Vector3(0,1,0);
         const rt=new THREE.Vector3().crossVectors(u,r).normalize();
         const up=new THREE.Vector3().crossVectors(r,rt).normalize();
         tgt.addScaledVector(rt,-dx*s); tgt.addScaledVector(up,dy*s); }
  drag={x:e.clientX,y:e.clientY,b:drag.b}; place(); });
cv.addEventListener('wheel',e=>{e.preventDefault();
  dist*=Math.exp(e.deltaY*0.0012); place();},{passive:false});
cv.addEventListener('contextmenu',e=>e.preventDefault());
addEventListener('resize',()=>{CAM.aspect=innerWidth/innerHeight;
  CAM.updateProjectionMatrix(); RE.setSize(innerWidth,innerHeight);});

function cloud(pts,color,size){
  const g=new THREE.BufferGeometry();
  const a=new Float32Array(pts.length*3);
  pts.forEach((p,i)=>{a[i*3]=p[0];a[i*3+1]=p[1];a[i*3+2]=p[2];});
  g.setAttribute('position',new THREE.BufferAttribute(a,3));
  const m=new THREE.PointsMaterial({color:color,size:size,
    sizeAttenuation:true});
  return new THREE.Points(g,m);}

const objs={}, ui=document.getElementById('toggles');
function addLayer(key,label,pts,color,size){
  const o=cloud(pts,color,size); S.add(o); objs[key]=o;
  const id='t_'+key;
  ui.insertAdjacentHTML('beforeend',
    '<label><input type="checkbox" id="'+id+'" checked>'+
    '<span class="sw" style="background:'+color+'"></span>'+label+'</label>');
  document.getElementById(id).addEventListener('change',
    e=>{o.visible=e.target.checked;});}

const ptSize = rad*0.012;
if (D.static_context)
  addLayer('static_context','static context',D.static_context,
           D.colors.static_context, ptSize*0.7);
addLayer('canonical','canonical (frame 0)',D.canonical,D.colors.canonical,ptSize);
const NAMES={phase1:'Phase 1 (free)',screw:'Phase 2 screw',
             prismatic:'Phase 2 P',revolute:'Phase 2 R'};
for(const k of ['phase1','screw','prismatic','revolute'])
  if(D.layers[k]) addLayer(k,NAMES[k],D.layers[k][0],D.colors[k],ptSize);

// ---- overlays: COMs, axes, frustums (section 27) ----------------------
const ov={}, oui=document.getElementById('otoggles');
function addOverlay(key,label,obj,color){
  S.add(obj); ov[key]=obj; const id='o_'+key;
  oui.insertAdjacentHTML('beforeend',
    '<label><input type="checkbox" id="'+id+'" checked>'+
    '<span class="sw" style="background:'+color+'"></span>'+label+'</label>');
  document.getElementById(id).addEventListener('change',
    e=>{obj.visible=e.target.checked;});}
function seg(a,b,color){
  const g=new THREE.BufferGeometry().setFromPoints(
    [new THREE.Vector3(...a),new THREE.Vector3(...b)]);
  return new THREE.Line(g,new THREE.LineBasicMaterial({color:color}));}
for(const [k,o] of Object.entries(D.overlays||{})){
  if(o.type==='line') addOverlay(k,o.label,seg(o.a,o.b,o.color),o.color);
  if(o.type==='points'){
    const p=cloud(o.points,o.color,ptSize*2.2); addOverlay(k,o.label,p,o.color);}
  if(o.type==='path'){
    const g=new THREE.BufferGeometry().setFromPoints(
      o.points.map(q=>new THREE.Vector3(...q)));
    addOverlay(k,o.label,new THREE.Line(g,
      new THREE.LineBasicMaterial({color:o.color})),o.color);}}
if(D.cameras&&D.cameras.length){
  const grp=new THREE.Group();
  D.cameras.forEach(f=>{ f.corners.forEach((c,i)=>{
      grp.add(seg(f.center,c,0xaaaaaa));
      grp.add(seg(c,f.corners[(i+1)%4],0xaaaaaa)); }); });
  addOverlay('frustums','camera frustums',grp,'#aaaaaa');}

// ---- frame slider (section 28) ---------------------------------------
const sl=document.getElementById('sl'), lbl=document.getElementById('lbl');
sl.max=Math.max(0,D.frames-1);
function setFrame(t){
  for(const k of ['phase1','screw','prismatic','revolute']){
    if(!D.layers[k]||!objs[k]) continue;
    const pts=D.layers[k][Math.min(t,D.layers[k].length-1)];
    const a=objs[k].geometry.attributes.position.array;
    pts.forEach((p,i)=>{a[i*3]=p[0];a[i*3+1]=p[1];a[i*3+2]=p[2];});
    objs[k].geometry.attributes.position.needsUpdate=true;}
  const L=D.frame_labels[t]||{frame:t};
  lbl.textContent=Object.entries(L).map(([k,v])=>k+': '+v).join('\\n');}
sl.addEventListener('input',e=>setFrame(+e.target.value));
setFrame(0);
(function loop(){requestAnimationFrame(loop); RE.render(S,CAM);})();
</script></body></html>"""
    doc = doc.replace("__TITLE__", title).replace(
        "__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    path.write_text(doc)
    return path
