"""The four-motion joint debugger: Phase 1 / screw / exact P / exact R.

Modelled on `debug/geometric_motion_debug.py` -- same shell, same controls,
same dark monospace dashboard -- but the stages are the four motion models
this pipeline actually produces, in one shared world frame:

    Canonical    frame 0, the state every model starts from
    Phase 1      free per-Gaussian motion, no joint model at all
    Screw        one generic rigid screw   T(q) = Exp_SE3(q xi^)
    Exact P      mu_i(t) = mu_i^0 + s_t e_P
    Exact R      mu_i(t) = a_R + R(e_R, theta_t)(mu_i^0 - a_R)

Two things it shows that the previous page did not:

* The MOVING SUPPORT, coloured against the static remainder. The support is
  upstream of every joint estimate -- the joint is fitted to whatever it
  contains -- so being able to see it is the difference between debugging the
  joint and debugging the thing that determined it.
* All four motions at once ("overlay"), so the departure of the free Phase-1
  field from one rigid screw, and of the screw from P and R, is visible
  directly rather than by flipping between stages.

Positions are baked per frame for all four stages. Phase 1 has no transform by
construction, and baking the other three keeps one code path instead of three
transport laws re-implemented in JavaScript -- where they could drift from the
estimator's own.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.min.js"

STAGES = [("canonical", "Canonical (frame 0)"), ("phase1", "Phase 1 (free)"),
          ("screw", "Phase 2 screw"), ("prismatic", "Phase 2 exact P"),
          ("revolute", "Phase 2 exact R")]
STAGE_COLORS = {"canonical": "#8899aa", "phase1": "#ffd166",
                "screw": "#4cc9f0", "prismatic": "#90e07a",
                "revolute": "#f06595", "static": "#4a515c"}


def _pts(x, idx) -> list:
    a = torch.as_tensor(x).double().cpu()[idx].numpy().astype(np.float32)
    return [round(float(v), 5) for v in a.reshape(-1)]


def build_payload(mu0, members, stage_positions, *, static_idx=None,
                  overlays=None, cameras=None, frame_labels=None,
                  max_points: int = 6000, seed: int = 0,
                  support_evidence=None, meta=None) -> dict:
    """`stage_positions` maps a stage name to a list of [N,3] per frame."""
    members = torch.as_tensor(members).long().reshape(-1).cpu()
    g = torch.Generator().manual_seed(int(seed))
    idx = members
    if idx.numel() > max_points:
        idx = idx[torch.randperm(idx.numel(), generator=g)[:max_points]].sort().values

    sidx = torch.as_tensor([] if static_idx is None else static_idx).long().reshape(-1)
    if sidx.numel() > max_points:
        g2 = torch.Generator().manual_seed(int(seed) + 1)
        sidx = sidx[torch.randperm(sidx.numel(), generator=g2)[:max_points]]

    stages, T = {}, None
    for name, seq in (stage_positions or {}).items():
        if not seq:
            continue
        stages[name] = [_pts(p, idx) for p in seq]
        T = len(stages[name]) if T is None else min(T, len(stages[name]))
    T = T or 1
    for k in stages:
        stages[k] = stages[k][:T]

    return {
        "frames": T,
        "indices": idx.tolist(),
        "canonical": _pts(mu0, idx),
        "static": _pts(mu0, sidx) if sidx.numel() else [],
        "stages": stages,
        "stage_list": [[k, t] for k, t in STAGES
                       if k == "canonical" or k in stages],
        "colors": STAGE_COLORS,
        "overlays": overlays or {},
        "cameras": cameras or [],
        "frame_labels": (frame_labels or [{"frame": t} for t in range(T)])[:T],
        "support": support_evidence or {},
        "meta": meta or {},
        "num_shown_moving": int(idx.numel()),
        "num_shown_static": int(sidx.numel()),
    }


def write_html(path, payload: dict, title: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = (_TEMPLATE
            .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
            .replace("__THREE__", THREE_CDN)
            .replace("__TITLE__", title))
    path.write_text(html)
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
 :root{--bg:#0e1116;--panel:#161b22;--line:#2a313a;--fg:#e6edf3;--dim:#8b949e;
       --accent:#58a6ff;--warn:#f0883e;--bad:#f85149;--good:#3fb950;}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;overflow:hidden}
 #app{display:flex;height:100vh;width:100vw}
 #view{flex:1;position:relative;min-width:0}
 canvas{display:block}
 #side{width:370px;flex:none;background:var(--panel);
       border-left:1px solid var(--line);overflow-y:auto;padding:12px}
 @media(max-width:900px){#app{flex-direction:column}
  #side{width:100%;height:46vh;border-left:none;border-top:1px solid var(--line)}}
 h1{font-size:14px;margin:0 0 4px} h2{font-size:12px;margin:14px 0 6px;
  color:var(--accent);text-transform:uppercase;letter-spacing:.06em}
 .row{display:flex;justify-content:space-between;gap:8px;padding:1px 0}
 .row span:last-child{color:var(--dim);text-align:right;word-break:break-all}
 label{display:flex;align-items:center;gap:6px;padding:1px 0;cursor:pointer}
 select,button{background:#21262d;color:var(--fg);border:1px solid var(--line);
  border-radius:5px;padding:4px 8px;font:inherit;cursor:pointer}
 button:hover{border-color:var(--accent)}
 #bar{position:absolute;left:0;right:0;bottom:0;background:rgba(13,17,23,.93);
  border-top:1px solid var(--line);padding:8px 12px;display:flex;
  align-items:center;gap:10px;flex-wrap:wrap}
 #slider{flex:1;min-width:160px}
 #hud{position:absolute;top:10px;left:12px;background:rgba(13,17,23,.85);
  border:1px solid var(--line);border-radius:6px;padding:8px 10px;
  pointer-events:none;font-size:12px;white-space:pre}
 .sw{display:inline-block;width:11px;height:11px;border:1px solid #000;
  vertical-align:middle}
 .note{color:var(--dim);font-size:11px;margin:6px 0}
 .bad{color:var(--bad)} .good{color:var(--good)} .warn{color:var(--warn)}
 table{width:100%;border-collapse:collapse;font-size:11px}
 td,th{padding:2px 4px;border-bottom:1px solid var(--line);text-align:right}
 th:first-child,td:first-child{text-align:left}
</style></head><body>
<div id="app">
 <div id="view"><canvas id="c"></canvas>
  <div id="hud"></div>
  <div id="bar">
   <button id="play">play</button>
   <input type="range" id="slider" min="0" max="0" value="0">
   <span id="fnum">0</span>
   <label style="gap:4px"><input type="checkbox" id="loop" checked>loop</label>
   <select id="speed"><option value="1">1x</option><option value="2">2x</option>
    <option value="4">4x</option><option value="0.5">0.5x</option></select>
  </div></div>
 <div id="side">
  <h1>__TITLE__</h1>
  <div class="note" id="meta"></div>
  <h2>Stage</h2>
  <select id="stage"></select>
  <label><input type="checkbox" id="overlay">overlay all four motions</label>
  <label><input type="checkbox" id="ghost" checked>ghost canonical</label>
  <h2>Show</h2><div id="toggles"></div>
  <h2>Moving support</h2><div id="sup"></div>
  <h2>Dashboard</h2><div id="dash"></div>
  <h2>Joint geometry</h2><div id="joint"></div>
 </div></div>
<script src="__THREE__"></script>
<script>
const D = __PAYLOAD__;
const S = new THREE.Scene(), cv = document.getElementById('c');
const RE = new THREE.WebGLRenderer({canvas:cv,antialias:true});
S.background = new THREE.Color(0x0e1116);
const CAM = new THREE.PerspectiveCamera(55,1,0.01,4000);
function fit(){const w=document.getElementById('view').clientWidth,
 h=document.getElementById('view').clientHeight;
 RE.setPixelRatio(devicePixelRatio);RE.setSize(w,h);
 CAM.aspect=w/Math.max(h,1);CAM.updateProjectionMatrix();}

let ctr=[0,0,0],N=D.canonical.length/3;
for(let i=0;i<N;i++){ctr[0]+=D.canonical[i*3];ctr[1]+=D.canonical[i*3+1];
 ctr[2]+=D.canonical[i*3+2];}
ctr=ctr.map(v=>v/Math.max(N,1));
let rad=1e-3;
for(let i=0;i<N;i++)rad=Math.max(rad,Math.hypot(D.canonical[i*3]-ctr[0],
 D.canonical[i*3+1]-ctr[1],D.canonical[i*3+2]-ctr[2]));
let th=0.9,ph=1.1,dist=rad*3.2;const tgt=new THREE.Vector3(...ctr);
function place(){CAM.position.set(tgt.x+dist*Math.sin(ph)*Math.cos(th),
 tgt.y+dist*Math.cos(ph),tgt.z+dist*Math.sin(ph)*Math.sin(th));CAM.lookAt(tgt);}
let drag=null;
cv.addEventListener('mousedown',e=>drag={x:e.clientX,y:e.clientY,b:e.button});
addEventListener('mouseup',()=>drag=null);
addEventListener('mousemove',e=>{if(!drag)return;
 const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
 if(drag.b===0){th+=dx*0.006;ph=Math.max(0.05,Math.min(3.09,ph+dy*0.006));}
 else{const s=dist*0.0016,r=new THREE.Vector3().subVectors(CAM.position,tgt).normalize(),
  u=new THREE.Vector3(0,1,0),rt=new THREE.Vector3().crossVectors(u,r).normalize(),
  up=new THREE.Vector3().crossVectors(r,rt).normalize();
  tgt.addScaledVector(rt,-dx*s);tgt.addScaledVector(up,dy*s);}
 drag={x:e.clientX,y:e.clientY,b:drag.b};place();});
cv.addEventListener('wheel',e=>{e.preventDefault();dist*=Math.exp(e.deltaY*0.0012);
 place();},{passive:false});
cv.addEventListener('contextmenu',e=>e.preventDefault());
addEventListener('resize',fit);fit();place();

const PS=rad*0.011;
function cloud(flat,color,size,op){
 const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(flat),3));
 return new THREE.Points(g,new THREE.PointsMaterial({color:color,size:size,
  sizeAttenuation:true,transparent:op<1,opacity:op}));}

const OBJ={};
if(D.static.length){OBJ.static=cloud(D.static,D.colors.static,PS*0.7,0.5);
 S.add(OBJ.static);}
OBJ.canonical=cloud(D.canonical,D.colors.canonical,PS,0.35);S.add(OBJ.canonical);
for(const k of Object.keys(D.stages))
 {OBJ[k]=cloud(D.stages[k][0],D.colors[k],PS,1.0);S.add(OBJ[k]);}

// overlays: joint axes, COM trails, frustums
const OV={},ot=document.getElementById('toggles');
function seg(a,b,c){const g=new THREE.BufferGeometry().setFromPoints(
 [new THREE.Vector3(...a),new THREE.Vector3(...b)]);
 return new THREE.Line(g,new THREE.LineBasicMaterial({color:c}));}
function addOv(k,label,obj,color){S.add(obj);OV[k]=obj;
 ot.insertAdjacentHTML('beforeend','<label><input type="checkbox" id="ov_'+k+
  '" checked><span class="sw" style="background:'+color+'"></span>'+label+'</label>');
 document.getElementById('ov_'+k).addEventListener('change',e=>obj.visible=e.target.checked);}
for(const [k,o] of Object.entries(D.overlays||{})){
 if(o.type==='line')addOv(k,o.label,seg(o.a,o.b,o.color),o.color);
 else if(o.type==='path'){const g=new THREE.BufferGeometry().setFromPoints(
   o.points.map(q=>new THREE.Vector3(...q)));
  addOv(k,o.label,new THREE.Line(g,new THREE.LineBasicMaterial({color:o.color})),o.color);}}
if(D.cameras&&D.cameras.length){const grp=new THREE.Group();
 D.cameras.forEach(f=>f.corners.forEach((c,i)=>{grp.add(seg(f.center,c,0x7d8590));
  grp.add(seg(c,f.corners[(i+1)%4],0x7d8590));}));
 addOv('frustums','camera frustums',grp,'#7d8590');}
ot.insertAdjacentHTML('beforeend','<label><input type="checkbox" id="ov_static"'+
 (D.static.length?' checked':' disabled')+'><span class="sw" style="background:'+
 D.colors.static+'"></span>static (non-support) Gaussians</label>');
document.getElementById('ov_static').addEventListener('change',e=>{
 if(OBJ.static)OBJ.static.visible=e.target.checked;});

// stage selector
const sel=document.getElementById('stage');
D.stage_list.forEach(([v,t])=>{const o=document.createElement('option');
 o.value=v;o.textContent=t;sel.appendChild(o);});
sel.value=D.stages.screw?'screw':(D.stage_list[1]||D.stage_list[0])[0];
function applyVis(){
 const ov=document.getElementById('overlay').checked,cur=sel.value;
 for(const k of Object.keys(D.stages))OBJ[k].visible=ov||k===cur;
 OBJ.canonical.visible=document.getElementById('ghost').checked||cur==='canonical';
 OBJ.canonical.material.opacity=(cur==='canonical'&&!ov)?1.0:0.35;}
sel.addEventListener('change',()=>{applyVis();draw(F);});
document.getElementById('overlay').addEventListener('change',applyVis);
document.getElementById('ghost').addEventListener('change',applyVis);

// support panel -- the thing upstream of every joint estimate
const sp=D.support||{};
document.getElementById('sup').innerHTML=
 (sp.num_members!==undefined?
  '<div class="row"><span>members</span><span>'+sp.num_members+' / '+sp.num_total+
  ' ('+(100*sp.kept_fraction).toFixed(1)+'%)</span></div>'+
  '<div class="row"><span>method</span><span>'+sp.method+'</span></div>'+
  '<div class="row"><span>threshold</span><span>'+(+sp.threshold).toFixed(4)+'</span></div>'+
  '<div class="row"><span>separability</span><span class="'+
   (sp.separability>0.5?'good':'warn')+'">'+(+sp.separability).toFixed(2)+'</span></div>'+
  '<div class="row"><span>selected median excursion</span><span>'+
   (+(sp.excursion||{}).selected_median).toFixed(4)+'</span></div>'+
  '<div class="row"><span>rejected mean excursion</span><span>'+
   (+(sp.excursion||{}).rejected_mean).toFixed(5)+'</span></div>'+
  '<div class="row"><span>contrast</span><span>'+
   (+(sp.excursion||{}).contrast).toFixed(1)+'x</span></div>'+
  '<div class="note">'+(sp.note||'')+'</div>'
 :'<div class="note">no support evidence recorded</div>');
document.getElementById('meta').textContent=
 Object.entries(D.meta||{}).map(([k,v])=>k+': '+v).join('   ');
document.getElementById('joint').innerHTML=
 Object.entries(D.overlays||{}).filter(([,o])=>o.info)
  .map(([,o])=>'<div class="row"><span>'+o.label+'</span><span>'+o.info+'</span></div>')
  .join('')||'<div class="note">no joint geometry recorded</div>';

let F=0;
function draw(t){
 F=t;document.getElementById('fnum').textContent=t;
 document.getElementById('slider').value=t;
 for(const k of Object.keys(D.stages)){
  const src=D.stages[k][Math.min(t,D.stages[k].length-1)];
  const a=OBJ[k].geometry.attributes.position.array;
  a.set(src);OBJ[k].geometry.attributes.position.needsUpdate=true;}
 const L=D.frame_labels[t]||{frame:t};
 document.getElementById('hud').textContent=
  Object.entries(L).map(([k,v])=>k.padEnd(16)+v).join('\n');
 document.getElementById('dash').innerHTML='<table>'+
  Object.entries(L).map(([k,v])=>'<tr><td>'+k+'</td><td>'+v+'</td></tr>').join('')+
  '</table>';}
const sl=document.getElementById('slider');sl.max=Math.max(0,D.frames-1);
sl.addEventListener('input',e=>{playing=false;
 document.getElementById('play').textContent='play';draw(+e.target.value);});
let playing=false,acc=0,last=performance.now();
document.getElementById('play').addEventListener('click',e=>{
 playing=!playing;e.target.textContent=playing?'pause':'play';});
applyVis();draw(0);
(function loop(now){requestAnimationFrame(loop);
 const dt=(now-last)/1000;last=now;
 if(playing){acc+=dt*12*parseFloat(document.getElementById('speed').value);
  while(acc>=1){acc-=1;let n=F+1;
   if(n>=D.frames){if(document.getElementById('loop').checked)n=0;
    else{n=D.frames-1;playing=false;
     document.getElementById('play').textContent='play';}}
   draw(n);}}
 RE.render(S,CAM);})(performance.now());
</script></body></html>"""
