"""Sections 23, 40, 41: the rigid Phase-2 and joint-reduction debug pages.

CODEX_IMPLEMENT_RIGID_SE3_PHASE2_AND_POSTHOC_JOINT_REDUCTION.md.

Three pages:

  rigid_phase2_debug.html   section 23. The rigid reconstruction ALONE, with
                            NO joint axis anywhere -- the point is to judge
                            whether the rigid body was recovered, separately
                            from whether a joint explains it.
  rigid_to_joint_debug.html section 40. Stage A repeats the rigid layers;
                            Stage B adds the P direction, the R axis and the
                            GT axis as overlays over the SAME Phase-2 centers.
  rigid_timelines.html      section 41.

Every 3D page shares one world frame and one frame slider, and every layer is
drawn at the SAME deterministic subset of Gaussians, so a point can be
followed from layer to layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

LAYER_COLORS = {
    "static_context": "#454b55",
    "canonical": "#8899aa",
    "phase1": "#ffd166",
    "kabsch": "#4cc9f0",
    "phase2": "#f06595",
}
LAYER_LABELS = {
    "static_context": "canonical static scene",
    "canonical": "coherence moving support (frame 0)",
    "phase1": "Phase 1 free motion",
    "kabsch": "Phase 1 Kabsch rigid",
    "phase2": "Phase 2 RGB-D rigid",
}


def _subset(members: torch.Tensor, n_total: int, max_points: int, seed=0):
    """One deterministic subset of the moving support, plus static context.

    Section 26 of the earlier debug spec and the same reason here: resampling
    per layer means the layers are not the same Gaussians and nothing can be
    compared point to point.
    """
    m = torch.as_tensor(members).long().reshape(-1).cpu()
    g = torch.Generator().manual_seed(int(seed))
    budget = int(max_points)
    mov = m if m.numel() <= budget // 2 else m[
        torch.randperm(m.numel(), generator=g)[: budget // 2]].sort().values
    mask = torch.ones(n_total, dtype=torch.bool)
    mask[m] = False
    stat = torch.nonzero(mask, as_tuple=False).reshape(-1)
    room = budget - mov.numel()
    if stat.numel() > room:
        stat = stat[torch.randperm(stat.numel(), generator=g)[:room]].sort().values
    return mov, stat


def _rigid_layers(mu0, members, positions, part, kabsch_R, kabsch_p,
                  mov, stat):
    """Per-frame positions for each layer, at the shared subset."""
    T = part.num_frames
    mu0 = torch.as_tensor(mu0).double().cpu()
    out = {k: [] for k in ("phase1", "kabsch", "phase2")}
    for t in range(T):
        p1 = torch.as_tensor(positions[t]).double().cpu()[mov]
        out["phase1"].append(p1)
        Rk, pk = kabsch_R[t].double().cpu(), kabsch_p[t].double().cpu()
        out["kabsch"].append(mu0[mov] @ Rk.T + pk)
        R2, p2 = part.R[t].double().cpu(), part.p[t].double().cpu()
        out["phase2"].append(mu0[mov] @ R2.T + p2)
    return {k: [v.numpy().round(5).reshape(-1).tolist() for v in vs]
            for k, vs in out.items()}


def _com(frames_flat):
    out = []
    for f in frames_flat:
        a = np.asarray(f, dtype=np.float64).reshape(-1, 3)
        out.append(a.mean(0).round(5).tolist())
    return out


def _frustums(cameras, limit=8):
    from estimator.debug_3d import _frustum
    out = []
    for cam in (cameras or [])[:limit]:
        f = _frustum(cam)
        if f is not None:
            out.append(f)
    return out


def _base_payload(mu0, members, positions, part, kabsch_R, kabsch_p,
                  cameras, max_points, seed=0):
    mu0 = torch.as_tensor(mu0).double().cpu()
    n = mu0.shape[0]
    mov, stat = _subset(members, n, max_points, seed)
    layers = _rigid_layers(mu0, members, positions, part, kabsch_R, kabsch_p,
                           mov, stat)
    rv = _rotvec_series(part.R)
    return {
        "static": mu0[stat].numpy().round(5).reshape(-1).tolist(),
        "canonical": mu0[mov].numpy().round(5).reshape(-1).tolist(),
        "layers": layers,
        "com": {k: _com(v) for k, v in layers.items()},
        "velocity": {"v": part.v.numpy().round(5).tolist(),
                     "omega": part.omega.numpy().round(5).tolist()},
        "rotvec": rv.numpy().round(6).tolist(),
        "translation": part.p.numpy().round(6).tolist(),
        "frames": part.num_frames,
        "num_moving_shown": int(mov.numel()),
        "num_static_shown": int(stat.numel()),
        "num_moving_total": int(torch.as_tensor(members).numel()),
        "num_gaussians": int(n),
        "frustums": _frustums(cameras),
        "colors": LAYER_COLORS, "labels": LAYER_LABELS,
    }


def _rotvec_series(R):
    from estimator.manifold import so3_log
    return so3_log(torch.as_tensor(R).double().reshape(-1, 3, 3))


# ---------------------------------------------------------------------------
# the shared viewer shell
# ---------------------------------------------------------------------------

_SHELL = r"""<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title><style>
 body{margin:0;background:#0e1116;color:#e6edf3;overflow:hidden;
  font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
 #app{display:flex;height:100vh} #view{flex:1;position:relative}
 #side{width:360px;background:#161b22;border-left:1px solid #2a313a;
  padding:12px;overflow-y:auto}
 h1{font-size:14px;margin:0 0 6px} h2{font-size:12px;color:#58a6ff;
  text-transform:uppercase;margin:16px 0 6px}
 label{display:flex;gap:6px;align-items:center;cursor:pointer}
 .sw{width:11px;height:11px;display:inline-block;border:1px solid #000;
  flex:0 0 auto}
 .row{display:flex;justify-content:space-between;gap:8px}
 .row span:last-child{color:#8b949e;white-space:nowrap}
 #bar{position:absolute;left:0;right:0;bottom:0;padding:8px 12px;
  background:rgba(13,17,23,.93);border-top:1px solid #2a313a;display:flex;
  gap:10px;align-items:center}
 input[type=range]{flex:1} .note{color:#8b949e;font-size:11px}
 button{background:#21262d;color:#e6edf3;border:1px solid #2a313a;
  border-radius:4px;padding:3px 9px;cursor:pointer;font:inherit}
 .tab{display:inline-block;padding:4px 10px;border:1px solid #2a313a;
  border-radius:4px;margin-right:4px;cursor:pointer;background:#21262d}
 .tab.on{background:#1f6feb;border-color:#1f6feb}
 table{width:100%;border-collapse:collapse;font-size:11px}
 td,th{padding:2px 4px;text-align:right}th{color:#58a6ff;font-weight:400}
 td:first-child,th:first-child{text-align:left}
</style></head><body><div id="app">
<div id="view"><canvas id="c"></canvas>
 <div id="bar"><button id="play">play</button>
 <input type="range" id="sl" min="0" max="0" value="0">
 <span id="fn">frame 0</span></div></div>
<div id="side"><h1>__TITLE__</h1>
__PANEL__
</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.min.js"></script>
<script>
const D=__PAYLOAD__;
const S=new THREE.Scene(),cv=document.getElementById('c');
const RE=new THREE.WebGLRenderer({canvas:cv,antialias:true});
S.background=new THREE.Color(0x0e1116);
const CAM=new THREE.PerspectiveCamera(55,1,0.001,8000);
function fit(){const v=document.getElementById('view');
 RE.setPixelRatio(devicePixelRatio);RE.setSize(v.clientWidth,v.clientHeight);
 CAM.aspect=v.clientWidth/Math.max(v.clientHeight,1);CAM.updateProjectionMatrix();}
const ALL=D.static.concat(D.canonical); const NA=ALL.length/3;
let ctr=[0,0,0];
for(let i=0;i<NA;i++){ctr[0]+=ALL[i*3];ctr[1]+=ALL[i*3+1];ctr[2]+=ALL[i*3+2];}
ctr=ctr.map(v=>v/Math.max(NA,1)); let rad=1e-3;
for(let i=0;i<NA;i++)rad=Math.max(rad,Math.hypot(ALL[i*3]-ctr[0],
 ALL[i*3+1]-ctr[1],ALL[i*3+2]-ctr[2]));
let th=.9,ph=1.1,dist=rad*3.0; const tgt=new THREE.Vector3(...ctr);
function place(){CAM.position.set(tgt.x+dist*Math.sin(ph)*Math.cos(th),
 tgt.y+dist*Math.cos(ph),tgt.z+dist*Math.sin(ph)*Math.sin(th));CAM.lookAt(tgt);}
let drag=null;cv.addEventListener('mousedown',e=>drag={x:e.clientX,y:e.clientY,b:e.button});
addEventListener('mouseup',()=>drag=null);
addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
 if(drag.b===0){th+=dx*.006;ph=Math.max(.05,Math.min(3.09,ph+dy*.006));}
 else{const s=dist*.0016,r=new THREE.Vector3().subVectors(CAM.position,tgt).normalize(),
  u=new THREE.Vector3(0,1,0),rt=new THREE.Vector3().crossVectors(u,r).normalize(),
  up=new THREE.Vector3().crossVectors(r,rt).normalize();
  tgt.addScaledVector(rt,-dx*s);tgt.addScaledVector(up,dy*s);}
 drag={x:e.clientX,y:e.clientY,b:drag.b};place();});
cv.addEventListener('wheel',e=>{e.preventDefault();dist*=Math.exp(e.deltaY*.0012);place();},{passive:false});
cv.addEventListener('contextmenu',e=>e.preventDefault());
addEventListener('resize',fit);fit();place();

function pts(flat,color,size,op){
 const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(flat),3));
 const o=new THREE.Points(g,new THREE.PointsMaterial({color:color,
  size:rad*size,sizeAttenuation:true,transparent:true,opacity:op||1}));
 S.add(o);return o;}
function lines(arr,color,op,width){
 const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(arr),3));
 const o=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:color,
  transparent:true,opacity:op===undefined?1:op}));
 S.add(o);return o;}
function polyline(points,color,op){
 const arr=[];for(let i=0;i+1<points.length;i++){
  arr.push(points[i][0],points[i][1],points[i][2],
           points[i+1][0],points[i+1][1],points[i+1][2]);}
 return lines(arr,color,op);}

const OB={};
OB.static_context=pts(D.static,parseInt(D.colors.static_context.slice(1),16),0.010,0.45);
OB.canonical=pts(D.canonical,parseInt(D.colors.canonical.slice(1),16),0.014,0.6);
for(const k of ['phase1','kabsch','phase2'])
 OB[k]=pts(D.layers[k][0],parseInt(D.colors[k].slice(1),16),0.018,1.0);
for(const k of ['phase1','kabsch','phase2'])
 OB['com_'+k]=polyline(D.com[k],parseInt(D.colors[k].slice(1),16),0.95);
// rigid velocity at the current frame, drawn from the Phase-2 centre of mass
OB.velocity=lines([0,0,0,0,0,0],0x7ee787,1.0);
if(D.frustums&&D.frustums.length){
 const arr=[];D.frustums.forEach(f=>{const c=f.center;
  f.corners.forEach(k=>{arr.push(c[0],c[1],c[2],k[0],k[1],k[2]);});
  for(let i=0;i<4;i++){const a=f.corners[i],b=f.corners[(i+1)%4];
   arr.push(a[0],a[1],a[2],b[0],b[1],b[2]);}});
 OB.cameras=lines(arr,0x30363d,0.8);}
__EXTRA_OBJECTS__

function setFrame(t){
 for(const k of ['phase1','kabsch','phase2']){
  const a=OB[k].geometry.attributes.position.array,f=D.layers[k][t];
  for(let i=0;i<a.length;i++)a[i]=f[i];
  OB[k].geometry.attributes.position.needsUpdate=true;}
 const c=D.com.phase2[t],v=D.velocity.v[t];
 const s=rad*0.5/Math.max(1e-9,Math.hypot(v[0],v[1],v[2]));
 const va=OB.velocity.geometry.attributes.position.array;
 va[0]=c[0];va[1]=c[1];va[2]=c[2];
 va[3]=c[0]+v[0]*s;va[4]=c[1]+v[1]*s;va[5]=c[2]+v[2]*s;
 OB.velocity.geometry.attributes.position.needsUpdate=true;
 document.getElementById('fn').textContent='frame '+t;
 __EXTRA_FRAME__
}
const sl=document.getElementById('sl');sl.max=D.frames-1;
sl.addEventListener('input',e=>setFrame(+e.target.value));
let timer=null;
document.getElementById('play').addEventListener('click',e=>{
 if(timer){clearInterval(timer);timer=null;e.target.textContent='play';return;}
 e.target.textContent='stop';
 timer=setInterval(()=>{sl.value=(+sl.value+1)%D.frames;setFrame(+sl.value);},110);});
function toggles(host,keys){
 const h=document.getElementById(host);h.innerHTML='';
 keys.forEach(([k,label,color])=>{
  if(!OB[k])return;
  h.insertAdjacentHTML('beforeend','<label><input type="checkbox" checked '+
   'data-k="'+k+'"><span class="sw" style="background:'+color+'"></span>'+
   label+'</label>');});
 h.querySelectorAll('input').forEach(e=>e.addEventListener('change',ev=>{
  OB[ev.target.dataset.k].visible=ev.target.checked;}));}
__PANEL_JS__
setFrame(0);
(function loop(){requestAnimationFrame(loop);RE.render(S,CAM);})();
</script></body></html>"""


def _emit(path, title, payload, panel, panel_js, extra_objects="",
          extra_frame=""):
    html = (_SHELL.replace("__TITLE__", title)
            .replace("__PANEL__", panel)
            .replace("__PANEL_JS__", panel_js)
            .replace("__EXTRA_OBJECTS__", extra_objects)
            .replace("__EXTRA_FRAME__", extra_frame)
            .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html)
    return p


# ---------------------------------------------------------------------------
# Section 23: rigid reconstruction, with NO joint axis
# ---------------------------------------------------------------------------

def write_rigid_phase2_3d(path, mu0, members, positions, part, kabsch_R,
                          kabsch_p, cameras=None, max_points=6000,
                          title="rigid SE(3) Phase 2", seed=0) -> Path:
    """Section 23. Deliberately shows no axis: the question this page answers
    is whether the RIGID BODY was recovered, which has to be judged before and
    separately from whether a joint explains it."""
    D = _base_payload(mu0, members, positions, part, kabsch_R, kabsch_p,
                      cameras, max_points, seed)
    panel = """
<div id="counts" class="note"></div>
<h2>Layers</h2><div id="tg"></div>
<h2>Pose at this frame</h2><div id="pose"></div>
<div class="note" style="margin-top:14px">
Section 23: no joint axis is drawn here, on purpose. Judge the rigid
reconstruction first; the axis lives in rigid_to_joint_debug.html.
</div>"""
    panel_js = """
toggles('tg',[['static_context',D.labels.static_context,D.colors.static_context],
 ['canonical',D.labels.canonical,D.colors.canonical],
 ['phase1',D.labels.phase1,D.colors.phase1],
 ['kabsch',D.labels.kabsch,D.colors.kabsch],
 ['phase2',D.labels.phase2,D.colors.phase2],
 ['com_phase1','COM trail: Phase 1',D.colors.phase1],
 ['com_kabsch','COM trail: Kabsch',D.colors.kabsch],
 ['com_phase2','COM trail: Phase 2',D.colors.phase2],
 ['velocity','rigid velocity','#7ee787'],
 ['cameras','camera frustums','#30363d']]);
document.getElementById('counts').textContent=
 'showing '+D.num_moving_shown+' of '+D.num_moving_total+' moving and '+
 D.num_static_shown+' static Gaussians (of '+D.num_gaussians+' total)';
"""
    extra_frame = """
 const rv=D.rotvec[t],tr=D.translation[t],om=D.velocity.omega[t];
 const ang=Math.hypot(rv[0],rv[1],rv[2]);
 document.getElementById('pose').innerHTML=
  [['translation p','['+tr.map(x=>x.toFixed(4)).join(', ')+']'],
   ['rotation vector','['+rv.map(x=>x.toFixed(4)).join(', ')+']'],
   ['rotation angle',ang.toFixed(5)+' rad ('+(ang*180/Math.PI).toFixed(2)+' deg)'],
   ['linear velocity','['+v.map(x=>x.toFixed(4)).join(', ')+']'],
   ['angular velocity','['+om.map(x=>x.toFixed(4)).join(', ')+']']]
  .map(([a,b])=>'<div class="row"><span>'+a+'</span><span>'+b+'</span></div>')
  .join('');
"""
    return _emit(path, title, D, panel, panel_js, extra_frame=extra_frame)


# ---------------------------------------------------------------------------
# Section 40: Stage A rigid + Stage B joint reduction
# ---------------------------------------------------------------------------

def write_rigid_to_joint_3d(path, mu0, members, positions, part, kabsch_R,
                            kabsch_p, reduction, gt_eval=None,
                            max_points=6000,
                            title="rigid reconstruction and joint reduction",
                            seed=0) -> Path:
    """Section 40. Stage B adds the axes as OVERLAYS -- the Phase-2 centers
    are never replaced by a reduced-model reconstruction, because then the
    page could no longer show where the reduction disagrees with the data."""
    D = _base_payload(mu0, members, positions, part, kabsch_R, kabsch_p,
                      None, max_points, seed)
    mu0 = torch.as_tensor(mu0).double().cpu()
    c0 = mu0[torch.as_tensor(members).long().reshape(-1)].mean(0)
    scale = float((mu0 - mu0.mean(0)).norm(dim=1).max())
    P, R = reduction["prismatic"], reduction["revolute"]
    D["reduction"] = {
        "scale": scale,
        "centroid": c0.tolist(),
        "prismatic": {"direction": P["direction"], "s": P["s"],
                      "energy": P["pose_residual"]["weighted_energy"],
                      "rank1": P.get("translation_rank1_fraction")},
        "revolute": {"axis": R["axis"], "axis_point": R["axis_point"],
                     "theta": R["theta"],
                     "energy": R["pose_residual"]["weighted_energy"],
                     "rank1": R.get("rank1_fraction"),
                     "cond": R["finite_axis_solver"]["condition_number"],
                     "radius": R.get("radius_from_centroid")},
        "one_dof": reduction["one_dof"]["gamma_1dof"],
        "singular_values": reduction["one_dof"]["singular_values"],
    }
    # reduced-model reconstructions, as OPTIONAL overlays (section 40)
    mov, _stat = _subset(members, mu0.shape[0], max_points, seed)
    for name, fit in (("p_model", P), ("r_model", R)):
        Rm = torch.as_tensor(fit["poses_R"]).double()
        pm = torch.as_tensor(fit["poses_p"]).double()
        D["layers"][name] = [
            (mu0[mov] @ Rm[t].T + pm[t]).numpy().round(5).reshape(-1).tolist()
            for t in range(part.num_frames)]
    D["colors"]["p_model"] = "#90e07a"
    D["colors"]["r_model"] = "#d2a8ff"
    D["labels"]["p_model"] = "P model reconstruction"
    D["labels"]["r_model"] = "R model reconstruction"
    D["gt"] = gt_eval or {"available": False}

    extra_objects = """
const RD=D.reduction, SC=RD.scale;
function axisSeg(origin,dir,len,color,op){
 const a=[origin[0]-dir[0]*len,origin[1]-dir[1]*len,origin[2]-dir[2]*len,
          origin[0]+dir[0]*len,origin[1]+dir[1]*len,origin[2]+dir[2]*len];
 return lines(a,color,op);}
OB.p_dir=axisSeg(RD.centroid,RD.prismatic.direction,SC*0.9,0x90e07a,1.0);
OB.r_axis=axisSeg(RD.revolute.axis_point,RD.revolute.axis,SC*0.9,0xd2a8ff,1.0);
if(D.gt&&D.gt.available){
 OB.gt_axis=axisSeg(D.gt.gt_axis_point,D.gt.gt_axis,SC*0.9,0xf85149,1.0);}
for(const k of ['p_model','r_model']){
 OB[k]=pts(D.layers[k][0],parseInt(D.colors[k].slice(1),16),0.016,0.85);
 OB[k].visible=false;}
"""
    extra_frame = """
 for(const k of ['p_model','r_model']){
  if(!OB[k])continue;
  const a=OB[k].geometry.attributes.position.array,f=D.layers[k][t];
  for(let i=0;i<a.length;i++)a[i]=f[i];
  OB[k].geometry.attributes.position.needsUpdate=true;}
 const q=D.reduction;
 document.getElementById('qnow').innerHTML=
  [['s(t)  [P]',q.prismatic.s[t].toFixed(5)],
   ['theta(t)  [R]',q.revolute.theta[t].toFixed(5)+' rad']]
  .map(([a,b])=>'<div class="row"><span>'+a+'</span><span>'+b+'</span></div>').join('');
"""
    panel = """
<div><span class="tab on" id="tabA">Stage A - rigid</span>
<span class="tab" id="tabB">Stage B - joint</span></div>
<div id="counts" class="note" style="margin-top:8px"></div>
<h2>Layers</h2><div id="tg"></div>
<h2 id="hq">At this frame</h2><div id="qnow"></div>
<h2>Reduction</h2><div id="red"></div>
<h2>Against GT</h2><div id="gt"></div>
<div class="note" style="margin-top:12px">
Section 40: the reduced-model reconstructions are overlays. The Phase-2
centers are always the ones drawn, so a disagreement between the joint and
the data stays visible instead of being hidden by a cleaner reconstruction.
</div>"""
    panel_js = """
const A=[['static_context',D.labels.static_context,D.colors.static_context],
 ['canonical',D.labels.canonical,D.colors.canonical],
 ['phase1',D.labels.phase1,D.colors.phase1],
 ['kabsch',D.labels.kabsch,D.colors.kabsch],
 ['phase2',D.labels.phase2,D.colors.phase2],
 ['com_phase1','COM trail: Phase 1',D.colors.phase1],
 ['com_kabsch','COM trail: Kabsch',D.colors.kabsch],
 ['com_phase2','COM trail: Phase 2',D.colors.phase2],
 ['velocity','rigid velocity','#7ee787']];
const B=A.concat([['p_dir','P direction','#90e07a'],
 ['r_axis','R axis','#d2a8ff'],['gt_axis','GT axis','#f85149'],
 ['p_model',D.labels.p_model,D.colors.p_model],
 ['r_model',D.labels.r_model,D.colors.r_model]]);
function stage(which){
 const on=which==='B'?B:A;
 const keys=new Set(on.map(x=>x[0]));
 ['p_dir','r_axis','gt_axis','p_model','r_model'].forEach(k=>{
  if(OB[k])OB[k].visible=which==='B'&&(k==='p_dir'||k==='r_axis'||k==='gt_axis');});
 toggles('tg',on);
 document.getElementById('tabA').className='tab'+(which==='A'?' on':'');
 document.getElementById('tabB').className='tab'+(which==='B'?' on':'');}
document.getElementById('tabA').onclick=()=>stage('A');
document.getElementById('tabB').onclick=()=>stage('B');
const RD2=D.reduction;
document.getElementById('red').innerHTML='<table>'+
 '<tr><th></th><th>P</th><th>R</th></tr>'+
 '<tr><td>pose residual</td><td>'+RD2.prismatic.energy.toExponential(3)+
 '</td><td>'+RD2.revolute.energy.toExponential(3)+'</td></tr>'+
 '<tr><td>rank-1 fraction</td><td>'+(RD2.prismatic.rank1||NaN).toFixed(4)+
 '</td><td>'+(RD2.revolute.rank1||NaN).toFixed(4)+'</td></tr>'+
 '</table>'+
 '<div class="row"><span>gamma 1-DOF</span><span>'+RD2.one_dof.toFixed(4)+'</span></div>'+
 '<div class="row"><span>axis-point cond</span><span>'+RD2.revolute.cond.toExponential(2)+'</span></div>'+
 '<div class="row"><span>axis radius</span><span>'+(RD2.revolute.radius||NaN).toFixed(4)+'</span></div>';
document.getElementById('gt').innerHTML=(D.gt&&D.gt.available)?
 '<div class="row"><span>GT type</span><span>'+D.gt.gt_joint_type+'</span></div>'+
 '<div class="row"><span>P axis error</span><span>'+D.gt.prismatic.axis_angle_deg.toFixed(2)+' deg</span></div>'+
 '<div class="row"><span>R axis error</span><span>'+D.gt.revolute.axis_angle_deg.toFixed(2)+' deg</span></div>'+
 '<div class="row"><span>R line error</span><span>'+D.gt.revolute.axis_line_error.distance.toFixed(4)+'</span></div>'
 :'<div class="note">'+((D.gt&&D.gt.reason)||'no GT available')+'</div>';
document.getElementById('counts').textContent=
 'showing '+D.num_moving_shown+' of '+D.num_moving_total+' moving and '+
 D.num_static_shown+' static Gaussians';
stage('A');
"""
    return _emit(path, title, D, panel, panel_js, extra_objects, extra_frame)


# ---------------------------------------------------------------------------
# Section 41: the timelines
# ---------------------------------------------------------------------------

def write_timelines(path, part, reduction, records, kabsch_rows,
                    title="timelines") -> Path:
    """Section 41. Everything needed to tell a wrong rotation apart from a
    wrong reduction of a correct one."""
    T = part.num_frames
    rv = _rotvec_series(part.R)
    P, R = reduction["prismatic"], reduction["revolute"]
    pr_P = P["pose_residual"]["per_frame_left"]
    pr_R = R["pose_residual"]["per_frame_left"]
    post = []
    for r in records:
        ev = (r.observability or {}).get("posterior_eigenvalues") or [float("nan")]
        post.append([min(ev), max(ev)])
    D = {
        "frames": T,
        "translation": part.p.numpy().round(6).tolist(),
        "rotvec": rv.numpy().round(6).tolist(),
        "angle": rv.norm(dim=1).numpy().round(6).tolist(),
        "v": part.v.numpy().round(6).tolist(),
        "omega": part.omega.numpy().round(6).tolist(),
        "s": P["s"], "theta": R["theta"],
        "res_P": [float(np.linalg.norm(x)) for x in pr_P],
        "res_R": [float(np.linalg.norm(x)) for x in pr_R],
        "res_P_rot": [float(np.linalg.norm(x[3:])) for x in pr_P],
        "res_R_rot": [float(np.linalg.norm(x[3:])) for x in pr_R],
        "posterior": post,
        "singular_values": reduction["one_dof"]["singular_values"],
        "explained": reduction["one_dof"]["explained_fraction"],
        "gamma": reduction["one_dof"]["gamma_1dof"],
        "d_rot": [r[1] for r in kabsch_rows],
        "d_trans": [r[3] for r in kabsch_rows],
        "energy": [r.energy_after for r in records],
    }
    html = r"""<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title><style>
 body{margin:0;background:#0e1116;color:#e6edf3;padding:16px;
  font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
 h1{font-size:15px;margin:0 0 4px} h2{font-size:12px;color:#58a6ff;
  text-transform:uppercase;margin:18px 0 4px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));
  gap:14px} .cell{background:#161b22;border:1px solid #2a313a;border-radius:6px;
  padding:8px} canvas{width:100%;height:180px;display:block}
 .note{color:#8b949e;font-size:11px} .cap{margin-bottom:4px}
</style></head><body>
<h1>__TITLE__</h1>
<div class="note">Section 41. If the axis is wrong, these say whether it is
because the rigid rotations are wrong or because a correct rotation sequence
was reduced badly: compare "rotation vector" against "R pose residual".</div>
<div class="grid" id="g"></div>
<script>
const D=__PAYLOAD__;
const COL=['#f06595','#4cc9f0','#ffd166','#7ee787','#d2a8ff','#f0883e'];
function plot(title,series,labels,caption){
 const d=document.createElement('div');d.className='cell';
 d.innerHTML='<div class="cap">'+title+'</div><canvas></canvas>'+
  (caption?'<div class="note">'+caption+'</div>':'');
 document.getElementById('g').appendChild(d);
 const cv=d.querySelector('canvas'),r=devicePixelRatio||1;
 cv.width=cv.clientWidth*r;cv.height=180*r;
 const x=cv.getContext('2d');x.scale(r,r);
 const W=cv.clientWidth,H=180,pad=26;
 let lo=Infinity,hi=-Infinity;
 series.forEach(s=>s.forEach(v=>{if(isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}}));
 if(!isFinite(lo)){lo=0;hi=1;} if(hi-lo<1e-12){hi=lo+1e-12;}
 const X=i=>pad+(W-pad-6)*i/Math.max(1,D.frames-1);
 const Y=v=>H-18-(H-28)*(v-lo)/(hi-lo);
 x.strokeStyle='#2a313a';x.beginPath();x.moveTo(pad,H-18);x.lineTo(W-6,H-18);x.stroke();
 if(lo<0&&hi>0){x.strokeStyle='#30363d';x.beginPath();
  x.moveTo(pad,Y(0));x.lineTo(W-6,Y(0));x.stroke();}
 x.fillStyle='#8b949e';x.font='10px monospace';
 x.fillText(hi.toExponential(2),2,12);x.fillText(lo.toExponential(2),2,H-22);
 series.forEach((s,k)=>{x.strokeStyle=COL[k%COL.length];x.lineWidth=1.4;
  x.beginPath();s.forEach((v,i)=>{if(!isFinite(v))return;
   i?x.lineTo(X(i),Y(v)):x.moveTo(X(i),Y(v));});x.stroke();});
 if(labels){x.font='10px monospace';let ox=pad+4;
  labels.forEach((l,k)=>{x.fillStyle=COL[k%COL.length];x.fillText(l,ox,12);
   ox+=x.measureText(l).width+10;});}
}
const col=(a,k)=>a.map(r=>r[k]);
plot('translation p(t)',[col(D.translation,0),col(D.translation,1),col(D.translation,2)],['px','py','pz']);
plot('rotation vector phi(t)',[col(D.rotvec,0),col(D.rotvec,1),col(D.rotvec,2)],['x','y','z']);
plot('rotation angle |phi| (rad)',[D.angle],['|phi|']);
plot('rigid linear velocity v(t)',[col(D.v,0),col(D.v,1),col(D.v,2)],['vx','vy','vz']);
plot('rigid angular velocity omega(t)',[col(D.omega,0),col(D.omega,1),col(D.omega,2)],['wx','wy','wz']);
plot('1-DOF singular values',[D.singular_values],['sigma_k'],
 'gamma_1DOF = '+D.gamma.toFixed(6)+'  (explained: '+
 D.explained.map(v=>v.toFixed(4)).join(', ')+')');
plot('P displacement s(t)',[D.s],['s']);
plot('R angle theta(t)',[D.theta],['theta']);
plot('pose residual |r_t|',[D.res_P,D.res_R],['P','R'],
 'the section 35 comparison, in pose space');
plot('rotational part of the pose residual',[D.res_P_rot,D.res_R_rot],['P','R'],
 'a P model cannot explain rotation at all, so its rotational residual IS the observed rotation');
plot('pose uncertainty (posterior eigenvalues)',[col(D.posterior,0),col(D.posterior,1)],['min','max']);
plot('Phase-1 Kabsch vs RGB-D rigid',[D.d_rot,D.d_trans],['d_rot (rad)','d_p'],
 'section 21: a large divergence is a reason to inspect the RGB-D objective before trusting the reduction');
plot('RGB-D energy after update',[D.energy],['E']);
</script></body></html>"""
    html = html.replace("__TITLE__", title).replace(
        "__PAYLOAD__", json.dumps(D, separators=(",", ":")))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html)
    return p


# ---------------------------------------------------------------------------
# Section 22: the fixed-camera render comparison
# ---------------------------------------------------------------------------

def write_render_html(path, renders: dict, rel: str = "../phase2_rigid/renders",
                      title="fixed-camera renders") -> Path:
    """One camera, every frame, four layers x (RGB | depth | residual).

    Section 22's question is narrow and worth keeping in view: does driving the
    rigid pose through RGB-D actually improve on the rigid projection of the
    free Phase-1 motion, or only move it? The Kabsch and Phase-2 rows are
    adjacent for exactly that comparison.
    """
    D = {"frames": renders.get("frames", []), "camera": renders.get("camera"),
         "rel": rel,
         "rows": [("gt", "ground truth"), ("phase1", "Phase 1 free"),
                  ("kabsch", "Phase 1 Kabsch rigid"),
                  ("phase2", "Phase 2 RGB-D rigid")]}
    html = r"""<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title><style>
 body{margin:0;background:#0e1116;color:#e6edf3;padding:14px;
  font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
 h1{font-size:15px;margin:0 0 8px}
 table{border-collapse:collapse} td,th{padding:3px;text-align:center}
 th{color:#58a6ff;font-weight:400;font-size:11px}
 img{width:260px;display:block;background:#161b22;border:1px solid #2a313a}
 #bar{position:sticky;top:0;background:#0e1116;padding:8px 0;display:flex;
  gap:10px;align-items:center;z-index:2}
 input[type=range]{flex:1} .note{color:#8b949e;font-size:11px}
 button{background:#21262d;color:#e6edf3;border:1px solid #2a313a;
  border-radius:4px;padding:3px 9px;cursor:pointer;font:inherit}
</style></head><body>
<h1>__TITLE__</h1>
<div id="bar"><button id="play">play</button>
 <input type="range" id="sl" min="0" max="0" value="0"><span id="fn"></span></div>
<div class="note">Section 22: compare the Kabsch row with the Phase-2 row.
That difference is what driving the rigid pose through RGB-D actually bought.</div>
<table id="t"></table>
<script>
const D=__PAYLOAD__;
const T=D.frames.length;
const sl=document.getElementById('sl');sl.max=Math.max(T-1,0);
function pad(n){return String(n).padStart(4,'0');}
function draw(i){
 const f=D.frames[i];document.getElementById('fn').textContent=
  'frame '+f+'  (camera '+D.camera+')';
 let h='<tr><th></th><th>RGB</th><th>depth</th><th>|residual|</th></tr>';
 D.rows.forEach(([k,label])=>{
  h+='<tr><th style="text-align:left">'+label+'</th>';
  h+='<td><img src="'+D.rel+'/'+k+'_rgb_'+pad(f)+'.png"></td>';
  h+='<td><img src="'+D.rel+'/'+k+'_depth_'+pad(f)+'.png"></td>';
  h+=(k==='gt')?'<td class="note">-</td>'
     :'<td><img src="'+D.rel+'/'+k+'_residual_'+pad(f)+'.png"></td>';
  h+='</tr>';});
 document.getElementById('t').innerHTML=h;}
sl.addEventListener('input',e=>draw(+e.target.value));
let timer=null;
document.getElementById('play').addEventListener('click',e=>{
 if(timer){clearInterval(timer);timer=null;e.target.textContent='play';return;}
 e.target.textContent='stop';
 timer=setInterval(()=>{sl.value=(+sl.value+1)%T;draw(+sl.value);},300);});
if(T)draw(0);
</script></body></html>"""
    html = html.replace("__TITLE__", title).replace(
        "__PAYLOAD__", json.dumps(D, separators=(",", ":")))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html)
    return p
