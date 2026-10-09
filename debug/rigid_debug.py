"""Sections 23, 40, 41: the rigid Phase-2 and joint-reduction debug pages.

CODEX_IMPLEMENT_RIGID_SE3_PHASE2_AND_POSTHOC_JOINT_REDUCTION.md.

Three pages:

  rigid_phase2_debug.html   section 23. The rigid reconstruction ALONE, with
                            NO joint axis anywhere -- the point is to judge
                            whether the rigid body was recovered, separately
                            from whether a joint explains it.
  rigid_to_joint_debug.html section 40. Stage A is the rigid reconstruction;
                            Stage B shows the Phase-2 centers plus the axes as
                            overlays, and drops the Stage-A comparison layers
                            so the axis is judged against the data and not
                            against four other point clouds.
  rigid_timelines.html      section 41.

TWO THINGS THE FIRST VERSION GOT WRONG, both visible on scissor_11100.

1. It animated only the frozen support. Every Gaussian OUTSIDE the support
   was drawn once at frame 0 and never moved -- on that scene 6208 of 8210,
   76% of the scene. Phase 1 solves a free trajectory for ALL of them, and
   whether any of those move is exactly the evidence needed to judge whether
   the support missed something. The Phase-1 layer now covers every drawn
   Gaussian and is split into two colours, inside and outside the support,
   with the worst excursion of each reported numerically.

2. It shipped five 60-frame point clouds, 14.8 MB of a 14.9 MB page. Four of
   those are a RIGID transform of the canonical centers, so the page can be
   given 60 poses (12 numbers each) and apply them in the browser instead.
   That is what paid for point 1: the page now animates three times as many
   Gaussians in about half the bytes.
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
    "phase1_outside": "#f0883e",
    "kabsch": "#4cc9f0",
    "phase2": "#f06595",
    "p_model": "#90e07a",
    "r_model": "#d2a8ff",
}
LAYER_LABELS = {
    "static_context": "outside the support (frame 0)",
    "canonical": "support, canonical (frame 0)",
    "phase1": "Phase 1 free motion - support",
    "phase1_outside": "Phase 1 free motion - OUTSIDE the support",
    "kabsch": "Phase 1 Kabsch rigid",
    "phase2": "Phase 2 RGB-D rigid",
    "p_model": "P model reconstruction",
    "r_model": "R model reconstruction",
}


def _subset(members: torch.Tensor, n_total: int, max_points: int, seed=0):
    """One deterministic subset, reused by every layer.

    Resampling per layer would mean the layers are not the same Gaussians and
    nothing could be compared point to point. The support is kept whole up to
    half the budget; the rest of the scene fills the remainder, because those
    are the points that answer "did the support miss anything".
    """
    m = torch.as_tensor(members).long().reshape(-1).cpu()
    g = torch.Generator().manual_seed(int(seed))
    budget = int(max_points)
    mov = m if m.numel() <= budget // 2 else m[
        torch.randperm(m.numel(), generator=g)[: budget // 2]].sort().values
    mask = torch.ones(n_total, dtype=torch.bool)
    mask[m] = False
    out = torch.nonzero(mask, as_tuple=False).reshape(-1)
    room = budget - mov.numel()
    if out.numel() > room:
        out = out[torch.randperm(out.numel(), generator=g)[:room]].sort().values
    return mov, out


def _poses(R, p):
    """[T, 12] -- row-major rotation then translation, for the browser."""
    R = torch.as_tensor(R).double().cpu().reshape(-1, 3, 3)
    p = torch.as_tensor(p).double().cpu().reshape(-1, 3)
    return torch.cat([R.reshape(-1, 9), p], dim=1).numpy().round(8).tolist()


def _com(flat_frames, n):
    out = []
    for f in flat_frames:
        a = np.asarray(f, dtype=np.float64).reshape(-1, 3)
        out.append(a.mean(0).round(5).tolist() if len(a) else [0.0, 0.0, 0.0])
    return out


def _com_from_pose(X0, poses):
    """COM trail of a rigidly transformed set, without expanding the frames."""
    c = np.asarray(X0, dtype=np.float64).mean(0)
    out = []
    for q in poses:
        R = np.asarray(q[:9], dtype=np.float64).reshape(3, 3)
        out.append((R @ c + np.asarray(q[9:], dtype=np.float64)).round(5).tolist())
    return out


def _frustums(cameras, limit=8):
    from estimator.debug_3d import _frustum
    out = []
    for cam in (cameras or [])[:limit]:
        f = _frustum(cam)
        if f is not None:
            out.append(f)
    return out


def _rotvec_series(R):
    from estimator.manifold import so3_log
    return so3_log(torch.as_tensor(R).double().reshape(-1, 3, 3))


def _base_payload(mu0, members, positions, part, kabsch_R, kabsch_p,
                  cameras, max_points, seed=0, extra_poses=None,
                  unexplained=None):
    mu0 = torch.as_tensor(mu0).double().cpu()
    n = mu0.shape[0]
    T = part.num_frames
    mov, out = _subset(members, n, max_points, seed)
    shown = torch.cat([mov, out])                 # support first, then the rest
    ns = int(mov.numel())

    # Phase 1 is the only layer that is NOT a rigid transform of the canonical
    # centers, so it is the only one that has to be shipped per frame -- and
    # it is shipped for EVERY drawn Gaussian, not just the support.
    p1 = [torch.as_tensor(positions[t]).double().cpu()[shown]
          .numpy().round(4).reshape(-1).tolist() for t in range(T)]

    # Per-Gaussian travel, as a DISTRIBUTION. The first version compared the
    # two maxima, which on scissor said geometry outside the support moves 70%
    # as far as the support -- driven by 13 points out of 6208. The medians
    # differ by 9x. A max over thousands of free trajectories is an outlier
    # detector, not a summary, so the warning below is a count instead.
    X0 = mu0[shown].numpy()
    trav = np.zeros(len(X0))
    for t in range(T):
        d = np.linalg.norm(np.asarray(p1[t]).reshape(-1, 3) - X0, axis=1)
        trav = np.maximum(trav, d)

    def _stats(d):
        if not len(d):
            return {"n": 0, "median": 0.0, "p90": 0.0, "max": 0.0}
        return {"n": int(len(d)), "median": float(np.median(d)),
                "p90": float(np.percentile(d, 90)), "max": float(d.max())}

    inside, outside = trav[:ns], trav[ns:]
    med = float(np.median(inside)) if ns else 0.0
    n_over = int((outside > med).sum()) if len(outside) else 0

    poses = {"kabsch": _poses(kabsch_R, kabsch_p),
             "phase2": _poses(part.R, part.p)}
    for k, v in (extra_poses or {}).items():
        poses[k] = _poses(*v)

    Xs = mu0[mov].numpy()
    com = {"phase1": _com(p1, ns)}
    for k in poses:
        com[k] = _com_from_pose(Xs, poses[k])
    # the Phase-1 COM above is over everything drawn; the support-only trail is
    # the one comparable with the rigid layers
    com["phase1"] = _com([f[: ns * 3] for f in p1], ns)

    return {
        "canonical": mu0[shown].numpy().round(5).reshape(-1).tolist(),
        "n_support_shown": ns,
        "n_outside_shown": int(out.numel()),
        "phase1": p1,
        "poses": poses,
        "com": com,
        "velocity": {"v": part.v.numpy().round(5).tolist(),
                     "omega": part.omega.numpy().round(5).tolist()},
        "rotvec": _rotvec_series(part.R).numpy().round(6).tolist(),
        "translation": part.p.numpy().round(6).tolist(),
        "frames": T,
        "num_moving_total": int(torch.as_tensor(members).numel()),
        "num_gaussians": int(n),
        "excursion": {"support": _stats(inside), "outside": _stats(outside),
                      "support_median": med, "outside_over_support_median": n_over,
                      "outside_over_fraction": (n_over / max(len(outside), 1))},
        "nonrigidity": {"unexplained": (float(unexplained)
                                        if unexplained is not None
                                        else float("nan"))},
        "frustums": _frustums(cameras),
        "colors": LAYER_COLORS, "labels": LAYER_LABELS,
    }


# ---------------------------------------------------------------------------
# the shared viewer shell
# ---------------------------------------------------------------------------

_SHELL = r"""<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title><style>
 body{margin:0;background:#0e1116;color:#e6edf3;overflow:hidden;
  font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
 #app{display:flex;height:100vh} #view{flex:1;position:relative}
 #side{width:370px;background:#161b22;border-left:1px solid #2a313a;
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
 .warn{color:#f0883e}
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
const NS=D.n_support_shown, NO=D.n_outside_shown, N=NS+NO;
const S=new THREE.Scene(),cv=document.getElementById('c');
const RE=new THREE.WebGLRenderer({canvas:cv,antialias:true});
S.background=new THREE.Color(0x0e1116);
const CAM=new THREE.PerspectiveCamera(55,1,0.001,8000);
function fit(){const v=document.getElementById('view');
 RE.setPixelRatio(devicePixelRatio);RE.setSize(v.clientWidth,v.clientHeight);
 CAM.aspect=v.clientWidth/Math.max(v.clientHeight,1);CAM.updateProjectionMatrix();}
let ctr=[0,0,0];
for(let i=0;i<N;i++){ctr[0]+=D.canonical[i*3];ctr[1]+=D.canonical[i*3+1];ctr[2]+=D.canonical[i*3+2];}
ctr=ctr.map(v=>v/Math.max(N,1)); let rad=1e-3;
for(let i=0;i<N;i++)rad=Math.max(rad,Math.hypot(D.canonical[i*3]-ctr[0],
 D.canonical[i*3+1]-ctr[1],D.canonical[i*3+2]-ctr[2]));
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

function pts(count,color,size,op){
 const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(count*3),3));
 const o=new THREE.Points(g,new THREE.PointsMaterial({color:color,
  size:rad*size,sizeAttenuation:true,transparent:true,opacity:op===undefined?1:op}));
 S.add(o);return o;}
function lines(arr,color,op){
 const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(arr),3));
 const o=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:color,
  transparent:true,opacity:op===undefined?1:op}));S.add(o);return o;}
function polyline(points,color,op){
 const a=[];for(let i=0;i+1<points.length;i++){
  a.push(points[i][0],points[i][1],points[i][2],
         points[i+1][0],points[i+1][1],points[i+1][2]);}
 return lines(a,color,op);}
const hx=s=>parseInt(s.slice(1),16);

const OB={};
// frame-0 reference clouds, never animated
OB.canonical=pts(NS,hx(D.colors.canonical),0.013,0.55);
OB.static_context=pts(NO,hx(D.colors.static_context),0.010,0.40);
{const a=OB.canonical.geometry.attributes.position.array;
 for(let i=0;i<NS*3;i++)a[i]=D.canonical[i];
 OB.canonical.geometry.attributes.position.needsUpdate=true;
 const b=OB.static_context.geometry.attributes.position.array;
 for(let i=0;i<NO*3;i++)b[i]=D.canonical[NS*3+i];
 OB.static_context.geometry.attributes.position.needsUpdate=true;}

// Phase 1 free motion, split at the support boundary. The second of these is
// the one that says whether the support missed moving geometry.
OB.phase1=pts(NS,hx(D.colors.phase1),0.018,1.0);
OB.phase1_outside=pts(NO,hx(D.colors.phase1_outside),0.016,1.0);

// every rigid layer is 60 poses applied in the browser, not 60 point clouds
const POSE_LAYERS=Object.keys(D.poses);
POSE_LAYERS.forEach(k=>{OB[k]=pts(NS,hx(D.colors[k]||'#ffffff'),0.018,1.0);});
function applyPose(obj,q){
 const a=obj.geometry.attributes.position.array;
 const r=q,px=q[9],py=q[10],pz=q[11];
 for(let i=0;i<NS;i++){const x=D.canonical[i*3],y=D.canonical[i*3+1],z=D.canonical[i*3+2];
  a[i*3]  =r[0]*x+r[1]*y+r[2]*z+px;
  a[i*3+1]=r[3]*x+r[4]*y+r[5]*z+py;
  a[i*3+2]=r[6]*x+r[7]*y+r[8]*z+pz;}
 obj.geometry.attributes.position.needsUpdate=true;}

for(const k in D.com)
 OB['com_'+k]=polyline(D.com[k],hx(D.colors[k]||'#ffffff'),0.95);
OB.velocity=lines([0,0,0,0,0,0],0x7ee787,1.0);
if(D.frustums&&D.frustums.length){
 const arr=[];D.frustums.forEach(f=>{const c=f.center;
  f.corners.forEach(k=>{arr.push(c[0],c[1],c[2],k[0],k[1],k[2]);});
  for(let i=0;i<4;i++){const a=f.corners[i],b=f.corners[(i+1)%4];
   arr.push(a[0],a[1],a[2],b[0],b[1],b[2]);}});
 OB.cameras=lines(arr,0x30363d,0.8);}
__EXTRA_OBJECTS__

function setFrame(t){
 const f=D.phase1[t];
 const a=OB.phase1.geometry.attributes.position.array;
 for(let i=0;i<NS*3;i++)a[i]=f[i];
 OB.phase1.geometry.attributes.position.needsUpdate=true;
 const b=OB.phase1_outside.geometry.attributes.position.array;
 for(let i=0;i<NO*3;i++)b[i]=f[NS*3+i];
 OB.phase1_outside.geometry.attributes.position.needsUpdate=true;
 POSE_LAYERS.forEach(k=>applyPose(OB[k],D.poses[k][t]));
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
  h.insertAdjacentHTML('beforeend','<label><input type="checkbox" '+
   (OB[k].visible?'checked':'')+' data-k="'+k+'"><span class="sw" style="background:'+
   color+'"></span>'+label+'</label>');});
 h.querySelectorAll('input').forEach(e=>e.addEventListener('change',ev=>{
  OB[ev.target.dataset.k].visible=ev.target.checked;}));}
function supportBox(){
 const e=D.excursion, f=e.outside_over_fraction;
 const row=(a,b,c)=>'<div class="row"><span>'+a+'</span><span'+
  (c?' class="warn"':'')+'>'+b+'</span></div>';
 let h=row('drawn: support',NS+' of '+D.num_moving_total)+
  row('drawn: outside',NO+' of '+(D.num_gaussians-D.num_moving_total))+
  '<table style="margin-top:4px"><tr><th>Phase-1 travel</th><th>median</th>'+
  '<th>p90</th><th>max</th></tr>'+
  '<tr><td>support</td><td>'+e.support.median.toFixed(4)+'</td><td>'+
  e.support.p90.toFixed(4)+'</td><td>'+e.support.max.toFixed(4)+'</td></tr>'+
  '<tr><td>outside</td><td>'+e.outside.median.toFixed(4)+'</td><td>'+
  e.outside.p90.toFixed(4)+'</td><td>'+e.outside.max.toFixed(4)+'</td></tr></table>'+
  row('outside travelling > support median',
      e.outside_over_support_median+' ('+(100*f).toFixed(1)+'%)',f>0.05);
 h+='<div class="note" style="margin-top:4px">'+
  (f>0.05
   ? 'A real share of the geometry outside the support moves as much as the '+
     'support does. Turn on the orange layer: that is motion the selector did '+
     'not claim.'
   : 'Compare the MEDIANS, not the maxima -- a max over thousands of free '+
     'trajectories finds the worst-tracked point, not the typical one. Here '+
     'the bulk outside the support barely moves. The orange layer is still '+
     'the check to look at: a few bright points moving with the part are the '+
     'ones the selector missed.')+'</div>';
 if(D.nonrigidity&&isFinite(D.nonrigidity.unexplained)){
  const r=D.nonrigidity.unexplained;
  h+='<h2>Explained by one rigid motion?</h2>'+
   row('unexplained share of the displacement',(100*r).toFixed(0)+'%',r>0.4)+
   '<div class="note">Kabsch residual over the mean per-point displacement, '+
   'so it does not depend on how big the part is. NOT a body count -- the '+
   'scissor, which really is two blades about one pivot, sits mid-pack here. '+
   'gamma 1-DOF is what finds that.</div>';}
 return h;}
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
                          title="rigid SE(3) Phase 2", seed=0,
                          unexplained=None) -> Path:
    """Section 23. Deliberately shows no axis: the question this page answers
    is whether the RIGID BODY was recovered, which has to be judged before and
    separately from whether a joint explains it."""
    D = _base_payload(mu0, members, positions, part, kabsch_R, kabsch_p,
                      cameras, max_points, seed,
                      unexplained=unexplained)
    panel = """
<h2>Support coverage</h2><div id="cov"></div>
<h2>Layers</h2><div id="tg"></div>
<h2>Pose at this frame</h2><div id="pose"></div>
<div class="note" style="margin-top:14px">
Section 23: no joint axis is drawn here, on purpose. Judge the rigid
reconstruction first; the axis lives in rigid_to_joint_debug.html.
</div>"""
    panel_js = """
document.getElementById('cov').innerHTML=supportBox();
toggles('tg',[['static_context',D.labels.static_context,D.colors.static_context],
 ['canonical',D.labels.canonical,D.colors.canonical],
 ['phase1',D.labels.phase1,D.colors.phase1],
 ['phase1_outside',D.labels.phase1_outside,D.colors.phase1_outside],
 ['kabsch',D.labels.kabsch,D.colors.kabsch],
 ['phase2',D.labels.phase2,D.colors.phase2],
 ['com_phase1','COM trail: Phase 1',D.colors.phase1],
 ['com_kabsch','COM trail: Kabsch',D.colors.kabsch],
 ['com_phase2','COM trail: Phase 2',D.colors.phase2],
 ['velocity','rigid velocity','#7ee787'],
 ['cameras','camera frustums','#30363d']]);
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
# Section 40: Stage A rigid, Stage B joint reduction
# ---------------------------------------------------------------------------

def write_rigid_to_joint_3d(path, mu0, members, positions, part, kabsch_R,
                            kabsch_p, reduction, gt_eval=None,
                            max_points=6000,
                            title="rigid reconstruction and joint reduction",
                            seed=0, unexplained=None) -> Path:
    """Section 40.

    Stage B shows the Phase-2 centers and the axes, and NOT the Stage-A
    comparison layers. Those answer a different question -- was the rigid body
    recovered -- and leaving them on means the axis is judged against four
    other point clouds instead of against the data it was fitted to. They stay
    one click away in Stage A.
    """
    P, R = reduction["prismatic"], reduction["revolute"]
    D = _base_payload(mu0, members, positions, part, kabsch_R, kabsch_p,
                      None, max_points, seed,
                      extra_poses={"p_model": (P["poses_R"], P["poses_p"]),
                                   "r_model": (R["poses_R"], R["poses_p"])},
                      unexplained=unexplained)
    mu0 = torch.as_tensor(mu0).double().cpu()
    c0 = mu0[torch.as_tensor(members).long().reshape(-1)].mean(0)
    scale = float((mu0 - mu0.mean(0)).norm(dim=1).max())
    D["reduction"] = {
        "scale": scale, "centroid": c0.tolist(),
        "prismatic": {"direction": P["direction"], "s": P["s"],
                      "energy": P["pose_residual"]["weighted_energy"],
                      "rank1": P.get("translation_rank1_fraction")},
        "revolute": {"axis": R["axis"], "axis_point": R["axis_point"],
                     "theta": R["theta"],
                     "energy": R["pose_residual"]["weighted_energy"],
                     "rank1": R.get("rank1_fraction"),
                     "cond": R["finite_axis_solver"]["condition_number"],
                     "radius": R.get("radius_from_centroid"),
                     "rad_over_L": R.get("axis_radius_over_part_scale")},
        "one_dof": reduction["one_dof"]["gamma_1dof"],
    }
    D["gt"] = gt_eval or {"available": False}

    extra_objects = """
const RD=D.reduction, SC=RD.scale;
function axisSeg(origin,dir,len,color){
 return lines([origin[0]-dir[0]*len,origin[1]-dir[1]*len,origin[2]-dir[2]*len,
               origin[0]+dir[0]*len,origin[1]+dir[1]*len,origin[2]+dir[2]*len],
              color,1.0);}
OB.p_dir=axisSeg(RD.centroid,RD.prismatic.direction,SC*0.9,0x90e07a);
OB.r_axis=axisSeg(RD.revolute.axis_point,RD.revolute.axis,SC*0.9,0xd2a8ff);
if(D.gt&&D.gt.available)
 OB.gt_axis=axisSeg(D.gt.gt_axis_point,D.gt.gt_axis,SC*0.9,0xf85149);
"""
    extra_frame = """
 const q=D.reduction;
 document.getElementById('qnow').innerHTML=
  [['s(t)  [P]',q.prismatic.s[t].toFixed(5)],
   ['theta(t)  [R]',q.revolute.theta[t].toFixed(5)+' rad']]
  .map(([a,b])=>'<div class="row"><span>'+a+'</span><span>'+b+'</span></div>').join('');
"""
    panel = """
<div><span class="tab on" id="tabA">Stage A - rigid</span>
<span class="tab" id="tabB">Stage B - joint</span></div>
<div id="stagenote" class="note" style="margin-top:8px"></div>
<h2>Support coverage</h2><div id="cov"></div>
<h2>Layers</h2><div id="tg"></div>
<h2 id="hq">At this frame</h2><div id="qnow"></div>
<h2>Reduction</h2><div id="red"></div>
<h2>Against GT</h2><div id="gt"></div>"""
    panel_js = """
const A=[['static_context',D.labels.static_context,D.colors.static_context],
 ['canonical',D.labels.canonical,D.colors.canonical],
 ['phase1',D.labels.phase1,D.colors.phase1],
 ['phase1_outside',D.labels.phase1_outside,D.colors.phase1_outside],
 ['kabsch',D.labels.kabsch,D.colors.kabsch],
 ['phase2',D.labels.phase2,D.colors.phase2],
 ['com_phase1','COM trail: Phase 1',D.colors.phase1],
 ['com_kabsch','COM trail: Kabsch',D.colors.kabsch],
 ['com_phase2','COM trail: Phase 2',D.colors.phase2],
 ['velocity','rigid velocity','#7ee787']];
// Section 40: Stage B keeps the Phase-2 centers and adds the axes. The
// Stage-A comparison layers are NOT carried over.
const B=[['static_context',D.labels.static_context,D.colors.static_context],
 ['phase2',D.labels.phase2,D.colors.phase2],
 ['com_phase2','COM trail: Phase 2',D.colors.phase2],
 ['p_dir','P direction','#90e07a'],
 ['r_axis','R axis','#d2a8ff'],['gt_axis','GT axis','#f85149'],
 ['p_model',D.labels.p_model,D.colors.p_model],
 ['r_model',D.labels.r_model,D.colors.r_model]];
const NOTE={A:'Was the rigid body recovered? Compare Phase 1 against Kabsch '+
  'against Phase 2. No axis is drawn here.',
 B:'Does a 1-DOF joint explain the recovered motion? The Phase-2 centers are '+
  'still the ones drawn -- the reduced models are overlays, so a disagreement '+
  'stays visible instead of being hidden by a cleaner reconstruction.'};
function stage(which){
 const on=which==='B'?B:A;
 const keys=new Set(on.map(x=>x[0]));
 ['static_context','canonical','phase1','phase1_outside','kabsch','phase2',
  'com_phase1','com_kabsch','com_phase2','velocity','p_dir','r_axis','gt_axis',
  'p_model','r_model'].forEach(k=>{if(OB[k])OB[k].visible=keys.has(k);});
 // the reduced reconstructions start off even in Stage B: they are overlays
 ['p_model','r_model'].forEach(k=>{if(OB[k])OB[k].visible=false;});
 toggles('tg',on);
 document.getElementById('stagenote').textContent=NOTE[which];
 document.getElementById('tabA').className='tab'+(which==='A'?' on':'');
 document.getElementById('tabB').className='tab'+(which==='B'?' on':'');}
document.getElementById('tabA').onclick=()=>stage('A');
document.getElementById('tabB').onclick=()=>stage('B');
const RD2=D.reduction;
document.getElementById('red').innerHTML='<table>'+
 '<tr><th></th><th>P</th><th>R</th></tr>'+
 '<tr><td>pose residual</td><td>'+RD2.prismatic.energy.toExponential(3)+
 '</td><td>'+RD2.revolute.energy.toExponential(3)+'</td></tr>'+
 '<tr><td>rank-1 fraction</td><td>'+(+RD2.prismatic.rank1).toFixed(4)+
 '</td><td>'+(+RD2.revolute.rank1).toFixed(4)+'</td></tr></table>'+
 '<div class="row"><span>gamma 1-DOF</span><span>'+RD2.one_dof.toFixed(4)+'</span></div>'+
 '<div class="row"><span>axis-point cond</span><span>'+RD2.revolute.cond.toExponential(2)+'</span></div>'+
 '<div class="row"><span>axis radius / part scale</span><span class="'+
 (RD2.revolute.rad_over_L>10?'warn':'')+'">'+
 (+RD2.revolute.rad_over_L).toFixed(1)+'</span></div>'+
 '<div class="note">radius/L of order 1 means the axis lies on the part. A '+
 'large value means R is approximating a translation with a distant axis.</div>';
document.getElementById('gt').innerHTML=(D.gt&&D.gt.available)?
 '<div class="row"><span>GT type</span><span>'+D.gt.gt_joint_type+'</span></div>'+
 '<div class="row"><span>P axis error</span><span>'+D.gt.prismatic.axis_angle_deg.toFixed(2)+' deg</span></div>'+
 '<div class="row"><span>R axis error</span><span>'+D.gt.revolute.axis_angle_deg.toFixed(2)+' deg</span></div>'+
 '<div class="row"><span>R line error</span><span>'+D.gt.revolute.axis_line_error.distance.toFixed(4)+'</span></div>'
 :'<div class="note">'+((D.gt&&D.gt.reason)||'no GT available')+'</div>';
document.getElementById('cov').innerHTML=supportBox();
stage('A');
"""
    return _emit(path, title, D, panel, panel_js, extra_objects, extra_frame)
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
