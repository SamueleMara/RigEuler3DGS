"""Debug pages for the motion-coherence filter.

CODEX_IMPLEMENT_MOTION_COHERENCE_STATIC_DYNAMIC_FILTER.md sections 25-34.

Three pages:
  finite_strain_heatmap.html        z_ij(t) per edge, edges sorted by C
  differential_strain_heatmap.html  epsilon_ij(t), same ordering
  representative_edges.html         the four regimes of section 9, over time
  motion_coherence_support_3d.html  the partition in 3D, with the graph edges

Edges are ordered by `C` in both heatmaps so the rigid block and the incoherent
block separate visually; the colour scale is shared between them so a reader
can compare an edge's finite strain against its own rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

_CSS = """
 body{margin:0;background:#0e1116;color:#e6edf3;
  font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;padding:18px}
 h1{font-size:15px;margin:0 0 4px} h2{font-size:12px;color:#58a6ff;
  text-transform:uppercase;letter-spacing:.06em;margin:16px 0 6px}
 .note{color:#8b949e;font-size:11px;max-width:900px;margin:6px 0}
 canvas{border:1px solid #2a313a;image-rendering:pixelated;display:block}
 table{border-collapse:collapse;font-size:11px;margin:8px 0}
 td,th{border:1px solid #2a313a;padding:3px 7px;text-align:right}
 th:first-child,td:first-child{text-align:left}
 .bar{height:12px;display:inline-block;vertical-align:middle}
"""


def _heatmap_html(title, z, order, C, labels_i, labels_j, vmax, caption):
    """One canvas, rows = edges (sorted by C), columns = frames."""
    Z = torch.as_tensor(z).double()
    o = torch.as_tensor(order).long()
    Zs = Z[:, o].T                                   # [E, T]
    E, T = Zs.shape
    step = max(1, E // 2000)                          # cap the payload
    Zs = Zs[::step]
    data = np.clip(Zs.numpy() / max(float(vmax), 1e-30), -1, 1)
    rows = data.round(4).tolist()
    Csorted = torch.as_tensor(C)[o][::step].tolist()
    same = [(int(labels_i[e]) == int(labels_j[e])) for e in o[::step].tolist()]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title><style>{_CSS}</style></head><body>
<h1>{title}</h1>
<p class="note">{caption}</p>
<p class="note">Rows are edges sorted by coherence C (most coherent at the
top), columns are frames. Blue is negative strain, red positive, black zero;
the scale is symmetric and shared with the companion page. The left margin
marks edges whose two endpoints landed in DIFFERENT predicted components.</p>
<canvas id="c"></canvas>
<script>
const R={json.dumps(rows)}, C={json.dumps([round(float(c),6) for c in Csorted])},
      SAME={json.dumps(same)};
const E=R.length, T=R[0]?R[0].length:0, PW=Math.max(2,Math.floor(900/Math.max(T,1))),
      PH=Math.max(1,Math.floor(620/Math.max(E,1))), M=26;
const cv=document.getElementById('c');
cv.width=M+T*PW; cv.height=Math.max(E*PH,1);
const g=cv.getContext('2d');
for(let e=0;e<E;e++){{
  g.fillStyle = SAME[e] ? '#2a313a' : '#f0883e';
  g.fillRect(0,e*PH,M-6,Math.max(PH,1));
  for(let t=0;t<T;t++){{
    const v=R[e][t];
    const r=Math.round(255*Math.max(v,0)), b=Math.round(255*Math.max(-v,0));
    g.fillStyle='rgb('+r+',0,'+b+')';
    g.fillRect(M+t*PW,e*PH,PW,Math.max(PH,1));
  }}
}}
</script>
<h2>scale</h2>
<p class="note">|value| = 1 at {vmax:.4e}; {E} edge rows shown
(every {step} of the full set), {T} frames.</p>
</body></html>"""


def write_finite_strain_heatmap(path, res, vmax=None) -> Path:
    z = res["z"]
    order = torch.argsort(res["C"])
    v = vmax or float(z.abs().quantile(0.99))
    html = _heatmap_html(
        "Finite log-strain z_ij(t)", z, order, res["C"],
        res["labels"][res["edge_i"]], res["labels"][res["edge_j"]], v,
        "z_ij(t) = log((l_ij(t)+eps)/(median_t l_ij + eps)). Exactly zero for "
        "every frame on a rigid edge, whatever the motion.")
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html); return p


def write_differential_strain_heatmap(path, res, vmax=None) -> Path:
    e = res["epsilon"]
    order = torch.argsort(res["C"])
    v = vmax or float(e.abs().quantile(0.99))
    html = _heatmap_html(
        "Differential symmetric strain rate eps_ij(t)", e, order, res["C"],
        res["labels"][res["edge_i"]], res["labels"][res["edge_j"]], v,
        "eps_ij(t) = (z(t+1)-z(t))/dt = r^T(v_j-v_i)/|r|^2 = r^T D r/|r|^2. "
        "The rotational part Omega drops out identically, so this is zero for "
        "rigid motion and non-zero only where the field actually strains.")
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html); return p


def write_representative_edges(path, res, n_each: int = 6) -> Path:
    """Section 29: the four regimes of section 9, as time series."""
    C, J = res["C"], res["J"]
    z = res["z"]
    qc, qj = float(C.quantile(0.5)), float(J.quantile(0.5))
    groups = {
        "rigid-compatible (low C, low J)": (C <= qc) & (J <= qj),
        "smooth non-rigid drift (high C, low J)": (C > qc) & (J <= qj),
        "chaotic (high C, high J)": (C > qc) & (J > qj),
        "oscillatory (low C, high J)": (C <= qc) & (J > qj),
    }
    series, meta = [], []
    for name, mask in groups.items():
        idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if not idx.numel():
            continue
        pick = idx[torch.argsort(C[idx])][:n_each] if "low C" in name \
            else idx[torch.argsort(-C[idx])][:n_each]
        for e in pick.tolist():
            series.append(z[:, e].tolist())
            meta.append({"group": name, "edge": e,
                         "node_i": int(res["edge_i"][e]),
                         "node_j": int(res["edge_j"][e]),
                         "C": float(C[e]), "J": float(J[e]),
                         "same_component": bool(
                             res["labels"][res["edge_i"][e]]
                             == res["labels"][res["edge_j"][e]])})
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Representative edge strain</title><style>{_CSS}
 .plot{{border:1px solid #2a313a;margin:8px 0;padding:6px}}
</style></head><body>
<h1>Representative edges &mdash; z_ij(t) over the window</h1>
<p class="note">Section 9's four regimes. A rigid-compatible edge is a flat
line at zero; smooth drift is a ramp; chaotic motion is noise at a comparable
amplitude. The distinction between the last two is exactly what a
displacement-magnitude rule cannot see.</p>
<div id="plots"></div>
<script>
const S={json.dumps([[round(v,6) for v in s] for s in series])},
      M={json.dumps(meta)};
const by={{}}; M.forEach((m,i)=>{{(by[m.group]=by[m.group]||[]).push(i);}});
const host=document.getElementById('plots');
let lo=0,hi=0; S.forEach(s=>s.forEach(v=>{{lo=Math.min(lo,v);hi=Math.max(hi,v);}}));
const pad=Math.max(1e-9,(hi-lo)*0.1); lo-=pad; hi+=pad;
for(const [grp,ids] of Object.entries(by)){{
 const d=document.createElement('div'); d.className='plot';
 d.innerHTML='<h2>'+grp+'</h2>';
 const cv=document.createElement('canvas'); cv.width=880; cv.height=170;
 d.appendChild(cv); host.appendChild(d);
 const g=cv.getContext('2d');
 g.strokeStyle='#2a313a'; g.beginPath();
 const y0=170-(0-lo)/(hi-lo)*170; g.moveTo(0,y0); g.lineTo(880,y0); g.stroke();
 ids.forEach((i,k)=>{{
  const s=S[i], n=s.length;
  g.strokeStyle=['#58a6ff','#3fb950','#f0883e','#f85149','#d2a8ff','#ffd166'][k%6];
  g.beginPath();
  s.forEach((v,t)=>{{const x=t/(n-1)*876+2, y=170-(v-lo)/(hi-lo)*170;
    t?g.lineTo(x,y):g.moveTo(x,y);}});
  g.stroke();
 }});
 const tb=document.createElement('table');
 tb.innerHTML='<tr><th>edge</th><th>i</th><th>j</th><th>C</th><th>J</th>'
  +'<th>same component</th></tr>'+ids.map(i=>{{const m=M[i];
   return '<tr><td>'+m.edge+'</td><td>'+m.node_i+'</td><td>'+m.node_j+
    '</td><td>'+m.C.toExponential(3)+'</td><td>'+m.J.toExponential(3)+
    '</td><td>'+m.same_component+'</td></tr>';}}).join('');
 d.appendChild(tb);
}}
</script></body></html>"""
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html); return p


def write_support_3d(path, res, node_xyz, traj=None, gt_node=None,
                     title="Motion coherence support", max_edges=6000,
                     frame_ids=None) -> Path:
    """Sections 30-33: the partition in 3D, with graph edges and a slider.

    traj: [T, num_nodes, 3] node (supernode-centroid) positions in ORIGINAL
    chronological order, row i = node i of `res`; frame_ids labels the slider
    with the original frame numbers (default 0..T-1)."""
    X = torch.as_tensor(node_xyz).double().cpu()
    lab = res["labels"]
    # Section 22: when the v2 extractor ran, the page shows its THREE classes
    # -- a node that is neither the moving body nor a coherent static one is
    # drawn as ambiguous rather than being forced to a side.
    klass = res.get("node_class")
    ident = res.get("identification", {})
    dyn_comp = ident.get("dynamic_component")
    ei, ej = res["edge_i"], res["edge_j"]
    C = res["C"]
    keep = torch.argsort(C)[:max_edges] if ei.numel() > max_edges \
        else torch.arange(ei.numel())
    frames = []
    if traj is not None:
        Tn = torch.as_tensor(traj).double().cpu()
        if Tn.dim() != 3 or Tn.shape[1] != X.shape[0]:
            raise ValueError(f"traj {tuple(Tn.shape)} does not match {X.shape[0]} nodes")
        frames = [Tn[t].numpy().round(5).reshape(-1).tolist()
                  for t in range(Tn.shape[0])]
    fids = (list(range(len(frames))) if frame_ids is None
            else [int(f) for f in frame_ids])
    payload = {
        "xyz": X.numpy().round(5).reshape(-1).tolist(),
        "labels": lab.tolist(),
        "node_class": klass,
        "class_counts": res.get("node_class_counts"),
        "dynamic_component": dyn_comp,
        "status": res.get("status"),
        "edges": [[int(ei[e]), int(ej[e]), float(C[e])] for e in keep.tolist()],
        "C_in": [None if v != v else round(float(v), 6)
                 for v in res["ambiguity"]["C_in"].tolist()],
        "C_out": [None if v != v else round(float(v), 6)
                  for v in res["ambiguity"]["C_out"].tolist()],
        "frames": frames,
        "frame_ids": fids,
        "gt": (gt_node.tolist() if gt_node is not None else []),
        "components": [{k: v for k, v in c.items() if k != "per_frame"}
                       for c in res.get("components", [])],
    }
    return write_support_page(path, payload, title)


def read_support_payload(path) -> dict:
    """The data embedded in an existing motion_coherence_support_3d.html."""
    s = Path(path).read_text()
    i = s.index("const D=") + len("const D=")
    j = s.index("\n", i)
    return json.loads(s[i:j].rstrip().rstrip(";"))


def write_support_page(path, payload: dict, title: str) -> Path:
    """Render the support page from its payload (also used to repair old pages)."""
    payload = dict(payload)
    payload.setdefault("frames", [])
    payload.setdefault("frame_ids", list(range(len(payload["frames"]))))
    html = """<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title><style>
 body{margin:0;background:#0e1116;color:#e6edf3;overflow:hidden;
  font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
 #app{display:flex;height:100vh} #view{flex:1;position:relative}
 #side{width:330px;background:#161b22;border-left:1px solid #2a313a;
  padding:12px;overflow-y:auto}
 h1{font-size:14px;margin:0 0 6px} h2{font-size:12px;color:#58a6ff;
  text-transform:uppercase;margin:14px 0 6px}
 label{display:flex;gap:6px;align-items:center;cursor:pointer}
 .sw{width:11px;height:11px;display:inline-block;border:1px solid #000}
 .row{display:flex;justify-content:space-between}.row span:last-child{color:#8b949e}
 #bar{position:absolute;left:0;right:0;bottom:0;padding:8px 12px;
  background:rgba(13,17,23,.93);border-top:1px solid #2a313a;display:flex;gap:10px}
 input[type=range]{flex:1} .note{color:#8b949e;font-size:11px}
</style></head><body><div id="app">
<div id="view"><canvas id="c"></canvas>
 <div id="bar"><button id="play">play</button>
 <input type="range" id="sl" min="0" max="0" value="0">
 <span id="fn">canonical</span></div></div>
<div id="side"><h1>__TITLE__</h1>
<div id="status" class="note"></div>
<h2>Show</h2><div id="tg"></div>
<h2>Components</h2><div id="comp"></div>
<h2>Legend</h2>
<div class="note"><span class="sw" style="background:#f06595"></span> predicted
DYNAMIC &nbsp; <span class="sw" style="background:#4a515c"></span> predicted
STATIC<br><span class="sw" style="background:#f0883e"></span> cross-component
graph edge (its length changes &mdash; this is the boundary the cut found)</div>
</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.min.js"></script>
<script>
const D=__PAYLOAD__;
const S=new THREE.Scene(),cv=document.getElementById('c');
const RE=new THREE.WebGLRenderer({canvas:cv,antialias:true});
S.background=new THREE.Color(0x0e1116);
const CAM=new THREE.PerspectiveCamera(55,1,0.01,4000);
function fit(){const v=document.getElementById('view');
 RE.setPixelRatio(devicePixelRatio);RE.setSize(v.clientWidth,v.clientHeight);
 CAM.aspect=v.clientWidth/Math.max(v.clientHeight,1);CAM.updateProjectionMatrix();}
const N=D.xyz.length/3; let ctr=[0,0,0];
for(let i=0;i<N;i++){ctr[0]+=D.xyz[i*3];ctr[1]+=D.xyz[i*3+1];ctr[2]+=D.xyz[i*3+2];}
ctr=ctr.map(v=>v/Math.max(N,1)); let rad=1e-3;
for(let i=0;i<N;i++)rad=Math.max(rad,Math.hypot(D.xyz[i*3]-ctr[0],
 D.xyz[i*3+1]-ctr[1],D.xyz[i*3+2]-ctr[2]));
let th=.9,ph=1.1,dist=rad*3.2; const tgt=new THREE.Vector3(...ctr);
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
const DYN=D.dynamic_component;
function mk(sel,color){const pts=[];for(let i=0;i<N;i++)if(sel(i))
  pts.push(D.xyz[i*3],D.xyz[i*3+1],D.xyz[i*3+2]);
 const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(pts),3));
 const o=new THREE.Points(g,new THREE.PointsMaterial({color:color,size:rad*0.02,
  sizeAttenuation:true})); S.add(o); return o;}
const K=D.node_class;
const isMov=i=>K?K[i]==='MOVING_RIGID':D.labels[i]===DYN;
const isAmb=i=>K?K[i]==='INCOHERENT_OR_AMBIGUOUS':false;
const idxDyn=[],idxSta=[],idxAmb=[];
for(let i=0;i<N;i++){if(isMov(i))idxDyn.push(i);else if(isAmb(i))idxAmb.push(i);
 else idxSta.push(i);}
const OB={dynamic:mk(isMov,0xf06595),
          static:mk(i=>!isMov(i)&&!isAmb(i),0x4a515c),
          ambiguous:mk(isAmb,0xf0883e)};
const segS=[],segX=[];
D.edges.forEach(([a,b])=>{const same=D.labels[a]===D.labels[b];
 (same?segS:segX).push(D.xyz[a*3],D.xyz[a*3+1],D.xyz[a*3+2],
  D.xyz[b*3],D.xyz[b*3+1],D.xyz[b*3+2]);});
function lines(arr,color,op){const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(arr),3));
 const o=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:color,
  transparent:true,opacity:op})); S.add(o); return o;}
OB.edges_same=lines(segS,0x2a313a,.35);
OB.edges_cross=lines(segX,0xf0883e,.9);
const tg=document.getElementById('tg');
for(const [k,label,c] of [['dynamic','MOVING_RIGID','#f06595'],
  ['static','STATIC_COMPATIBLE','#4a515c'],
  ['ambiguous','INCOHERENT_OR_AMBIGUOUS','#f0883e'],
  ['edges_same','within-component edges','#2a313a'],
  ['edges_cross','cross-component edges','#f0883e']]){
 tg.insertAdjacentHTML('beforeend','<label><input type="checkbox" id="t_'+k+
  '" checked><span class="sw" style="background:'+c+'"></span>'+label+'</label>');
 document.getElementById('t_'+k).addEventListener('change',e=>OB[k].visible=e.target.checked);}
document.getElementById('status').textContent='status: '+D.status+
 '   nodes: '+idxDyn.length+' MOVING_RIGID / '+idxSta.length+
 ' STATIC_COMPATIBLE / '+idxAmb.length+' AMBIGUOUS';
document.getElementById('comp').innerHTML=D.components.map((c,k)=>
 '<div class="row"><span>component '+k+(k===DYN?' (DYNAMIC)':' (STATIC)')+
 '</span><span>'+c.num_nodes+' nodes</span></div>'+
 '<div class="row"><span>&nbsp;&nbsp;E_rigid</span><span>'+
 (+c.E_rigid).toExponential(3)+'</span></div>'+
 '<div class="row"><span>&nbsp;&nbsp;E_identity</span><span>'+
 (+c.E_identity).toExponential(3)+'</span></div>'+
 '<div class="row"><span>&nbsp;&nbsp;gain</span><span>'+
 (+c.gain).toExponential(3)+'</span></div>').join('');
const sl=document.getElementById('sl'),fn=document.getElementById('fn'),
 pb=document.getElementById('play');
// node trajectories in ORIGINAL chronological order; every drawn object moves
// with them (all three node classes and both edge sets)
const pairsS=D.edges.filter(([a,b])=>D.labels[a]===D.labels[b]),
 pairsX=D.edges.filter(([a,b])=>D.labels[a]!==D.labels[b]);
function goto(t){const f=D.frames[t];fn.textContent='frame '+D.frame_ids[t]+
  ' ('+(t+1)+'/'+D.frames.length+')';
 [['dynamic',idxDyn],['static',idxSta],['ambiguous',idxAmb]].forEach(([k,ids])=>{
  const a=OB[k].geometry.attributes.position.array;
  ids.forEach((i,n)=>{a[n*3]=f[i*3];a[n*3+1]=f[i*3+1];a[n*3+2]=f[i*3+2];});
  OB[k].geometry.attributes.position.needsUpdate=true;});
 const upd=(o,pairs)=>{const a=o.geometry.attributes.position.array;let n=0;
  pairs.forEach(([x,y])=>{a[n++]=f[x*3];a[n++]=f[x*3+1];a[n++]=f[x*3+2];
   a[n++]=f[y*3];a[n++]=f[y*3+1];a[n++]=f[y*3+2];});
  o.geometry.attributes.position.needsUpdate=true;};
 upd(OB.edges_same,pairsS);upd(OB.edges_cross,pairsX);}
if(D.frames.length){sl.max=D.frames.length-1;
 sl.addEventListener('input',e=>goto(+e.target.value));
 let timer=null;
 pb.addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;
   pb.textContent='play';return;}
  pb.textContent='pause';timer=setInterval(()=>{sl.value=(+sl.value+1)%D.frames.length;
   goto(+sl.value);},120);});
 goto(0);}
else {sl.disabled=true;pb.disabled=true;
 fn.textContent='no per-frame trajectories in this page';}
(function loop(){requestAnimationFrame(loop);RE.render(S,CAM);})();
</script></body></html>"""
    html = html.replace("__TITLE__", title).replace(
        "__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html); return p


# ===========================================================================
# Section 21: the candidate tree
# ===========================================================================

def write_candidate_tree(path, extraction: dict, title: str) -> Path:
    """Every candidate, its coherence statistics and its held-out evidence.

    Section 21 requires Storage and Blade to be understandable from this page,
    so the four scatter plots share one hover and the selected candidate is
    marked in all of them: the question "why that component and not a bigger
    one" is answered by reading size against eta and against per-node-frame
    motion, not by trusting a scalar.
    """
    import json as _json

    sel = (extraction.get("selection") or {}).get("selected")
    rows = []
    for c in extraction.get("candidates", []):
        r, rw, e = c["rigidity"], c.get("rigidity_world", {}), c["edges"]
        rows.append({
            "id": c["candidate_id"], "n": c["num_members"],
            "birth": c["birth_edge_cost"],
            "death": (None if c["death_merge_edge_cost"] == float("inf")
                      else c["death_merge_edge_cost"]),
            "persistence": (None if c["persistence"] == float("inf")
                            else c["persistence"]),
            "iq50": e["internal_q50"], "iq75": e["internal_q75"],
            "iq90": e["internal_q90"],
            "bq10": e["boundary_q10"], "bq25": e["boundary_q25"],
            "bq50": e["boundary_q50"], "bdry": e["num_boundary_edges"],
            "rss_i": r.get("rss_identity_cv"), "rss_r": r.get("rss_rigid_cv"),
            "eta": r.get("eta_cv"), "drss": r.get("delta_rss_cv"),
            "dbic": r.get("delta_bic"),
            "relmot": r.get("relative_motion_ratio"),
            "pernf": rw.get("delta_rss_cv_per_node_frame"),
            "sel": int(c["candidate_id"] == sel),
        })
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title><style>{_CSS}
 .grid{{display:grid;grid-template-columns:repeat(2,minmax(380px,1fr));gap:14px}}
 .sel{{background:#1d2b1d}} tr:hover{{background:#1b2230}}
 canvas{{background:#0e1116}}
</style></head><body>
<h1>{title}</h1>
<p class="note">Section 21. Each row is a candidate component from the
coherence single-linkage hierarchy. <b>eta_cv</b>, <b>dRSS</b> and
<b>dBIC</b> are measured in the COMPLEMENT-RELATIVE frame (does articulation
exist); <b>perNF</b> is held-out delta-RSS per node-frame in the WORLD frame
(which side moves). The selected candidate is highlighted and ringed in the
plots. delta_BIC is a diagnostic and does not gate selection.</p>
<div class="grid">
 <div><h2>size vs eta_cv (relative)</h2><canvas id="p1" width="430" height="270"></canvas></div>
 <div><h2>size vs per-node-frame motion (world)</h2><canvas id="p2" width="430" height="270"></canvas></div>
 <div><h2>size vs persistence</h2><canvas id="p3" width="430" height="270"></canvas></div>
 <div><h2>internal q75 vs boundary q25</h2><canvas id="p4" width="430" height="270"></canvas></div>
</div>
<h2>candidates</h2>
<table id="t"><tr><th>id</th><th>n</th><th>birth C</th><th>death C</th>
<th>persist</th><th>int q50</th><th>int q75</th><th>int q90</th>
<th>bnd q10</th><th>bnd q25</th><th>bnd q50</th><th>RSS_I</th><th>RSS_R</th>
<th>eta_cv</th><th>dRSS</th><th>dBIC</th><th>rel mot</th><th>perNF</th></tr></table>
<script>
const R={_json.dumps(rows)};
const f=(v,d)=>v===null||v===undefined||v!==v?'&mdash;':(Math.abs(v)<1e-3&&v!==0?v.toExponential(2):v.toFixed(d));
document.getElementById('t').insertAdjacentHTML('beforeend', R.map(r=>
 '<tr class="'+(r.sel?'sel':'')+'"><td>'+r.id+'</td><td>'+r.n+'</td><td>'+
 f(r.birth,4)+'</td><td>'+f(r.death,4)+'</td><td>'+f(r.persistence,4)+'</td><td>'+
 f(r.iq50,4)+'</td><td>'+f(r.iq75,4)+'</td><td>'+f(r.iq90,4)+'</td><td>'+
 f(r.bq10,4)+'</td><td>'+f(r.bq25,4)+'</td><td>'+f(r.bq50,4)+'</td><td>'+
 f(r.rss_i,3)+'</td><td>'+f(r.rss_r,3)+'</td><td>'+f(r.eta,4)+'</td><td>'+
 f(r.drss,3)+'</td><td>'+f(r.dbic,2)+'</td><td>'+f(r.relmot,3)+'</td><td>'+
 f(r.pernf,3)+'</td></tr>').join(''));
function plot(id,xk,yk,logx,logy){{
 const cv=document.getElementById(id),g=cv.getContext('2d'),W=cv.width,H=cv.height,M=42;
 const pts=R.filter(r=>r[xk]!==null&&r[yk]!==null&&r[xk]===r[xk]&&r[yk]===r[yk]);
 if(!pts.length)return;
 const tx=v=>logx?Math.log10(Math.max(v,1e-12)):v, ty=v=>logy?Math.log10(Math.max(v,1e-12)):v;
 let x0=Math.min(...pts.map(p=>tx(p[xk]))),x1=Math.max(...pts.map(p=>tx(p[xk])));
 let y0=Math.min(...pts.map(p=>ty(p[yk]))),y1=Math.max(...pts.map(p=>ty(p[yk])));
 if(x1===x0)x1=x0+1; if(y1===y0)y1=y0+1;
 g.strokeStyle='#2a313a';g.strokeRect(M,6,W-M-8,H-M-6);
 g.fillStyle='#8b949e';g.font='10px monospace';
 g.fillText((logx?'log10 ':'')+xk,W/2-20,H-8);
 g.save();g.translate(11,H/2+20);g.rotate(-Math.PI/2);
 g.fillText((logy?'log10 ':'')+yk,0,0);g.restore();
 g.fillText(x0.toPrecision(3),M,H-24); g.fillText(x1.toPrecision(3),W-60,H-24);
 g.fillText(y1.toPrecision(3),4,14); g.fillText(y0.toPrecision(3),4,H-M+2);
 pts.forEach(p=>{{
  const X=M+(tx(p[xk])-x0)/(x1-x0)*(W-M-14), Y=H-M-(ty(p[yk])-y0)/(y1-y0)*(H-M-12);
  g.fillStyle=p.sel?'#3fb950':'#58a6ff'; g.beginPath();
  g.arc(X,Y,p.sel?5:2.6,0,7); g.fill();
  if(p.sel){{g.strokeStyle='#3fb950';g.beginPath();g.arc(X,Y,9,0,7);g.stroke();}}
 }});
}}
plot('p1','n','eta',true,false); plot('p2','n','pernf',true,true);
plot('p3','n','persistence',true,true); plot('p4','iq75','bq25',true,true);
</script></body></html>"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc)
    return p


# ===========================================================================
# Section 24: the two selectors, side by side, per GAUSSIAN
# ===========================================================================

def stratified_indices(gt, cur, coh, max_points: int = 8000, seed: int = 0):
    """Keep every point the two selectors disagree on, then fill with the rest.

    A uniform sample is the wrong tool here. The disagreements are the whole
    subject of the page and on a scene like storage they are a few percent of
    a few percent -- uniform sampling would show a handful of them. So the
    groups are filled in order of how much they say, and the side panel
    reports the FULL counts so nobody reads on-screen density as a fraction.
    """
    n = int(gt.numel())
    g = torch.Generator().manual_seed(int(seed))
    disagree = cur != coh
    wrong = ((cur != (gt == 1)) | (coh != (gt == 1))) & (gt >= 0)
    groups = [disagree,                      # the comparison itself
              wrong & ~disagree,             # agreed, and both wrong
              (gt == 1) & ~disagree & ~wrong,  # agreed-correct dynamic
              ~disagree & ~wrong & (gt != 1)]  # the static bulk
    keep, budget = [], int(max_points)
    for k, m in enumerate(groups):
        idx = torch.nonzero(m, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue
        if idx.numel() > budget:
            idx = idx[torch.randperm(idx.numel(), generator=g)[:budget]]
        keep.append(idx)
        budget -= idx.numel()
        if budget <= 0:
            break
    sel = (torch.cat(keep).sort().values if keep
           else torch.arange(min(n, max_points)))
    return sel


def write_selector_comparison_3d(path, mu0, positions, gt, cur, coh,
                                 metrics=None, title="static/dynamic",
                                 max_points: int = 8000,
                                 max_frames: int = 20, seed: int = 0) -> Path:
    """The motion baseline and coherence v2 on the SAME points, in 3D.

    Four views over one shared world frame and one frame slider:

        GT            the optimized-3DGS labels
        baseline      TP / FP / FN / TN of the Otsu excursion split
        coherence     TP / FP / FN / TN of the v2 extractor
        A vs B        where the two selectors disagree, and which way

    The slider plays the Phase-1 trajectories, so a point labelled STATIC that
    visibly travels is a false negative you can see rather than infer.
    """
    mu0 = torch.as_tensor(mu0).double().cpu()
    gt = torch.as_tensor(gt).long().cpu().reshape(-1)
    cur = torch.as_tensor(cur).bool().cpu().reshape(-1)
    coh = torch.as_tensor(coh).bool().cpu().reshape(-1)
    sel = stratified_indices(gt, cur, coh, max_points=max_points, seed=seed)

    T = len(positions)
    step = max(1, (T + max_frames - 1) // max_frames)
    ft = list(range(0, T, step))
    if ft and ft[-1] != T - 1:
        ft.append(T - 1)
    frames = [torch.as_tensor(positions[t]).double().cpu()[sel]
              .numpy().round(4).reshape(-1).tolist() for t in ft]

    full = {"n": int(gt.numel()),
            "gt_dynamic": int((gt == 1).sum()),
            "gt_static": int((gt == 0).sum()),
            "gt_unlabelled": int((gt < 0).sum()),
            "cur_dynamic": int(cur.sum()), "coh_dynamic": int(coh.sum()),
            "disagree": int((cur != coh).sum()),
            "cur_only": int((cur & ~coh).sum()),
            "coh_only": int((coh & ~cur).sum()),
            "both_dynamic": int((cur & coh).sum())}

    payload = {
        "xyz": mu0[sel].numpy().round(4).reshape(-1).tolist(),
        "gt": gt[sel].tolist(),
        "cur": cur[sel].long().tolist(),
        "coh": coh[sel].long().tolist(),
        "frames": frames, "frame_ids": ft,
        "shown": int(sel.numel()), "full": full,
        "metrics": metrics or {},
        "stratified": bool(sel.numel() < gt.numel())}

    html = r"""<!doctype html><html><head><meta charset="utf-8">
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
 table{width:100%;border-collapse:collapse;font-size:11px}
 td,th{padding:2px 4px;text-align:right}th{color:#58a6ff;font-weight:400}
 td:first-child,th:first-child{text-align:left}
 .win{color:#3fb950}.lose{color:#f85149}
 button{background:#21262d;color:#e6edf3;border:1px solid #2a313a;
  border-radius:4px;padding:3px 9px;cursor:pointer;font:inherit}
</style></head><body><div id="app">
<div id="view"><canvas id="c"></canvas>
 <div id="bar"><button id="play">play</button>
 <input type="range" id="sl" min="0" max="0" value="0">
 <span id="fn">frame 0</span></div></div>
<div id="side"><h1>__TITLE__</h1>
<div id="shown" class="note"></div>
<h2>View</h2><div id="views"></div>
<h2>Classes</h2><div id="tg"></div>
<h2>Scored against GT</h2><div id="mt"></div>
<h2>Counts (all Gaussians)</h2><div id="cnt"></div>
<div class="note" id="strat"></div>
</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.min.js"></script>
<script>
const D=__PAYLOAD__;
const S=new THREE.Scene(),cv=document.getElementById('c');
const RE=new THREE.WebGLRenderer({canvas:cv,antialias:true});
S.background=new THREE.Color(0x0e1116);
const CAM=new THREE.PerspectiveCamera(55,1,0.01,4000);
function fit(){const v=document.getElementById('view');
 RE.setPixelRatio(devicePixelRatio);RE.setSize(v.clientWidth,v.clientHeight);
 CAM.aspect=v.clientWidth/Math.max(v.clientHeight,1);CAM.updateProjectionMatrix();}
const N=D.xyz.length/3; let ctr=[0,0,0];
for(let i=0;i<N;i++){ctr[0]+=D.xyz[i*3];ctr[1]+=D.xyz[i*3+1];ctr[2]+=D.xyz[i*3+2];}
ctr=ctr.map(v=>v/Math.max(N,1)); let rad=1e-3;
for(let i=0;i<N;i++)rad=Math.max(rad,Math.hypot(D.xyz[i*3]-ctr[0],
 D.xyz[i*3+1]-ctr[1],D.xyz[i*3+2]-ctr[2]));
let th=.9,ph=1.1,dist=rad*3.2; const tgt=new THREE.Vector3(...ctr);
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

// ---- the four views. Each is a map class-name -> per-point predicate ----
const isDyn=i=>D.gt[i]===1, lab=i=>D.gt[i]>=0;
function conf(p){return {
 'true positive':  i=>lab(i)&&p[i]===1&&isDyn(i),
 'false positive': i=>lab(i)&&p[i]===1&&!isDyn(i),
 'false negative': i=>lab(i)&&p[i]===0&&isDyn(i),
 'true negative':  i=>lab(i)&&p[i]===0&&!isDyn(i),
 'unlabelled':     i=>!lab(i)};}
const COL={'true positive':0x3fb950,'false positive':0xf0883e,
 'false negative':0xf85149,'true negative':0x3a4049,'unlabelled':0x1f6feb,
 'GT dynamic':0xf06595,'GT static':0x3a4049,
 'both dynamic':0xa371f7,'baseline only':0xf0883e,'coherence only':0x58a6ff,
 'both static':0x3a4049};
const VIEWS={
 'GT (optimized 3DGS)':{'GT dynamic':isDyn,'GT static':i=>D.gt[i]===0,
   'unlabelled':i=>!lab(i)},
 'motion baseline (Otsu)':conf(D.cur),
 'coherence v2':conf(D.coh),
 'baseline vs coherence':{
   'both dynamic': i=>D.cur[i]===1&&D.coh[i]===1,
   'baseline only':i=>D.cur[i]===1&&D.coh[i]===0,
   'coherence only':i=>D.cur[i]===0&&D.coh[i]===1,
   'both static':  i=>D.cur[i]===0&&D.coh[i]===0}};

// One Points object per (view, class). Index lists are kept so the frame
// slider can rewrite positions without rebuilding any geometry.
const OBJ={},IDX={};
for(const v in VIEWS)for(const k in VIEWS[v]){
 const ids=[];for(let i=0;i<N;i++)if(VIEWS[v][k](i))ids.push(i);
 const a=new Float32Array(ids.length*3);
 ids.forEach((i,n)=>{a[n*3]=D.xyz[i*3];a[n*3+1]=D.xyz[i*3+1];a[n*3+2]=D.xyz[i*3+2];});
 const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(a,3));
 const o=new THREE.Points(g,new THREE.PointsMaterial({color:COL[k],
  size:rad*(k==='true negative'||k==='both static'||k==='GT static'?0.012:0.022),
  sizeAttenuation:true,transparent:true,
  opacity:(k==='true negative'||k==='both static'||k==='GT static')?0.5:1}));
 o.visible=false;S.add(o);OBJ[v+'|'+k]=o;IDX[v+'|'+k]=ids;}

let cur=Object.keys(VIEWS)[3];
const tg=document.getElementById('tg');
function drawClasses(){tg.innerHTML='';
 for(const k in VIEWS[cur]){const n=IDX[cur+'|'+k].length;
  tg.insertAdjacentHTML('beforeend','<label><input type="checkbox" checked '+
   'data-k="'+k+'"><span class="sw" style="background:#'+
   COL[k].toString(16).padStart(6,'0')+'"></span>'+k+
   ' <span style="color:#8b949e">('+n+')</span></label>');}
 tg.querySelectorAll('input').forEach(e=>e.addEventListener('change',ev=>{
  OBJ[cur+'|'+ev.target.dataset.k].visible=ev.target.checked;}));}
function show(v){for(const key in OBJ)OBJ[key].visible=false;
 cur=v;for(const k in VIEWS[v])OBJ[v+'|'+k].visible=true;drawClasses();}
const vs=document.getElementById('views');
Object.keys(VIEWS).forEach((v,i)=>{
 vs.insertAdjacentHTML('beforeend','<label><input type="radio" name="v" '+
  (i===3?'checked':'')+' value="'+v+'">'+v+'</label>');});
vs.querySelectorAll('input').forEach(e=>e.addEventListener('change',
 ev=>show(ev.target.value)));
show(cur);

// ---- metrics table ------------------------------------------------------
const M=D.metrics;
if(M.current&&M.coherence){
 const r=(k,d)=>{const a=M.current[k],b=M.coherence[k];
  return '<tr><td>'+d+'</td><td>'+a.toFixed(3)+'</td><td class="'+
   (b>a?'win':(b<a?'lose':''))+'">'+b.toFixed(3)+'</td></tr>';};
 document.getElementById('mt').innerHTML='<table><tr><th></th>'+
  '<th>baseline</th><th>coherence</th></tr>'+
  r('dynamic_precision','precision')+r('dynamic_recall','recall')+
  r('dynamic_f1','F1')+r('dynamic_iou','IoU')+
  r('balanced_accuracy','bal. acc')+'</table>';}
const F=D.full;
document.getElementById('cnt').innerHTML=
 [['Gaussians',F.n],['GT dynamic',F.gt_dynamic],['GT static',F.gt_static],
  ['GT unlabelled',F.gt_unlabelled],['baseline dynamic',F.cur_dynamic],
  ['coherence dynamic',F.coh_dynamic],['they disagree on',F.disagree],
  ['&nbsp;&nbsp;baseline only',F.cur_only],
  ['&nbsp;&nbsp;coherence only',F.coh_only],
  ['&nbsp;&nbsp;both dynamic',F.both_dynamic]]
 .map(([a,b])=>'<div class="row"><span>'+a+'</span><span>'+b+'</span></div>').join('');
document.getElementById('shown').textContent=
 'showing '+D.shown+' of '+F.n+' Gaussians, '+D.frames.length+' of '+
 (D.frame_ids[D.frame_ids.length-1]+1)+' frames';
document.getElementById('strat').innerHTML=D.stratified?
 'The view is STRATIFIED: every Gaussian the two selectors disagree on is '+
 'kept, the agreeing bulk is subsampled. On-screen density is therefore not '+
 'a fraction &mdash; read the counts above for that.':'';

// ---- frame slider -------------------------------------------------------
const sl=document.getElementById('sl'),fn=document.getElementById('fn');
sl.max=D.frames.length-1;
function goto(t){const f=D.frames[t];fn.textContent='frame '+D.frame_ids[t];
 for(const key in OBJ){const ids=IDX[key],a=OBJ[key].geometry.attributes.position.array;
  for(let n=0;n<ids.length;n++){const i=ids[n];
   a[n*3]=f[i*3];a[n*3+1]=f[i*3+1];a[n*3+2]=f[i*3+2];}
  OBJ[key].geometry.attributes.position.needsUpdate=true;}}
sl.addEventListener('input',e=>goto(+e.target.value));
let timer=null;
document.getElementById('play').addEventListener('click',e=>{
 if(timer){clearInterval(timer);timer=null;e.target.textContent='play';return;}
 e.target.textContent='stop';
 timer=setInterval(()=>{sl.value=(+sl.value+1)%D.frames.length;
  goto(+sl.value);},120);});
(function loop(){requestAnimationFrame(loop);RE.render(S,CAM);})();
</script></body></html>"""
    html = html.replace("__TITLE__", title).replace(
        "__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html)
    return p
