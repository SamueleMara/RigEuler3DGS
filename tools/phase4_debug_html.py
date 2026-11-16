#!/usr/bin/env python
"""Phase-4 membership debug page, in the style of the Phase-3 rigid_to_joint_debug.html (three.js, dark panel,
frame slider). Read-only, CPU only; reads a finished Phase-4 run and writes <RUN>/html/phase4_membership_debug.html.

    python tools/phase4_debug_html.py --phase4-dir OUT/<scene>/phase4/<RUN> [--phase4-dir ...]

What it shows, animated over every Phase-3 frame with the FIXED joint:
  tab "Phase 3"   only the Phase-3 support (4A CORE) moves -- what Phase 3 animated
  tab "Phase 4"   every Gaussian whose final render label is moving moves (core + 4A recalled + 4C births +
                  provisional unresolved) -- so the recalled points visibly lag in one tab and follow in the other
  layers          one per 4A status (CORE / recalled MOVING_CONFIRMED / STATIC / each UNRESOLVED kind / conflicts),
                  4C births, and -- synthetic only, EVALUATION ONLY -- GT-moving Gaussians still drawn static
                  ("missed") and GT-static drawn moving ("false moving")
  panel           per-status counts, Phase 3 vs Phase 4 precision / recall against GT, the unresolved breakdown,
                  4B / 4C summary, q at the current frame
GT never feeds anything back: it is read only here, after Phase 4 finished.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

STATUS = ["CORE_MOVING_ANCHOR", "MOVING_CONFIRMED", "STATIC_CONFIRMED", "UNRESOLVED_AMBIGUOUS",
          "SUSPECT_ORIGINAL_GEOMETRY", "UPSTREAM_JOINT_OR_SUPPORT_CONFLICT", "UNTESTED_RESOURCE_LIMIT",
          "UNRESOLVED_UNOBSERVABLE", "4C_ACCEPTED_BIRTH", "SUPPORT_RELABELLED_STATIC"]
LABEL = ["Phase-3 support (4A core)", "recalled by 4A (MOVING_CONFIRMED)", "static (STATIC_CONFIRMED)",
         "unresolved: ambiguous", "suspect original geometry", "upstream joint/support conflict",
         "untested (resource limit)", "unresolved: unobservable", "4C accepted moving birth",
         "Phase-3 support relabelled STATIC by 4A"]
COLOR = ["#4cc9f0", "#90e07a", "#454b55", "#ffd166", "#d2a8ff", "#ff4d4d", "#ffffff", "#f0883e", "#f06595", "#ff7b00"]
SIZE = [0.012, 0.016, 0.009, 0.018, 0.016, 0.018, 0.016, 0.018, 0.016, 0.022]
OPAC = [0.75, 1.0, 0.35, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def _ply_xyz(path):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    return np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1).astype(np.float64)


def _joint(jd):
    from recovery.contracts import Joint
    fr = tuple(int(f) for f in jd["frames_processing_order"])
    return Joint(type=jd["type"], axis=tuple(jd["axis"]), point=tuple(jd["point"]) if jd.get("point") else None,
                 frames=fr, q=tuple(float(x) for x in jd["q"]), posterior_sd_q=tuple(0.0 for _ in fr),
                 direction=jd.get("direction"), geometry_file=jd.get("geometry_file"),
                 trajectory_file=jd.get("trajectory_file"), geometry_sha256=jd.get("geometry_sha256"),
                 trajectory_sha256=jd.get("trajectory_sha256"))


def _prf(pred, gt):
    lab = gt >= 0
    p, g = pred[lab], gt[lab] == 1
    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": pr, "recall": rc, "f1": 2 * pr * rc / max(1e-12, pr + rc)}


def build_payload(run: Path) -> dict:
    a_dir, b_dir, c_dir = run / "4a_original_membership", run / "4b_new_surface_proposals", run / "4c_accepted_moving"
    man = json.loads((a_dir / "manifest.json").read_text())
    ca = json.loads((a_dir / "complete.json").read_text()) if (a_dir / "complete.json").is_file() else None
    cb = json.loads((b_dir / "complete.json").read_text()) if (b_dir / "complete.json").is_file() else None
    cc = json.loads((c_dir / "complete.json").read_text()) if (c_dir / "complete.json").is_file() else None
    z = np.load(a_dir / "membership.npz")
    status = z["status_code"].astype(np.int64)
    render = z["render_label"].astype(bool)
    n0 = len(status)
    core_member = z["core_member"].astype(bool) if "core_member" in z.files else status == 0
    status = np.where(core_member & (status == 2), 9, status)      # support rows 4A relabelled STATIC
    setup = man["setup"]
    fc = c_dir / "final_canonical"
    if cc is not None and (fc / "point_cloud.ply").is_file():
        xyz = _ply_xyz(fc / "point_cloud.ply")
        mat = np.load(fc / "material_ids.npz")["material"].astype(bool)
        nb = len(xyz) - n0
        status = np.concatenate([status, np.full(nb, 8)])
        p4 = mat
    else:                                           # 4C missing: originals from the upstream model, 4A labels only
        xyz = _ply_xyz(Path(setup["model"]) / "point_cloud" / sorted(
            p.name for p in (Path(setup["model"]) / "point_cloud").iterdir())[-1] / "point_cloud.ply")[:n0]
        p4 = render
    p3 = np.concatenate([core_member, np.zeros(len(xyz) - n0, bool)])
    j = _joint(man["joint"])
    frames = sorted(j.frames)
    poses, qs = [], []
    for f in frames:
        Q, p = j.transform(f)
        poses.append(np.concatenate([np.asarray(Q).reshape(-1), np.asarray(p)]).round(8).tolist())
        qs.append(j.q_of(f))
    # ground truth: EVALUATION ONLY (synthetic scenes ship a labelled endpoint cloud)
    gt = None
    try:
        from utils import gt_part_labels as G
        cl = G.labelled_cloud(setup["source"], "initial" if setup["direction"] == "forward" else "final")
        if cl is not None:
            gt = G.dynamic_from_part(G.nn_labels(xyz, cl)["part"]).astype(np.int64)
    except Exception as e:                          # pragma: no cover
        print(f"  GT labels unavailable: {e}")
    rows = []
    for k, name in enumerate(STATUS):
        m = status == k
        if not m.any():
            continue
        r = {"k": k, "status": name, "n": int(m.sum()), "moves_p3": int((m & p3).sum()), "moves_p4": int((m & p4).sum())}
        if gt is not None:
            lab = m & (gt >= 0)
            r["gt_moving_pct"] = float((gt[lab] == 1).mean() * 100) if lab.any() else None
        rows.append(r)
    ev = None
    if gt is not None:
        orig = np.arange(len(xyz)) < n0
        ev = {"phase3_support": _prf(p3 & orig, np.where(orig, gt, -1)),
              "phase4_originals": _prf(p4 & orig, np.where(orig, gt, -1)),
              "phase4_with_births": _prf(p4, gt),
              "births_gt_moving_pct": float((gt[(status == 8) & (gt >= 0)] == 1).mean() * 100)
              if ((status == 8) & (gt >= 0)).any() else None,
              "unlabelled": int((gt < 0).sum())}
    relab = None
    if (status == 9).any():
        relab = {"n": int((status == 9).sum())}
        if gt is not None:
            m = (status == 9) & (gt >= 0)
            relab["gt_static_pct"] = float((gt[m] == 0).mean() * 100) if m.any() else None
    ctr = xyz.mean(0)
    return {
        "title": f"{Path(setup['source']).name} - Phase 4 membership (4A recall, unresolved, 4C births)",
        "n": len(xyz), "n0": n0, "xyz": xyz.round(5).reshape(-1).tolist(), "status": status.tolist(),
        "p3": p3.astype(int).tolist(), "p4": p4.astype(int).tolist(),
        "gt": gt.tolist() if gt is not None else None, "frames": frames, "q": qs, "poses": poses,
        "units": man["joint"].get("units", ""), "joint_type": j.type, "direction": setup["direction"],
        "anchor": int(man["joint"].get("anchor_original_frame", frames[0])),
        "labels": LABEL, "colors": COLOR, "sizes": SIZE, "opac": OPAC, "rows": rows, "eval": ev, "relabelled": relab,
        "core_relabelled": (ca or {}).get("core_relabelled_static"),
        "stages": {"4a": "complete" if ca else "INCOMPLETE", "4b": "complete" if cb else "missing / incomplete",
                   "4c": "complete" if cc else "missing / incomplete"},
        "core_audit": (ca or {}).get("core_audit"),
        "b": {"candidates": cb.get("candidates"), "sigma_model_m": cb.get("sigma_model_m"),
              "recovery_views": cb.get("recovery_views")} if cb else None,
        "c": {"births": cc.get("N_accepted_births"), "counts": cc.get("counts"), "cycles": cc.get("cycles"),
              "recovery_views": cc.get("recovery_views")} if cc else None,
        "not_for_results": bool(man.get("NOT_FOR_RESULTS")), "center": ctr.round(5).tolist(),
    }


SHELL = r"""<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title><style>
 body{margin:0;background:#0e1116;color:#e6edf3;overflow:hidden;
  font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
 #app{display:flex;height:100vh} #view{flex:1;position:relative}
 #side{width:400px;background:#161b22;border-left:1px solid #2a313a;
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
<div><span class="tab" id="tab3">Phase 3 motion</span>
<span class="tab on" id="tab4">Phase 4 motion</span></div>
<div id="modenote" class="note" style="margin-top:8px"></div>
<h2>Layers</h2><div id="lay"></div>
<h2>Against GT (evaluation only)</h2><div id="gtl"></div>
<h2>4A statuses</h2><div id="tbl"></div>
<h2>Phase 3 support vs Phase 4</h2><div id="ev"></div>
<h2>Phase-3 support revised by 4A</h2><div id="rev"></div>
<h2>Unresolved</h2><div id="unr"></div>
<h2>4B / 4C</h2><div id="bc"></div>
<h2>At this frame</h2><div id="now"></div>
</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.min.js"></script>
<script>
const D=__PAYLOAD__;
const N=D.n, X=D.xyz;
const S=new THREE.Scene(),cv=document.getElementById('c');
const RE=new THREE.WebGLRenderer({canvas:cv,antialias:true});
S.background=new THREE.Color(0x0e1116);
const CAM=new THREE.PerspectiveCamera(55,1,0.001,8000);
function fit(){const v=document.getElementById('view');
 RE.setPixelRatio(devicePixelRatio);RE.setSize(v.clientWidth,v.clientHeight);
 CAM.aspect=v.clientWidth/Math.max(v.clientHeight,1);CAM.updateProjectionMatrix();}
const ctr=D.center; let rad=1e-3;
for(let i=0;i<N;i++)rad=Math.max(rad,Math.hypot(X[i*3]-ctr[0],X[i*3+1]-ctr[1],X[i*3+2]-ctr[2]));
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
const hx=s=>parseInt(s.slice(1),16);

// one point layer per index set; positions recomputed per frame from canonical + the fixed joint pose
const L={};
function layer(key,idx,color,size,op,visible){
 const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(idx.length*3),3));
 const o=new THREE.Points(g,new THREE.PointsMaterial({color:color,size:rad*size,
  sizeAttenuation:true,transparent:true,opacity:op}));
 o.visible=visible;S.add(o);L[key]={o:o,idx:idx};}
const byStatus=D.labels.map(()=>[]);
for(let i=0;i<N;i++)byStatus[D.status[i]].push(i);
byStatus.forEach((idx,k)=>{if(idx.length)layer('s'+k,idx,hx(D.colors[k]),D.sizes[k],D.opac[k],true);});
if(D.gt){
 const miss=[],fpos=[],rec=[];
 for(let i=0;i<N;i++){
  if(D.gt[i]===1&&!D.p4[i])miss.push(i);
  if(D.gt[i]===0&&D.p4[i])fpos.push(i);
  if(D.gt[i]===1&&D.p4[i]&&!D.p3[i])rec.push(i);}
 layer('g_miss',miss,0xff2d2d,0.024,1.0,false);
 layer('g_fpos',fpos,0xff00ff,0.024,1.0,false);
 layer('g_rec',rec,0x2dff6a,0.020,1.0,false);}

let mode='p4';
function setFrame(t){
 const q=D.poses[t],M=(mode==='p3')?D.p3:D.p4;
 for(const k in L){const a=L[k].o.geometry.attributes.position.array,idx=L[k].idx;
  for(let j=0;j<idx.length;j++){const i=idx[j],x=X[i*3],y=X[i*3+1],z=X[i*3+2];
   if(M[i]){a[j*3]=q[0]*x+q[1]*y+q[2]*z+q[9];a[j*3+1]=q[3]*x+q[4]*y+q[5]*z+q[10];
    a[j*3+2]=q[6]*x+q[7]*y+q[8]*z+q[11];}
   else{a[j*3]=x;a[j*3+1]=y;a[j*3+2]=z;}}
  L[k].o.geometry.attributes.position.needsUpdate=true;}
 document.getElementById('fn').textContent='frame '+D.frames[t];
 document.getElementById('now').innerHTML=row('original frame',D.frames[t]+(D.frames[t]===D.anchor?' (anchor)':''))+
  row('q',D.q[t].toFixed(4)+' '+D.units)+row('moving drawn',(mode==='p3'?'Phase-3 support only':'Phase-4 render labels'));}
function row(a,b,w){return '<div class="row"><span>'+a+'</span><span'+(w?' class="warn"':'')+'>'+b+'</span></div>';}
function toggles(host,keys){
 const h=document.getElementById(host);h.innerHTML='';
 keys.forEach(([k,label,color])=>{if(!L[k])return;
  h.insertAdjacentHTML('beforeend','<label><input type="checkbox" '+(L[k].o.visible?'checked':'')+
   ' data-k="'+k+'"><span class="sw" style="background:'+color+'"></span>'+label+
   ' <span class="note">('+L[k].idx.length+')</span></label>');});
 h.querySelectorAll('input').forEach(e=>e.addEventListener('change',ev=>{L[ev.target.dataset.k].o.visible=ev.target.checked;}));}
toggles('lay',D.labels.map((l,k)=>['s'+k,l,D.colors[k]]));
if(D.gt){toggles('gtl',[['g_rec','recalled by Phase 4 (GT moving, not in Phase-3 support)','#2dff6a'],
 ['g_miss','missed: GT moving, drawn static','#ff2d2d'],['g_fpos','false moving: GT static, drawn moving','#ff00ff']]);
 document.getElementById('gtl').insertAdjacentHTML('beforeend','<div class="note">Nearest labelled GT surface sample; '+
  D.eval.unlabelled+' Gaussians too far from any sample are unlabelled and not counted. Never used by Phase 4.</div>');}
else document.getElementById('gtl').innerHTML='<div class="note">no GT part labels for this scene</div>';
{let h='<table><tr><th>status</th><th>n</th><th>moves P3</th><th>moves P4</th>'+(D.gt?'<th>GT mov %</th>':'')+'</tr>';
 D.rows.forEach(r=>{h+='<tr><td><span class="sw" style="background:'+D.colors[r.k]+'"></span> '+D.labels[r.k]+'</td><td>'+r.n+
  '</td><td>'+r.moves_p3+'</td><td>'+r.moves_p4+'</td>'+(D.gt?'<td>'+(r.gt_moving_pct==null?'-':r.gt_moving_pct.toFixed(1))+'</td>':'')+'</tr>';});
 document.getElementById('tbl').innerHTML=h+'</table>'+(D.not_for_results?'<div class="warn">NOT_FOR_RESULTS (view subset run)</div>':'');}
if(D.eval){const e=D.eval,f=x=>x.toFixed(3);
 let h='<table><tr><th></th><th>prec</th><th>recall</th><th>F1</th><th>missed</th><th>false</th></tr>';
 [['Phase 3 support',e.phase3_support],['Phase 4 (originals)',e.phase4_originals],['Phase 4 + births',e.phase4_with_births]]
  .forEach(([n,r])=>{h+='<tr><td>'+n+'</td><td>'+f(r.precision)+'</td><td>'+f(r.recall)+'</td><td>'+f(r.f1)+'</td><td>'+r.fn+'</td><td>'+r.fp+'</td></tr>';});
 h+='</table>'+row('recalled by 4A (GT-moving gained)',(e.phase3_support.fn-e.phase4_originals.fn)+' Gaussians')+
  (e.births_gt_moving_pct!=null?row('4C births on GT-moving surface',e.births_gt_moving_pct.toFixed(1)+' %',e.births_gt_moving_pct<90):'');
 document.getElementById('ev').innerHTML=h;}
else document.getElementById('ev').innerHTML='<div class="note">needs GT part labels</div>';
{const r=D.relabelled;document.getElementById('rev').innerHTML=r?
 row('support Gaussians relabelled STATIC',r.n,true)+(r.gt_static_pct!=null?row('of which GT static',r.gt_static_pct.toFixed(1)+' %',r.gt_static_pct<90):'')+
 '<div class="note">Orange layer: in the "Phase 3" tab they slide with the part (what Phase 3 did); in the "Phase 4" tab they stay put.</div>'
 :row('none','the whole Phase-3 support was kept');}
{let h='';D.rows.filter(r=>[3,4,5,6,7].includes(r.k)).forEach(r=>{
  h+=row(D.labels[r.k],r.n+' ('+r.moves_p4+' drawn moving'+(D.gt&&r.gt_moving_pct!=null?', GT moving '+r.gt_moving_pct.toFixed(0)+'%':'')+')',true);});
 document.getElementById('unr').innerHTML=(h||row('none',''))+'<div class="note">Unresolved originals are rendered with a '+
  'provisional, frozen label (that is not a decision). Turn off the other layers and scrub the slider: an unresolved point '+
  'that sits on the moving part but stays put in the "Phase 4" tab is one 4A could not decide.</div>';}
{let h=row('4A',D.stages['4a'],D.stages['4a']!=='complete')+row('4B',D.stages['4b'],D.stages['4b']!=='complete')+
  row('4C',D.stages['4c'],D.stages['4c']!=='complete');
 if(D.b){h+=row('4B candidates',D.b.candidates)+row('sigma_model',(1000*D.b.sigma_model_m).toFixed(2)+' mm');
  if(D.b.recovery_views)h+=row('4B/4C views',D.b.recovery_views.views+' (frame stride '+D.b.recovery_views.frame_stride+')');}
 if(D.c){h+=row('4C accepted births',D.c.births)+row('4C cycles',D.c.cycles);
  for(const k in (D.c.counts||{}))h+=row('&nbsp;&nbsp;'+k,D.c.counts[k]);}
 document.getElementById('bc').innerHTML=h;}
function setMode(m){mode=m;document.getElementById('tab3').className='tab'+(m==='p3'?' on':'');
 document.getElementById('tab4').className='tab'+(m==='p4'?' on':'');
 document.getElementById('modenote').textContent=m==='p3'
  ?'Only the Phase-3 support moves with the joint. Recalled points (green) stay behind -- that is the motion Phase 3 left out.'
  :'Everything Phase 4 renders as moving follows the fixed joint: core, 4A-recalled, 4C births, provisional unresolved.';
 setFrame(+sl.value);}
const sl=document.getElementById('sl');sl.max=D.frames.length-1;
sl.addEventListener('input',e=>setFrame(+e.target.value));
document.getElementById('tab3').onclick=()=>setMode('p3');
document.getElementById('tab4').onclick=()=>setMode('p4');
let timer=null;
document.getElementById('play').addEventListener('click',e=>{
 if(timer){clearInterval(timer);timer=null;e.target.textContent='play';return;}
 e.target.textContent='stop';
 timer=setInterval(()=>{sl.value=(+sl.value+1)%D.frames.length;setFrame(+sl.value);},110);});
sl.value=D.frames.length-1;setMode('p4');
(function loop(){requestAnimationFrame(loop);RE.render(S,CAM);})();
</script></body></html>"""


def write_page(run: Path, out=None) -> Path:
    d = build_payload(run)
    html = SHELL.replace("__TITLE__", d["title"]).replace("__PAYLOAD__", json.dumps(d, separators=(",", ":")))
    p = Path(out) if out else run / "html" / "phase4_membership_debug.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".html.tmp")
    tmp.write_text(html)
    tmp.replace(p)
    ev = d["eval"]
    msg = f"{p}  ({p.stat().st_size / 1e6:.1f} MB, {d['n']} Gaussians, {len(d['frames'])} frames)"
    if ev:
        msg += (f"\n    recall vs GT: Phase 3 support {ev['phase3_support']['recall']:.3f} -> Phase 4 "
                f"{ev['phase4_originals']['recall']:.3f} (precision {ev['phase4_originals']['precision']:.3f})")
    print(msg, flush=True)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase4-dir", action="append", required=True, help="Phase-4 run directory; repeatable")
    a = ap.parse_args(argv)
    rc = 0
    for r in a.phase4_dir:
        r = Path(r)
        if not (r / "4a_original_membership" / "membership.npz").is_file():
            print(f"{r}: no finished 4A (membership.npz) -- skipped")
            rc = 1
            continue
        write_page(r)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
