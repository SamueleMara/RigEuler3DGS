"""Section 45-49: the one synchronized debug page.

CODEX_DEEP_STAGEWISE_DEBUG_AND_THEORY_AUDIT.md section 45 is explicit that
there must be ONE authoritative file, not a scatter of unrelated HTMLs, and
that its regions move together on a single time cursor:

  Region A  the 3D physical state at frame t
  Region B  fixed-camera image evidence
  Region C  timelines
  Region D  the theory / status panel

Region B is deliberately empty here and says so. The dataset images and the
rasterizer are not in the cached tensors, and section 58 requires naming the
quantity that cannot be reconstructed rather than drawing something that looks
like evidence.

Section 24's rule is enforced in Region D: an axis is drawn as physically
meaningful only when its angular vector is observable, and otherwise carries
`AXIS_COSINE_NOT_INTERPRETABLE`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def _ser(t, nd=5):
    return [[round(float(v), nd) for v in row]
            for row in torch.as_tensor(t).double().reshape(-1, 3)]


def _load_render_manifests(renders: Path) -> dict:
    """Section 47: what the render pass actually wrote, read off disk.

    The page REFERENCES the PNGs rather than embedding them. They are the
    deliverable in their own right -- openable, diffable, usable outside this
    page -- and a 60-frame seven-stage sequence base64'd into one HTML would be
    hundreds of megabytes.
    """
    out: dict = {"cameras": [], "frames": {}, "unavailable": {}, "strips": []}
    if not renders.is_dir():
        return out
    for cam in sorted(renders.glob("camera_*")):
        man = cam / "sequence_manifest.json"
        if not man.is_file():
            continue
        try:
            m = json.loads(man.read_text())
        except Exception:
            continue
        cid = int(m.get("camera", 0))
        out["cameras"].append(cid)
        out["unavailable"] = m.get("unavailable_stages", out["unavailable"])
        for fr in m.get("frames", []):
            t = int(fr["frame"])
            rel = f"../renders/{cam.name}/frame_{t:04d}"
            imgs = {k: f"{rel}/{v}" for k, v in fr.get("images", {}).items()
                    if isinstance(v, str)}
            scal = {k: v for k, v in fr.get("images", {}).items()
                    if not isinstance(v, str)}
            out["frames"].setdefault(str(cid), {})[str(t)] = {
                "images": imgs, "scalars": scal,
                "unavailable": fr.get("unavailable", {})}
    sm = renders / "strips_manifest.json"
    if sm.is_file():
        try:
            out["strips"] = [f"../renders/{p}" for p in
                             json.loads(sm.read_text()).get("strips", [])]
        except Exception:
            pass
    return out


def write_joint_debug_html(outdir: Path, scene: str, ph1: dict, evidence: dict,
                           candidate: dict, three_way: dict, p2_omega,
                           stages: dict, matrix: list, dt: float) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    X = ph1["tracks"]
    T, N = int(X.shape[0]), int(X.shape[1])
    step = max(1, N // 400)                      # display downsample only
    idx = list(range(0, N, step))

    frames = [_ser(X[t][idx], 4) for t in range(T)]
    vel = evidence["velocity"]["velocity"]
    vels = [_ser(vel[t][idx], 5) for t in range(T)]

    om_d = evidence["twist"]["omega"]
    om_dm = 0.5 * (om_d[1:] + om_d[:-1])
    fin = three_way["finite_p1"]["omega"]
    p2 = (p2_omega if p2_omega is not None
          else torch.zeros((0, 3), dtype=torch.float64))
    strain = evidence["strain_statistic"]
    chi = evidence["rigidity_chi2"]
    phi = evidence["finite"]["phi"]

    data = {
        "scene": scene, "T": T, "N": N, "shown": len(idx), "dt": dt,
        "frames": frames, "velocities": vels,
        "omega_diff_p1": _ser(om_dm, 6),
        "omega_finite_p1": _ser(fin, 6),
        "omega_p2": _ser(p2, 6),
        "phi": _ser(phi, 6),
        "strain_Q_D": [round(float(v), 6) for v in strain],
        "chi2_rigid_vs_affine": [round(float(v), 6) for v in chi],
        "three_way": {k: (v if not isinstance(v, float) else round(v, 6))
                      for k, v in three_way["summary"].items()},
        "candidate": {"status": candidate["status"],
                      "candidate": candidate["candidate"],
                      "reason": candidate["reason"],
                      "stage_a": candidate["stage_a_rigid"],
                      "stage_b": candidate["stage_b_one_dof"],
                      "stage_c": candidate["stage_c_angular"]},
        "stages": {k: {"status": v.get("status"),
                       "note": v.get("answers", v.get("reason", ""))}
                   for k, v in stages.items()},
        "matrix": [{"key": r["key"], "status": r["status"],
                    "batch": r["runs_in_standard_batch"],
                    "requirement": r["requirement"]} for r in matrix],
        "renders": _load_render_manifests(outdir.parent / "renders"),
    }

    html = _TEMPLATE.replace("__DATA__", json.dumps(data))
    p = outdir / "joint_debug.html"
    p.write_text(html)
    return p


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stagewise Joint Debug</title>
<style>
:root{--bg:#11131a;--fg:#e8eaf0;--mut:#8b93a7;--line:#262a36;--card:#171a23;
--ok:#4ad28a;--bad:#ff6b6b;--warn:#ffc857;--a:#6aa9ff;--b:#c99bff;--c:#ff9f68;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:12px 16px;border-bottom:1px solid var(--line);
display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600}
.mut{color:var(--mut)}
#grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}
@media(max-width:900px){#grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px;min-width:0}
.card h2{font-size:12px;margin:0 0 8px;color:var(--mut);
text-transform:uppercase;letter-spacing:.08em}
canvas{width:100%;height:auto;display:block;background:#0d0f15;border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:12px}
td,th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left;
vertical-align:top}
th{color:var(--mut);font-weight:500}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px}
.CONSISTENT,.MATCHES_THEORY{background:#14301f;color:var(--ok)}
.INCONSISTENT,.BUG{background:#331617;color:var(--bad)}
.LOW_SIGNAL,.UNOBSERVABLE,.UNCALIBRATED,.APPROXIMATE,.UNVERIFIED
{background:#33290f;color:var(--warn)}
.NOT_AVAILABLE,.NOT_WIRED,.MISSING,.DEBUG_ONLY,.DIFFERENT_STATISTIC,.CACHED
{background:#1d2130;color:var(--mut)}
input[type=range]{width:260px;vertical-align:middle}
.k{color:var(--mut)}
.val{color:var(--fg)}
.legend span{margin-right:12px}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;
margin-right:4px;vertical-align:middle}
.empty{padding:22px;text-align:center;color:var(--mut);border:1px dashed
var(--line);border-radius:6px;font-size:12px}
</style></head><body>
<header>
  <h1 id="title">Stagewise Joint Debug</h1>
  <span class="mut" id="meta"></span>
  <span>frame <b id="fnum">0</b></span>
  <input type="range" id="slider" min="0" value="0">
  <button id="play">play</button>
</header>
<div id="grid">
  <section class="card"><h2>Region A — 3D physical state</h2>
    <canvas id="scene" width="760" height="460"></canvas>
    <div class="legend mut" style="margin-top:6px">
      <span><i class="sw" style="background:var(--a)"></i>supernode (frame t)</span>
      <span><i class="sw" style="background:var(--c)"></i>free velocity &times;8</span>
      <span><i class="sw" style="background:var(--mut)"></i>frame-0 position</span>
    </div>
  </section>
  <section class="card"><h2>Region B — fixed-camera image evidence</h2>
    <div class="empty" id="regionB"></div>
  </section>
  <section class="card"><h2>Region C — timelines</h2>
    <canvas id="tl" width="760" height="460"></canvas>
    <div class="legend mut" style="margin-top:6px">
      <span><i class="sw" style="background:var(--a)"></i>|&omega;| diff-P1</span>
      <span><i class="sw" style="background:var(--b)"></i>|&omega;| finite-P1</span>
      <span><i class="sw" style="background:var(--c)"></i>|&omega;| P2</span>
      <span><i class="sw" style="background:var(--warn)"></i>&chi;&sup2; rigid-vs-affine (log)</span>
    </div>
  </section>
  <section class="card"><h2>Region D — theory / status</h2>
    <div id="panel"></div>
  </section>
</div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
$("#title").textContent = "Stagewise Joint Debug — " + D.scene;
$("#meta").textContent = `${D.N} supernodes (showing ${D.shown}) · ${D.T} frames · dt=${D.dt}`;
const slider = $("#slider"); slider.max = D.T - 1;

// ---- projection: fixed axes across all frames so motion is visible -------
const all = D.frames.flat();
const ax = [0,1], lo=[1e18,1e18], hi=[-1e18,-1e18];
for (const p of all) for (let k=0;k<2;k++){
  lo[k]=Math.min(lo[k],p[ax[k]]); hi[k]=Math.max(hi[k],p[ax[k]]); }
const pad=0.08;
for(let k=0;k<2;k++){const d=(hi[k]-lo[k])||1; lo[k]-=d*pad; hi[k]+=d*pad;}
function proj(p,w,h){
  return [ (p[ax[0]]-lo[0])/((hi[0]-lo[0])||1)*w,
           h - (p[ax[1]]-lo[1])/((hi[1]-lo[1])||1)*h ];
}
function drawScene(t){
  const c=$("#scene"), g=c.getContext("2d"), w=c.width, h=c.height;
  g.clearRect(0,0,w,h);
  g.fillStyle="#555b6e";
  for(const p of D.frames[0]){const [x,y]=proj(p,w,h);
    g.fillRect(x-1,y-1,2,2);}
  const F=D.frames[t], V=D.velocities[t];
  g.strokeStyle="#ff9f68"; g.lineWidth=.7; g.beginPath();
  for(let i=0;i<F.length;i++){
    const [x,y]=proj(F[i],w,h);
    const [x2,y2]=proj([F[i][0]+V[i][0]*8,F[i][1]+V[i][1]*8,F[i][2]+V[i][2]*8],w,h);
    g.moveTo(x,y); g.lineTo(x2,y2);}
  g.stroke();
  g.fillStyle="#6aa9ff";
  for(const p of F){const [x,y]=proj(p,w,h); g.beginPath();
    g.arc(x,y,1.9,0,6.283); g.fill();}
}
// ---- timelines ----------------------------------------------------------
const n3=a=>a.map(v=>Math.hypot(v[0],v[1],v[2]));
const S={d:n3(D.omega_diff_p1), f:n3(D.omega_finite_p1), p:n3(D.omega_p2),
         c:D.chi2_rigid_vs_affine.map(v=>Math.log10(Math.max(v,1e-12)))};
function drawTl(t){
  const c=$("#tl"), g=c.getContext("2d"), w=c.width, h=c.height;
  g.clearRect(0,0,w,h);
  const top=h*0.60, bot=h-28;
  const mx=Math.max(1e-12,...S.d,...S.f,...S.p);
  const line=(arr,col,y0,y1,lo,hi)=>{
    if(!arr.length)return; g.strokeStyle=col; g.lineWidth=1.4; g.beginPath();
    arr.forEach((v,i)=>{const x=i/(D.T-1)*w,
      y=y1-(v-lo)/((hi-lo)||1)*(y1-y0);
      i?g.lineTo(x,y):g.moveTo(x,y);}); g.stroke();};
  g.strokeStyle="#262a36"; g.beginPath();
  g.moveTo(0,top);g.lineTo(w,top); g.stroke();
  line(S.d,"#6aa9ff",10,top-8,0,mx);
  line(S.f,"#c99bff",10,top-8,0,mx);
  line(S.p,"#ff9f68",10,top-8,0,mx);
  const cl=Math.min(...S.c), ch=Math.max(...S.c);
  line(S.c,"#ffc857",top+10,bot,cl,ch);
  g.fillStyle="#8b93a7"; g.font="11px ui-monospace";
  g.fillText("angular velocity  max "+mx.toExponential(2),6,12);
  g.fillText("log10 chi2 rigid-vs-affine  ["+cl.toFixed(1)+", "+ch.toFixed(1)+"]",6,top+22);
  const x=t/(D.T-1)*w;
  g.strokeStyle="#e8eaf0"; g.lineWidth=1; g.beginPath();
  g.moveTo(x,0); g.lineTo(x,bot); g.stroke();
}
// ---- Region B / D -------------------------------------------------------
// ---- Region B: the rendered sequence, read from renders/ on disk --------
const RB = D.renders || {cameras:[], frames:{}, unavailable:{}, strips:[]};
let cam = (RB.cameras && RB.cameras.length) ? RB.cameras[0] : null;
let showResidual = false;
function nearestRenderedFrame(t){
  const avail = Object.keys((RB.frames||{})[String(cam)]||{}).map(Number);
  if(!avail.length) return null;
  return avail.reduce((a,b)=>Math.abs(b-t)<Math.abs(a-t)?b:a);
}
function regionB(t){
  const el=$("#regionB");
  if(cam===null){
    el.className="empty";
    el.innerHTML="No renders on disk for this run.<br><br>"+
      "Produce them with:<br><code>python tools/deep_debug_render.py --run &lt;run&gt;</code>"+
      "<br><br>They need the trained model, the dataset images and a GPU, which "+
      "is why they are not reconstructible from cached tensors (&sect;58).";
    return;
  }
  el.className="";
  const ft=nearestRenderedFrame(t);
  const rec=((RB.frames||{})[String(cam)]||{})[String(ft)]||{images:{},scalars:{},unavailable:{}};
  const order=["gt","canonical","phase1_free","phase2_rigid","P","R","refined"];
  let cams=RB.cameras.map(c=>`<option value="${c}"${c===cam?" selected":""}>camera ${c}</option>`).join("");
  let h=`<div style="margin-bottom:8px">
    <select id="camsel">${cams}</select>
    <label style="margin-left:10px"><input type="checkbox" id="resid"${showResidual?" checked":""}> residual</label>
    <span class="mut" style="margin-left:10px">rendered frame ${ft}${ft!==t?` (nearest to ${t})`:""} &middot; same camera, same crop, one depth scale for the whole sequence (&sect;49)</span>
  </div><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px">`;
  for(const st of order){
    const key = (showResidual && st!=="gt") ? st+"_rgb_residual" : st+"_rgb";
    const src = rec.images[key];
    const l1 = rec.scalars[st+"_rgb_l1"];
    if(src){
      h+=`<figure style="margin:0"><img src="${src}" style="width:100%;border-radius:4px;background:#0d0f15" loading="lazy">
      <figcaption class="mut" style="font-size:11px;margin-top:2px">${st}${l1!==undefined?` &middot; L1 ${(+l1).toExponential(2)}`:""}</figcaption></figure>`;
    } else {
      const why = rec.unavailable[st] || (RB.unavailable||{})[st] || "not rendered";
      h+=`<figure style="margin:0"><div class="empty" style="padding:14px;font-size:11px">${st}<br><span style="font-size:10px">${why}</span></div></figure>`;
    }
  }
  h+="</div>";
  const dl = order.filter(st=>rec.scalars[st+"_depth_l1"]!==undefined && rec.scalars[st+"_depth_l1"]!==null);
  if(dl.length){
    h+='<table style="margin-top:8px"><tr><th>stage</th><th>RGB L1</th><th>depth L1 (m)</th></tr>';
    for(const st of dl) h+=`<tr><td>${st}</td><td>${(+rec.scalars[st+"_rgb_l1"]).toExponential(3)}</td><td>${(+rec.scalars[st+"_depth_l1"]).toFixed(5)}</td></tr>`;
    h+="</table>";
  }
  if(RB.strips && RB.strips.length)
    h+=`<div class="mut" style="margin-top:8px">key-frame strips (&sect;48): `+
       RB.strips.map((p,i)=>`<a href="${p}" style="color:var(--a)">sheet ${i+1}</a>`).join(" &middot; ")+"</div>";
  el.innerHTML=h;
  const cs=$("#camsel"); if(cs) cs.onchange=e=>{cam=+e.target.value; regionB(t);};
  const rr=$("#resid"); if(rr) rr.onchange=e=>{showResidual=e.target.checked; regionB(t);};
}

function tag(s){return `<span class="tag ${s}">${s}</span>`;}
function row(k,v){return `<tr><td class="k">${k}</td><td class="val">${v}</td></tr>`;}
function panel(t){
  const tw=D.three_way, C=D.candidate;
  const interp = tw.axis_cosine_interpretable===false
    ? `<div style="color:var(--warn);margin:6px 0">AXIS_COSINE_NOT_INTERPRETABLE — at least one angular vector is unobservable, so the cosines below carry no physical meaning (section 24).</div>` : "";
  let st="<table>";
  for(const [k,v] of Object.entries(D.stages))
    st+=row(k, tag(v.status)+" <span class='mut'>"+(v.note||"")+"</span>");
  st+="</table>";
  const f=(x)=>x===""||x===null||x===undefined?"<span class=mut>n/a</span>":x;
  let three="<table>"+
    row("|&omega;| diff-P1 @ frame", f((S.d[t]??"").toExponential?S.d[t].toExponential(3):""))+
    row("|&omega;| finite-P1 @ frame", f(S.f[t]!==undefined?S.f[t].toExponential(3):""))+
    row("|&omega;| P2 @ frame", f(S.p[t]!==undefined?S.p[t].toExponential(3):""))+
    row("median cos(diff,finite)", f(tw.median_cos_diff_vs_finite_p1))+
    row("median cos(diff,P2)", f(tw.median_cos_diff_p1_vs_p2))+
    row("median cos(finite,P2)", f(tw.median_cos_finite_p1_vs_p2))+
    row("Mahalanobis(diff,finite)", f(tw.median_mahal_diff_vs_finite_p1)+
        " vs chi2(3) cut "+f(tw.chi2_3dof_cut))+
    row("diff-P1 observable", (tw.fraction_diff_observable*100).toFixed(1)+"% of frames")+
    row("finite-P1 observable", (tw.fraction_finite_p1_observable*100).toFixed(1)+"% of frames")+
    row("P2 covariance", "<span class=mut>not available — no Mahalanobis fabricated</span>")+
    "</table>";
  let cand="<table>"+
    row("status", tag(C.status)+" "+(C.candidate||""))+
    row("A rigid", (C.stage_a.passes?"pass":"<b style='color:var(--bad)'>FAIL</b>")+
        ` &chi;&sup2;=${(+C.stage_a.median_chi2_rigid_vs_affine).toExponential(3)} cut ${(+C.stage_a.cut).toFixed(2)}`)+
    row("B one-DoF", (C.stage_b.passes?"pass":"<b style='color:var(--bad)'>FAIL</b>")+
        ` ${(+C.stage_b.per_dof).toExponential(3)} per dof`)+
    row("C angular", `&omega;&ne;0 ${C.stage_c.angular_present} · one axis ${C.stage_c.angular_shares_one_axis}`)+
    row("reason", "<span class=mut>"+C.reason+"</span>")+
    "</table>";
  let mtx="<table><tr><th>requirement</th><th>batch</th><th>status</th></tr>";
  for(const r of D.matrix)
    mtx+=`<tr><td>${r.key}</td><td>${r.batch?"yes":"<span class=mut>no</span>"}</td><td>${tag(r.status)}</td></tr>`;
  mtx+="</table>";
  $("#panel").innerHTML =
    "<h2 style='margin-top:0'>stage status</h2>"+st+
    "<h2>three-way angular (&sect;21-23)</h2>"+interp+three+
    `<div class="mut" style="margin:6px 0">${tw.interpretation||""}</div>`+
    "<h2>geometric candidate (&sect;35)</h2>"+cand+
    "<h2>theory matrix (&sect;4)</h2>"+mtx;
}
let t=0, timer=null;
function render(){ $("#fnum").textContent=t; slider.value=t;
  drawScene(t); drawTl(t); panel(t); regionB(t); }
slider.oninput=e=>{t=+e.target.value; render();};
$("#play").onclick=()=>{ if(timer){clearInterval(timer);timer=null;$("#play").textContent="play";}
  else{ $("#play").textContent="pause";
    timer=setInterval(()=>{t=(t+1)%D.T; render();},90);} };
render();
</script></body></html>
"""
