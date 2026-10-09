"""The interactive HTML debugger.

CODEX_ADD_VISUAL_AND_NUMERICAL_DEBUG_GEOMETRIC_PIPELINE.md sections 1-12,
39, 42-46.

Produces `geometric_motion_debug.html`: one self-contained file with a 3D
scene, a time slider, a stage selector and a live numerical dashboard.

On section 0 rule 3 ("do not regenerate an approximate motion only for
visualization"): the page is given the estimator's OWN state -- the canonical
Gaussians, the recovered {T_{k,t}}, the fitted joint parameters and the
recovered q(t) -- and applies section 2's three transport laws

    rigid      x = R_{k,t} x^0 + t_{k,t}
    prismatic  x = x^0 + q_{k,t} e_k
    revolute   x = a_k + R(e_k, q_{k,t}) (x^0 - a_k)

verbatim. Those are the same formulas and the same numbers the estimator used,
so nothing is re-derived or approximated; sending transforms rather than
baked-out point clouds also keeps the file small enough to open. The bootstrap
stage has no transform by construction -- it is a free per-Gaussian field --
so its positions are sent directly.

Every quantity shown here is also written to `debug_numeric/`, which is what
section 0 rule 5 requires: the picture can be checked against the numbers
without trusting the picture.
"""

from __future__ import annotations

import json
import math

import torch

from .recorder import PART_COLORS, GeometricDebug, part_color

THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.min.js"


def _subsample(n: int, limit: int, seed: int = 0) -> torch.Tensor:
    if limit <= 0 or n <= limit:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g)[:limit].sort().values


def _quat(R: torch.Tensor) -> list:
    m = R.double()
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [float(m[2, 1] - m[1, 2]) / s, float(m[0, 2] - m[2, 0]) / s,
                float(m[1, 0] - m[0, 1]) / s, 0.25 * s]
    i = int(torch.argmax(torch.tensor([m[0, 0], m[1, 1], m[2, 2]])))
    if i == 0:
        s = math.sqrt(1.0 + float(m[0, 0] - m[1, 1] - m[2, 2])) * 2
        return [0.25 * s, float(m[0, 1] + m[1, 0]) / s,
                float(m[0, 2] + m[2, 0]) / s, float(m[2, 1] - m[1, 2]) / s]
    if i == 1:
        s = math.sqrt(1.0 + float(m[1, 1] - m[0, 0] - m[2, 2])) * 2
        return [float(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                float(m[1, 2] + m[2, 1]) / s, float(m[0, 2] - m[2, 0]) / s]
    s = math.sqrt(1.0 + float(m[2, 2] - m[0, 0] - m[1, 1])) * 2
    return [float(m[0, 2] + m[2, 0]) / s, float(m[1, 2] + m[2, 1]) / s,
            0.25 * s, float(m[1, 0] - m[0, 1]) / s]


def _camera_frustums(dbg: GeometricDebug) -> list:
    """Section 11: world-frame frustums, fixed, with IDs."""
    out = []
    for ci, cam in enumerate(dbg.camera_meta or []):
        V = cam.get("viewmatrix")
        if V is None:
            continue
        V = torch.as_tensor(V, dtype=torch.float64)
        # The project's CameraSpec stores a row-vector view matrix, so the
        # world->camera rotation is V[:3,:3]^T and the centre is -R^T t.
        Rwc = V[:3, :3]
        t = V[3, :3]
        centre = -(Rwc @ t)
        tx = float(cam.get("tanfovx", 0.5))
        ty = float(cam.get("tanfovy", 0.5))
        d = float(cam.get("depth", 0.35))
        corners = []
        for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            p = torch.tensor([sx * tx * d, sy * ty * d, d], dtype=torch.float64)
            corners.append([float(x) for x in (centre + Rwc @ p)])
        out.append({"id": ci, "name": cam.get("name", f"cam_{ci:03d}"),
                    "centre": [float(x) for x in centre], "corners": corners})
    return out


def build_payload(dbg: GeometricDebug, *, max_gaussians: int = 4000,
                  frame_stride: int = 1, glyph_limit: int = 240) -> dict:
    """Everything the page needs, as plain JSON."""
    n = dbg.num_gaussians
    idx = _subsample(n, max_gaussians, seed=0)
    frames = list(range(0, dbg.num_frames, max(1, frame_stride)))

    mu0 = dbg.mu0[idx].detach().cpu()
    labels = (dbg.labels[idx] if dbg.labels is not None
              else torch.zeros(len(idx), dtype=torch.long)).cpu()
    resid = (dbg.rigid_residual[idx].detach().cpu()
             if dbg.rigid_residual is not None else torch.zeros(len(idx)))
    scales = (dbg.scales[idx].detach().cpu() if dbg.scales is not None
              else torch.full((len(idx), 3), 0.02))
    colors = (dbg.colors[idx].detach().cpu() if dbg.colors is not None
              else torch.full((len(idx), 3), 0.7))

    # Section 2: the bootstrap stage is a free per-Gaussian field with no
    # transform, so it is the one stage whose positions must be sent.
    boot = [[round(float(v), 6) for v in
             dbg.positions[t][idx].reshape(-1).detach().cpu().tolist()]
            for t in frames] if dbg.positions else []

    rigid = []
    if dbg.transforms is not None:
        for t in frames:
            per = []
            for k in range(dbg.transforms.K):
                per.append({"R": dbg.transforms.R[t, k].detach().cpu()
                            .reshape(-1).tolist(),
                            "t": dbg.transforms.t[t, k].detach().cpu().tolist()})
            rigid.append(per)

    joints = []
    for k in range(1, dbg.K + 1):
        j = dbg.joints[k - 1] if k - 1 < len(dbg.joints) else None
        if j is None:
            joints.append(None)
            continue
        ne = j.get("normalized_error") or {}
        pr = j.get("prismatic") or {}
        rv = j.get("revolute") or {}
        com = dbg.com_trajectory(k, "rigid")
        joints.append({
            "part": k, "selected": j.get("selected"), "status": j.get("status"),
            "axis": j.get("axis"), "axis_point": j.get("axis_point"),
            # Section 5: the Plucker moment m = a x e, shown alongside e and a.
            "moment": ([float(x) for x in torch.linalg.cross(
                torch.tensor(j["axis_point"], dtype=torch.float64),
                torch.tensor(j["axis"], dtype=torch.float64))]
                if j.get("axis_point") and j.get("axis") else None),
            "q": (dbg.q_traj[k - 1] if k - 1 < len(dbg.q_traj)
                  else j.get("q")),
            "q_geometric": j.get("q"),
            "E_P": ne.get("prismatic"), "E_R": ne.get("revolute"),
            "floor": ne.get("indistinguishable_floor"),
            "gap": ne.get("absolute_gap"),
            # Section 7: BOTH candidates are always carried, so an unresolved
            # run can draw them side by side instead of silently picking one.
            "prismatic_axis": pr.get("axis"),
            "prismatic_rank1": (pr.get("diagnostics") or {}).get(
                "translation_rank1_ratio"),
            "prismatic_rotation_residual": (pr.get("diagnostics") or {}).get(
                "max_body_rotation_rad"),
            "revolute_axis": rv.get("axis"),
            "revolute_axis_point": rv.get("axis_point"),
            "revolute_gauge": (rv.get("diagnostics") or {}).get("gauge_e_dot_a"),
            "axis_iterates": rv.get("axis_iterates"),
            "com": com,
            "length_scale": j.get("length_scale"),
            "rigid_rmse_median": j.get("rigid_rmse_median"),
        })

    # Section 3: orientation glyphs for a configurable subset.
    gl = _subsample(len(idx), glyph_limit, seed=1)
    glyph = {"index": gl.tolist(),
             "quat0": ([_quat(dbg.R0[idx][int(i)]) for i in gl]
                       if dbg.R0 is not None else []),
             "scale": [scales[int(i)].tolist() for i in gl]}

    # Section 18/45: the supernode table.
    sn = []
    if dbg.supernode_id is not None and dbg.mu0 is not None:
        sid = dbg.supernode_id.long().clamp_min(0)
        C = dbg.num_supernodes
        cnt = torch.zeros(C, device=sid.device).index_add_(
            0, sid, torch.ones(sid.shape[0], device=sid.device))
        c0 = torch.zeros((C, 3), device=sid.device).index_add_(
            0, sid, dbg.mu0) / cnt.clamp_min(1).unsqueeze(1)
        costs = dbg.assignment_costs
        for c in range(C):
            row = {"id": c, "members": int(cnt[c]),
                   "centroid": [float(x) for x in c0[c]],
                   "label": int(dbg.node_labels[c]) if dbg.node_labels is not None else 0,
                   "evidence": (float(dbg.motion_statistic[c])
                                if dbg.motion_statistic is not None
                                and c < dbg.motion_statistic.numel() else None)}
            if costs is not None and c < costs.shape[0]:
                v = [float(x) for x in costs[c]]
                s = sorted(v)
                row["costs"] = v
                row["margin"] = (s[1] - s[0]) if len(s) > 1 else None
            sn.append(row)

    from tools.iekf_artifacts import jsonable
    return {
        "meta": {
            "model": dbg.model, "dataset": dbg.dataset, "K": dbg.K,
            "status": dbg.status,
            "num_gaussians": n, "shown_gaussians": int(len(idx)),
            "num_supernodes": dbg.num_supernodes,
            "frames_total": dbg.num_frames, "frame_stride": max(1, frame_stride),
            "frame_ids": [dbg.frame_ids[t] if t < len(dbg.frame_ids) else t
                          for t in frames],
            "dt": dbg.dt,
            "image_size": list(dbg.image_size),
            "rigid_converged": dbg.rigid_converged,
            "rigid_stop_reason": dbg.rigid_stop_reason,
            "part_colors": PART_COLORS[:max(dbg.K + 1, 2)],
            "stages_available": {s: dbg.stage_available(s) for s in
                                 ("canonical", "bootstrap", "rigid", "joint",
                                  "refined")},
            "gt_overlay": False,
        },
        "x0": [round(float(v), 6) for v in mu0.reshape(-1).tolist()],
        "labels": labels.tolist(),
        "rigid_residual": [round(float(v), 8) for v in resid.tolist()],
        "supernode_of": ((dbg.supernode_id[idx].long().clamp_min(0)
                          .cpu().tolist())
                         if dbg.supernode_id is not None else []),
        "evidence_by_supernode": ([float(x) for x in dbg.motion_statistic]
                                  if dbg.motion_statistic is not None else []),
        "scales": [round(float(v), 6) for v in scales.reshape(-1).tolist()],
        "colors": [round(float(v), 4) for v in colors.reshape(-1).tolist()],
        "glyph": glyph,
        "bootstrap": boot,
        "rigid": rigid,
        "joints": joints,
        "direct_estimators": jsonable(dbg.direct_joint_estimates),
        "direct_heat": {name: values[idx].tolist() for name,values in dbg.direct_joint_heat.items()},
        "direct_frame_indices": frames,
        "cameras": _camera_frustums(dbg),
        "supernodes": sn,
        "frame_rows": dbg.frame_rows,
        "camera_rows": dbg.camera_rows,
        "energy_by_stage": dbg.energy_by_stage,
        "observability": dbg.observability,
        "reduced": {k: v for k, v in (dbg.reduced or {}).items()
                    if k not in ("per_frame", "frames")},
        "notes": dbg.notes,
        # Section 12: the panels are written beside the page and linked
        # from it rather than embedded, which that section allows.
        "render_panels": dbg.render_panels,
    }


def write_html(dbg: GeometricDebug, writer, *, max_gaussians: int = 4000,
               frame_stride: int = 1) -> str:
    """Write `geometric_motion_debug.html` and return its path."""
    payload = build_payload(dbg, max_gaussians=max_gaussians,
                            frame_stride=frame_stride)
    html = _TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    html = html.replace("__THREE__", THREE_CDN)
    p = writer.dir / "geometric_motion_debug.html"
    p.write_text(html)
    writer.written.append(p.name)
    writer.json("geometric_motion_debug_payload", payload)
    return str(p)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Geometric Motion Debug</title>
<style>
  :root{--bg:#0e1116;--panel:#161b22;--line:#2a313a;--fg:#e6edf3;--dim:#8b949e;
        --accent:#58a6ff;--warn:#f0883e;--bad:#f85149;--good:#3fb950;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;overflow:hidden}
  #app{display:flex;height:100vh;width:100vw}
  #view{flex:1;position:relative;min-width:0}
  canvas{display:block}
  #side{width:355px;flex:none;background:var(--panel);border-left:1px solid var(--line);
        overflow-y:auto;padding:12px}
  @media (max-width:900px){#app{flex-direction:column}#side{width:100%;height:46vh;
        border-left:none;border-top:1px solid var(--line)}}
  h1{font-size:14px;margin:0 0 4px}
  h2{font-size:12px;margin:14px 0 6px;color:var(--accent);
     text-transform:uppercase;letter-spacing:.06em}
  .row{display:flex;justify-content:space-between;gap:8px;padding:1px 0}
  .row span:last-child{color:var(--dim);text-align:right;word-break:break-all}
  .chip{display:inline-block;padding:1px 6px;border-radius:9px;font-size:11px}
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
       pointer-events:none;font-size:12px}
  table{width:100%;border-collapse:collapse;font-size:11px}
  td,th{padding:2px 4px;border-bottom:1px solid var(--line);text-align:left}
  tr.sel{background:#1f6feb33}
  tbody tr{cursor:pointer}
  .sw{display:inline-block;width:9px;height:9px;border-radius:2px;
      margin-right:4px;vertical-align:middle}
  .warn{color:var(--warn)} .bad{color:var(--bad)} .good{color:var(--good)}
  .note{color:var(--dim);font-size:11px;margin:4px 0}
</style>
</head>
<body>
<div id="app">
  <div id="view">
    <div id="hud"></div>
    <div id="bar">
      <button id="play">play</button>
      <button id="first">|&lt;</button>
      <button id="last">&gt;|</button>
      <input id="slider" type="range" min="0" value="0" step="1">
      <span id="fnum"></span>
      <label>speed <select id="speed">
        <option>0.25</option><option>0.5</option><option selected>1</option>
        <option>2</option><option>4</option></select></label>
      <label><input type="checkbox" id="loop" checked> loop</label>
    </div>
  </div>
  <div id="side">
    <h1>Geometric motion debug</h1>
    <div class="note" id="meta"></div>

    <h2>Stage</h2>
    <select id="stage"></select>
    <div class="note">Section 43: pick a ghost stage to overlay.</div>
    <select id="ghost"></select>
    <h2>Joint estimator</h2>
    <select id="estimator"><option value="SE3">SE3 trajectory</option>
      <option value="Gaussian">Direct Gaussian tracks</option>
      <option value="RGBD">Direct RGB-D</option></select>
    <select id="jointcolor"><option value="none">joint color: off</option>
      <option value="information">joint information</option><option value="support">R-over-P support</option>
      <option value="disagreement">camera disagreement</option><option value="leverage">leverage</option>
      <option value="visibility">visibility/alpha</option></select>

    <h2>Show</h2>
    <div id="toggles"></div>

    <h2>Dashboard</h2>
    <div id="dash"></div>

    <h2>Parts</h2>
    <div id="parts"></div>

    <h2>Supernodes</h2>
    <select id="snfilter">
      <option value="all">all</option><option value="moving">moving only</option>
    </select>
    <div style="max-height:190px;overflow-y:auto"><table id="sntab">
      <thead><tr><th>id</th><th>lab</th><th>n</th><th>evid</th><th>margin</th></tr></thead>
      <tbody></tbody></table></div>
    <div id="snsel" class="note"></div>

    <h2>Image panels</h2>
    <div id="panels" class="note"></div>

    <h2>Cameras</h2>
    <select id="cam"><option value="-1">all</option></select>
    <div id="caminfo" class="note"></div>
  </div>
</div>
<script src="__THREE__"></script>
<script>
const D = __PAYLOAD__;
const M = D.meta, NF = M.frame_ids.length, K = M.K;
const NG = M.shown_gaussians;
const STAGES = [["canonical","Canonical"],["bootstrap","Bootstrap"],
                ["rigid","Rigid parts"],["joint","Geometric joint"],
                ["refined","Renderer-refined joint"],["reduced","Reduced filter"]];
if((D.direct_estimators||[]).length) STAGES.push(["direct","Direct joint estimator"]);

/* ---------- section 2: the estimator's own transports, verbatim ---------- */
function rotAxis(e, q){           // Rodrigues, same as Exp(q [e]_x)
  const c=Math.cos(q), s=Math.sin(q), t=1-c, [x,y,z]=e;
  return [t*x*x+c, t*x*y-s*z, t*x*z+s*y,
          t*x*y+s*z, t*y*y+c, t*y*z-s*x,
          t*x*z-s*y, t*y*z+s*x, t*z*z+c];
}
function mul3(R,v){return [R[0]*v[0]+R[1]*v[1]+R[2]*v[2],
                           R[3]*v[0]+R[4]*v[1]+R[5]*v[2],
                           R[6]*v[0]+R[7]*v[1]+R[8]*v[2]];}
function positions(stage, f){
  const out = new Float32Array(NG*3);
  if(stage==="direct"){
    for(let i=0;i<NG;i++){
      const k=D.labels[i],E=(D.direct_estimators||[])[k-1];
      const J=E?E[directEstimator]:null,x=[D.x0[3*i],D.x0[3*i+1],D.x0[3*i+2]];
      let p=x;
      if(J&&J.s&&J.c){
        const t=(D.direct_frame_indices||[])[f]||0,s=J.s[t]||0;
        if(J.finite_axis_fallback){const a=J.axis_point,R=rotAxis(J.axis,J.q[t]||0);
          p=mul3(R,x.map((v,j)=>v-a[j])).map((v,j)=>v+a[j]);}
        else{
          const eta=J.eta,d=J.d,c=J.c,th=Math.hypot(...eta)*Math.abs(s);
          const phi=eta.map(v=>v*s),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
          const B=th<.01?.5-th*th/24+th**4/720:(1-Math.cos(th))/(th*th);
          const C=th<.01?1/6-th*th/120+th**4/5040:(th-Math.sin(th))/(th**3);
          const w=cross(phi,d),ww=cross(phi,w),v=d.map((a,j)=>s*(a+B*w[j]+C*ww[j]));
          const norm=Math.hypot(...eta),R=rotAxis(norm?eta.map(a=>a/norm):[1,0,0],norm*s);
          p=mul3(R,x.map((a,j)=>a-c[j])).map((a,j)=>a+c[j]+v[j]);
        }
      }
      out.set(p,3*i);
    }return out;
  }
  if(stage==="canonical"){ for(let i=0;i<NG*3;i++) out[i]=D.x0[i]; return out; }
  if(stage==="bootstrap"){
    const b=D.bootstrap[f]; if(!b) return positions("canonical",f);
    for(let i=0;i<NG*3;i++) out[i]=b[i]; return out;
  }
  if(stage==="rigid"){
    const T=D.rigid[f]; if(!T) return positions("canonical",f);
    for(let i=0;i<NG;i++){
      const k=D.labels[i];
      const x=[D.x0[3*i],D.x0[3*i+1],D.x0[3*i+2]];
      if(k===0){out[3*i]=x[0];out[3*i+1]=x[1];out[3*i+2]=x[2];continue;}
      const tr=T[k-1]; const p=mul3(tr.R,x);           // x = R x0 + t
      out[3*i]=p[0]+tr.t[0];out[3*i+1]=p[1]+tr.t[1];out[3*i+2]=p[2]+tr.t[2];
    }
    return out;
  }
  // joint / refined / reduced all use the discovered joint model
  for(let i=0;i<NG;i++){
    const k=D.labels[i];
    const x=[D.x0[3*i],D.x0[3*i+1],D.x0[3*i+2]];
    const J=(k>0&&D.joints[k-1])?D.joints[k-1]:null;
    if(!J||!J.selected||!J.q){out[3*i]=x[0];out[3*i+1]=x[1];out[3*i+2]=x[2];continue;}
    const q=J.q[f]!==undefined?J.q[f]:0;
    if(J.selected==="prismatic"){                      // x = x0 + q e
      out[3*i]=x[0]+q*J.axis[0];out[3*i+1]=x[1]+q*J.axis[1];out[3*i+2]=x[2]+q*J.axis[2];
    }else{                                             // x = a + R(e,q)(x0-a)
      const a=J.axis_point||[0,0,0], R=rotAxis(J.axis,q);
      const d=[x[0]-a[0],x[1]-a[1],x[2]-a[2]], p=mul3(R,d);
      out[3*i]=a[0]+p[0];out[3*i+1]=a[1]+p[1];out[3*i+2]=a[2]+p[2];
    }
  }
  return out;
}

/* ---------------------------- three.js scene ---------------------------- */
const view=document.getElementById("view");
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0e1116);
const cam=new THREE.PerspectiveCamera(50,1,0.01,500);
const renderer=new THREE.WebGLRenderer({antialias:true});
view.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff,0.85));
const dl=new THREE.DirectionalLight(0xffffff,0.55); dl.position.set(1,2,3); scene.add(dl);

const c0=(()=>{let s=[0,0,0];for(let i=0;i<NG;i++){s[0]+=D.x0[3*i];s[1]+=D.x0[3*i+1];s[2]+=D.x0[3*i+2];}
  return s.map(v=>v/Math.max(NG,1));})();
let radius=0.5;
for(let i=0;i<NG;i++){const dx=D.x0[3*i]-c0[0],dy=D.x0[3*i+1]-c0[1],dz=D.x0[3*i+2]-c0[2];
  radius=Math.max(radius,Math.hypot(dx,dy,dz));}
const target=new THREE.Vector3(...c0);
let az=0.6, el=0.35, dist=radius*3.4;
function place(){cam.position.set(target.x+dist*Math.cos(el)*Math.sin(az),
  target.y+dist*Math.sin(el), target.z+dist*Math.cos(el)*Math.cos(az)); cam.lookAt(target);}
function resize(){const w=view.clientWidth,h=view.clientHeight;
  renderer.setSize(w,h); renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  cam.aspect=w/Math.max(h,1); cam.updateProjectionMatrix();}
addEventListener("resize",resize);
let drag=null;
renderer.domElement.addEventListener("pointerdown",e=>drag={x:e.clientX,y:e.clientY});
addEventListener("pointerup",()=>drag=null);
addEventListener("pointermove",e=>{if(!drag)return;
  az-=(e.clientX-drag.x)*0.006; el=Math.max(-1.4,Math.min(1.4,el+(e.clientY-drag.y)*0.006));
  drag={x:e.clientX,y:e.clientY}; place();});
renderer.domElement.addEventListener("wheel",e=>{e.preventDefault();
  dist=Math.max(radius*0.3,Math.min(radius*40,dist*(1+Math.sign(e.deltaY)*0.1))); place();},
  {passive:false});

function colFor(k){const c=M.part_colors[k%M.part_colors.length]||"#888";
  return new THREE.Color(c);}

/* main point cloud + ghost cloud (section 43) */
function makeCloud(op){
  const g=new THREE.BufferGeometry();
  g.setAttribute("position",new THREE.BufferAttribute(new Float32Array(NG*3),3));
  g.setAttribute("color",new THREE.BufferAttribute(new Float32Array(NG*3),3));
  const m=new THREE.PointsMaterial({size:Math.max(radius*0.012,0.002),
    vertexColors:true,transparent:true,opacity:op,sizeAttenuation:true});
  const p=new THREE.Points(g,m); scene.add(p); return p;
}
const cloud=makeCloud(0.95);
/* Section 1 lists the four stage clouds as TOGGLES as well as a selector, so
   each gets its own translucent overlay and several can be shown at once.
   That is also section 43's ghost comparison. */
const overlays={canonical:makeCloud(0.22),bootstrap:makeCloud(0.22),
                rigid:makeCloud(0.22),refined:makeCloud(0.22)};
const ghost=makeCloud(0.25);
const trails=new THREE.Group(); scene.add(trails);
const comGrp=new THREE.Group(); scene.add(comGrp);
const jointGrp=new THREE.Group(); scene.add(jointGrp);
const frameGrp=new THREE.Group(); scene.add(frameGrp);
const camGrp=new THREE.Group(); scene.add(camGrp);
const glyphGrp=new THREE.Group(); scene.add(glyphGrp);

function line(pts,color,w){
  const g=new THREE.BufferGeometry().setFromPoints(
    pts.map(p=>new THREE.Vector3(p[0],p[1],p[2])));
  return new THREE.Line(g,new THREE.LineBasicMaterial({color:color,linewidth:w||1}));
}

/* ---- section 11: camera frustums, fixed in the world frame ---- */
D.cameras.forEach(c=>{
  const g=new THREE.Group();
  c.corners.forEach(k=>g.add(line([c.centre,k],0x58a6ff)));
  g.add(line([...c.corners,c.corners[0]],0x58a6ff));
  g.userData.id=c.id; camGrp.add(g);
});

/* ---- sections 5/6/7: the joint display ---- */
function buildJoints(){
  jointGrp.clear();
  (D.direct_estimators||[]).forEach(E=>{
    ["SE3","Gaussian","RGBD"].forEach((name,i)=>{
      const J=E[name];if(!J||!J.axis)return;
      const a=J.axis_point||J.c,e=J.axis,s=(J.length_scale||radius)*4;
      jointGrp.add(line([a.map((v,j)=>v-s*e[j]),a.map((v,j)=>v+s*e[j])],[0x58a6ff,0x3fb950,0xbc8cff][i],2));
    });
  });
  (D.joints||[]).forEach((J,i)=>{
    if(!J) return; const k=i+1, col=colFor(k).getHex();
    const L=(J.length_scale||radius*0.5);
    if(J.selected==="revolute"||J.status==="UNRESOLVED_GEOMETRY"){
      const e=J.revolute_axis||J.axis, a=J.revolute_axis_point||J.axis_point;
      if(e&&a){
        const s=L*6;
        jointGrp.add(line([[a[0]-e[0]*s,a[1]-e[1]*s,a[2]-e[2]*s],
                           [a[0]+e[0]*s,a[1]+e[1]*s,a[2]+e[2]*s]],col,3));
        const dot=new THREE.Mesh(new THREE.SphereGeometry(L*0.09,12,12),
          new THREE.MeshBasicMaterial({color:col}));
        dot.position.set(a[0],a[1],a[2]); jointGrp.add(dot);
        // section 5: shortest segment from the part COM to the axis
        const com=(J.com&&J.com[0])?J.com[0]:null;
        if(com){const d=[com[0]-a[0],com[1]-a[1],com[2]-a[2]];
          const t=d[0]*e[0]+d[1]*e[1]+d[2]*e[2];
          const foot=[a[0]+e[0]*t,a[1]+e[1]*t,a[2]+e[2]*t];
          jointGrp.add(line([com,foot],0xf0883e,2));}
      }
    }
    if(J.selected==="prismatic"||J.status==="UNRESOLVED_GEOMETRY"){
      const e=J.prismatic_axis||J.axis;
      const anchor=(J.com&&J.com[0])?J.com[0]:c0;   // section 6: DISPLAY ANCHOR
      if(e){const s=L*4;
        jointGrp.add(line([[anchor[0]-e[0]*s,anchor[1]-e[1]*s,anchor[2]-e[2]*s],
                           [anchor[0]+e[0]*s,anchor[1]+e[1]*s,anchor[2]+e[2]*s]],
                          J.selected==="prismatic"?col:0x8b949e,2));}
    }
  });
}
buildJoints();

/* ---- section 9: COM trajectories ---- */
function buildCom(){
  comGrp.clear();
  (D.joints||[]).forEach((J,i)=>{ if(J&&J.com&&J.com.length>1)
    comGrp.add(line(J.com,colFor(i+1).getHex(),2)); });
}
buildCom();

/* ---- section 10: motion trails ---- */
let trailIdx=[];
for(let i=0;i<NG && trailIdx.length<40;i+=1) if(D.labels[i]>0) trailIdx.push(i);
function buildTrails(stage,f){
  trails.clear();
  trailIdx.forEach(i=>{
    const pts=[];
    for(let t=0;t<=f;t++){const P=cachePos(stage,t);
      pts.push([P[3*i],P[3*i+1],P[3*i+2]]);}
    if(pts.length>1) trails.add(line(pts,colFor(D.labels[i]).getHex()));
  });
}

/* ---- section 8: SE(3) triads ---- */
function buildFrames(f){
  frameGrp.clear();
  if(!D.rigid||!D.rigid[f]) return;
  D.rigid[f].forEach((tr,ki)=>{
    const L=radius*0.18, o=tr.t.slice();
    const ax=[[0xff5555,[tr.R[0],tr.R[3],tr.R[6]]],
              [0x55ff55,[tr.R[1],tr.R[4],tr.R[7]]],
              [0x5599ff,[tr.R[2],tr.R[5],tr.R[8]]]];
    // the triad is drawn at the part's transported centroid, not the origin
    const J=D.joints[ki]; const base=(J&&J.com&&J.com[f])?J.com[f]:o;
    ax.forEach(([c,v])=>frameGrp.add(
      line([base,[base[0]+v[0]*L,base[1]+v[1]*L,base[2]+v[2]*L]],c,2)));
  });
}

/* ---- section 3: orientation glyphs ---- */
function buildGlyphs(stage,f){
  glyphGrp.clear();
  if(!D.glyph||!D.glyph.index.length) return;
  const P=cachePos(stage,f), L=radius*0.05;
  D.glyph.index.forEach((gi,n)=>{
    const k=D.labels[gi]; let R=[1,0,0,0,1,0,0,0,1];
    if(stage==="rigid"&&D.rigid[f]&&k>0) R=D.rigid[f][k-1].R;
    else if((stage==="joint"||stage==="refined")&&k>0&&D.joints[k-1]
            &&D.joints[k-1].selected==="revolute"){
      const J=D.joints[k-1]; R=rotAxis(J.axis,J.q[f]||0);
    }
    const o=[P[3*gi],P[3*gi+1],P[3*gi+2]];
    [[0xff7777,[R[0],R[3],R[6]]],[0x77ff77,[R[1],R[4],R[7]]],
     [0x7799ff,[R[2],R[5],R[8]]]].forEach(([c,v])=>
      glyphGrp.add(line([o,[o[0]+v[0]*L,o[1]+v[1]*L,o[2]+v[2]*L]],c)));
  });
}

/* ------------------------------- state ------------------------------- */
const cache={};
function cachePos(stage,f){const key=stage+"|"+f;
  if(!cache[key]) cache[key]=positions(stage,f); return cache[key];}
let stage="refined", ghostStage="none", frame=0, playing=false, selPart=1, selSn=-1, selCam=-1;
let directEstimator="RGBD",jointColor="none";
document.getElementById("estimator").onchange=e=>{directEstimator=e.target.value;
  Object.keys(cache).forEach(k=>delete cache[k]);stage="direct";paint();};
document.getElementById("jointcolor").onchange=e=>{jointColor=e.target.value;paint();};
const show={gauss:true,labels:true,resid:false,evidence:false,cams:true,
            com:true,frames:false,joint:true,trails:false,glyph:false,ghost:false,
            ov_canonical:false,ov_bootstrap:false,ov_rigid:false,ov_refined:false};

function paint(){
  const P=cachePos(stage,frame);
  const pos=cloud.geometry.attributes.position, col=cloud.geometry.attributes.color;
  let rmax=1e-9; for(let i=0;i<NG;i++) rmax=Math.max(rmax,D.rigid_residual[i]||0);
  let emax=1e-9; (D.evidence_by_supernode||[]).forEach(v=>emax=Math.max(emax,v||0));
  const heatValues=(D.direct_heat||{})[jointColor];
  const heatMax=heatValues?Math.max(...heatValues.map(v=>Math.abs(v)),1e-30):1;
  for(let i=0;i<NG;i++){
    pos.array[3*i]=P[3*i]; pos.array[3*i+1]=P[3*i+1]; pos.array[3*i+2]=P[3*i+2];
    let c;
    if(jointColor!=="none"&&heatValues){
      const v=heatValues[i]/heatMax;c=new THREE.Color(v>0?Math.abs(v):0,.2,v<0?Math.abs(v):1-Math.abs(v));}
    else if(show.resid){const v=Math.min(1,(D.rigid_residual[i]||0)/rmax);
      c=new THREE.Color(v,0.25,1-v);}
    else if(show.evidence){          /* section 1: motion-evidence heatmap */
      const sn=(D.supernode_of||[])[i];
      const e=(sn!==undefined&&D.evidence_by_supernode)?
        (D.evidence_by_supernode[sn]||0):0;
      const v=Math.min(1,Math.log10(1+Math.max(e,0))/Math.log10(1+emax));
      c=new THREE.Color(v,v*0.55,1-v);}
    else if(show.labels) c=colFor(D.labels[i]);
    else c=new THREE.Color(D.colors[3*i],D.colors[3*i+1],D.colors[3*i+2]);
    if(selSn>=0&&D.supernodes[selSn]&&D.labels[i]!==D.supernodes[selSn].label)
      c=c.clone().multiplyScalar(0.25);
    col.array[3*i]=c.r; col.array[3*i+1]=c.g; col.array[3*i+2]=c.b;
  }
  pos.needsUpdate=true; col.needsUpdate=true;
  cloud.visible=show.gauss;

  /* section 1: independent stage overlays */
  Object.entries(overlays).forEach(([name,pt])=>{
    if(!show["ov_"+name]||name===stage){pt.visible=false;return;}
    const Q=cachePos(name,frame);
    const op=pt.geometry.attributes.position, oc=pt.geometry.attributes.color;
    const tint=({canonical:[0.55,0.55,0.60],bootstrap:[0.95,0.75,0.25],
                 rigid:[0.35,0.85,0.95],refined:[0.45,0.95,0.55]})[name];
    for(let i=0;i<NG;i++){op.array[3*i]=Q[3*i];op.array[3*i+1]=Q[3*i+1];
      op.array[3*i+2]=Q[3*i+2];oc.array[3*i]=tint[0];oc.array[3*i+1]=tint[1];
      oc.array[3*i+2]=tint[2];}
    op.needsUpdate=true; oc.needsUpdate=true; pt.visible=true;
  });

  if(show.ghost&&ghostStage!=="none"){
    const G=cachePos(ghostStage,frame);
    const gp=ghost.geometry.attributes.position, gc=ghost.geometry.attributes.color;
    for(let i=0;i<NG;i++){gp.array[3*i]=G[3*i];gp.array[3*i+1]=G[3*i+1];
      gp.array[3*i+2]=G[3*i+2];gc.array[3*i]=1;gc.array[3*i+1]=1;gc.array[3*i+2]=1;}
    gp.needsUpdate=true; gc.needsUpdate=true; ghost.visible=true;
  } else ghost.visible=false;

  camGrp.visible=show.cams; comGrp.visible=show.com; jointGrp.visible=show.joint;
  if(show.frames) buildFrames(frame); frameGrp.visible=show.frames;
  if(show.trails) buildTrails(stage,frame); trails.visible=show.trails;
  if(show.glyph) buildGlyphs(stage,frame); glyphGrp.visible=show.glyph;
  camGrp.children.forEach(g=>g.visible=show.cams&&(selCam<0||g.userData.id===selCam));
  dash();
}

/* --------------------------- section 39 dashboard --------------------------- */
function fmt(v,d){return (v===null||v===undefined||v==="")?"—":
  (typeof v==="number"?(Math.abs(v)<1e-3&&v!==0?v.toExponential(2):v.toFixed(d===undefined?4:d)):v);}
function rows(el,pairs){document.getElementById(el).innerHTML=
  pairs.map(([a,b])=>`<div class="row"><span>${a}</span><span>${b}</span></div>`).join("");}

function dash(){
  const fr=D.frame_rows.find(r=>r.frame===M.frame_ids[frame])||{};
  const J=(D.joints||[])[selPart-1];
  const p=[["frame",`${M.frame_ids[frame]} (${frame+1}/${NF})`],
           ["timestamp",fmt(frame*M.dt)],
           ["stage",stage],["selected part",selPart]];
  if(J){
    const q=(J.q&&J.q[frame]!==undefined)?J.q[frame]:null;
    const qp=(J.q&&frame>0&&J.q[frame-1]!==undefined)?(q-J.q[frame-1])/M.dt:null;
    p.push(["joint type",`<span class="chip" style="background:${
      J.selected?"#2ea04355":"#f0883e55"}">${J.selected||J.status}</span>`]);
    p.push(["q",fmt(q,5)],["qdot",fmt(qp,5)]);
    p.push(["part Gaussians",D.labels.filter(l=>l===selPart).length]);
    p.push(["E_P geom",fmt(J.E_P)],["E_R geom",fmt(J.E_R)]);
    p.push(["gap / floor",`${fmt(J.gap)} / ${fmt(J.floor)}`]);
    p.push(["rigid residual (median)",fmt(J.rigid_rmse_median)]);
    if(J.selected==="revolute"||J.status==="UNRESOLVED_GEOMETRY"){
      p.push(["e",(J.revolute_axis||[]).map(x=>x.toFixed(4)).join(", ")]);
      p.push(["a",(J.revolute_axis_point||[]).map(x=>x.toFixed(4)).join(", ")]);
      p.push(["m = a x e",(J.moment||[]).map(x=>x.toFixed(4)).join(", ")]);
      p.push(["gauge e.a",fmt(J.revolute_gauge)]);
      if(J.com&&J.com[frame]&&J.revolute_axis_point){
        const a=J.revolute_axis_point,e=J.revolute_axis,c=J.com[frame];
        const d=[c[0]-a[0],c[1]-a[1],c[2]-a[2]];
        const t=d[0]*e[0]+d[1]*e[1]+d[2]*e[2];
        p.push(["COM-axis distance",fmt(Math.hypot(d[0]-e[0]*t,d[1]-e[1]*t,d[2]-e[2]*t))]);
      }
    }
    if(J.selected==="prismatic"||J.status==="UNRESOLVED_GEOMETRY"){
      p.push(["e (P)",(J.prismatic_axis||[]).map(x=>x.toFixed(4)).join(", ")]);
      p.push(["rank-1 ratio",fmt(J.prismatic_rank1,6)]);
      p.push(["rotation residual",fmt(J.prismatic_rotation_residual)]);
    }
  }
  const cr=D.camera_rows.filter(r=>r.frame===M.frame_ids[frame]);
  if(cr.length){
    const avg=(k)=>cr.reduce((s,r)=>s+(r[k]||0),0)/cr.length;
    p.push(["rgb rmse (mean over cams)",fmt(avg("rgb_rmse"))]);
    p.push(["depth rmse",fmt(avg("depth_rmse"))]);
    p.push(["fisher trace",fmt(avg("fisher_trace"))]);
  }
  if(D.observability&&D.observability.condition!==undefined)
    p.push(["SE(3) Fisher condition",fmt(D.observability.condition)]);
  const E=(D.direct_estimators||[])[selPart-1], C=E?E[directEstimator]:null;
  if(C){
    p.push(["direct estimator",directEstimator],["curvature",fmt(C.kappa)],
           ["fit status",C.status]);
    const U=C.curvature||{}, V=E.decision||{};
    p.push(["curvature SNR",fmt(U.curvature_snr)],["curvature rank",fmt(U.rank_beta)],
           ["RGB-D decision",V.status||"—"],["validation mode",V.validation_mode||"—"]);
    if(V.predictive&&V.predictive.ci)
      p.push(["held-out gain CI",V.predictive.ci.map(v=>fmt(v)).join(" … ")]);
  }
  rows("dash",p);
  document.getElementById("hud").innerHTML =
    `frame ${M.frame_ids[frame]} &middot; ${stage}` +
    (J?` &middot; part ${selPart}: <b>${J.selected||J.status}</b>`:"");
}

/* ------------------------------- UI ------------------------------- */
const stageSel=document.getElementById("stage"), ghostSel=document.getElementById("ghost");
STAGES.forEach(([v,t])=>{
  const ok=M.stages_available[v]!==false;
  stageSel.insertAdjacentHTML("beforeend",
    `<option value="${v}" ${ok?"":"disabled"} ${v==="refined"?"selected":""}>${t}${ok?"":" (n/a)"}</option>`);
  ghostSel.insertAdjacentHTML("beforeend",`<option value="${v}">ghost: ${t}</option>`);
});
ghostSel.insertAdjacentHTML("afterbegin",`<option value="none" selected>ghost: off</option>`);
stageSel.onchange=e=>{stage=e.target.value;paint();};
ghostSel.onchange=e=>{ghostStage=e.target.value;show.ghost=ghostStage!=="none";paint();};

const TOG=[["gauss","Gaussians (selected stage)"],
           ["ov_canonical","canonical Gaussians"],
           ["ov_bootstrap","bootstrap/free Gaussians"],
           ["ov_rigid","rigid-part Gaussians"],
           ["ov_refined","joint-constrained Gaussians"],
           ["labels","part labels"],
           ["resid","rigid residual heatmap"],["evidence","motion-evidence heatmap"],
           ["cams","camera frustums"],["com","COM trajectories"],
           ["frames","rigid SE(3) frames"],["joint","estimated joint"],
           ["trails","motion trails"],["glyph","orientation glyphs"],
           ["ghost","ghost overlay"]];
document.getElementById("toggles").innerHTML=TOG.map(([k,t])=>
  `<label><input type="checkbox" data-k="${k}" ${show[k]?"checked":""}> ${t}</label>`).join("");
document.querySelectorAll("#toggles input").forEach(i=>i.onchange=e=>{
  show[e.target.dataset.k]=e.target.checked; paint();});

const slider=document.getElementById("slider");
slider.max=NF-1;
slider.oninput=e=>{frame=+e.target.value;document.getElementById("fnum").textContent=
  `${frame+1}/${NF}`;paint();panelLinks();};
document.getElementById("first").onclick=()=>{frame=0;slider.value=0;paint();};
document.getElementById("last").onclick=()=>{frame=NF-1;slider.value=frame;paint();};
const playBtn=document.getElementById("play");
playBtn.onclick=()=>{playing=!playing;playBtn.textContent=playing?"pause":"play";};

document.getElementById("meta").innerHTML=
  `${M.model||"model"} &middot; ${M.num_gaussians} Gaussians `+
  `(${M.shown_gaussians} shown) &middot; ${M.num_supernodes} supernodes &middot; `+
  `K=${K} &middot; ${M.frames_total} frames &middot; `+
  `<span class="${M.rigid_converged?"good":"warn"}">rigid ${
    M.rigid_converged?"converged":"NOT converged"}</span>`;

/* section 46: joint fit inspection */
document.getElementById("parts").innerHTML=(D.joints||[]).map((J,i)=>{
  if(!J) return `<div class="row"><span>part ${i+1}</span><span>no fit</span></div>`;
  const q=J.q||[];
  const bad=!J.selected;
  return `<div class="row" data-part="${i+1}" style="cursor:pointer">
    <span><span class="sw" style="background:${M.part_colors[(i+1)%M.part_colors.length]}"></span>
    part ${i+1}</span><span class="${bad?"warn":"good"}">${J.selected||J.status}</span></div>
    <div class="note">E_P ${fmt(J.E_P)} &middot; E_R ${fmt(J.E_R)} &middot;
    floor ${fmt(J.floor)}<br>q range ${fmt(Math.min(...q))} … ${fmt(Math.max(...q))}</div>`;
}).join("")||"<div class='note'>no moving parts</div>";
document.querySelectorAll("#parts .row[data-part]").forEach(r=>
  r.onclick=()=>{selPart=+r.dataset.part;paint();});

/* section 45: supernode inspection */
function snTable(){
  const f=document.getElementById("snfilter").value;
  const body=document.querySelector("#sntab tbody");
  body.innerHTML=(D.supernodes||[]).filter(s=>f==="all"||s.label>0).map(s=>
    `<tr data-id="${s.id}" class="${s.id===selSn?"sel":""}"><td>${s.id}</td>
     <td><span class="sw" style="background:${M.part_colors[s.label%M.part_colors.length]}"></span>${s.label}</td>
     <td>${s.members}</td><td>${fmt(s.evidence,1)}</td><td>${fmt(s.margin)}</td></tr>`).join("");
  body.querySelectorAll("tr").forEach(tr=>tr.onclick=()=>{
    selSn=+tr.dataset.id; const s=D.supernodes[selSn];
    document.getElementById("snsel").innerHTML=
      `supernode ${s.id} &middot; label ${s.label} &middot; ${s.members} members<br>`+
      `evidence ${fmt(s.evidence)}<br>`+
      (s.costs?`costs ${s.costs.map(c=>fmt(c)).join(" / ")}<br>margin ${fmt(s.margin)}`:"");
    snTable(); paint();});
}
document.getElementById("snfilter").onchange=snTable; snTable();

/* section 44: camera inspection */
const camSel=document.getElementById("cam");
D.cameras.forEach(c=>camSel.insertAdjacentHTML("beforeend",
  `<option value="${c.id}">${c.name}</option>`));
camSel.onchange=e=>{selCam=+e.target.value;
  const r=D.camera_rows.filter(x=>x.camera_id===selCam&&x.frame===M.frame_ids[frame])[0];
  document.getElementById("caminfo").innerHTML=r?
    `valid rgb ${r.valid_rgb_px} &middot; valid depth ${r.valid_depth_px}<br>`+
    `rgb rmse ${fmt(r.rgb_rmse)} &middot; depth rmse ${fmt(r.depth_rmse)}<br>`+
    `fisher trace ${fmt(r.fisher_trace)}`:
    "<i>no per-camera metrics for this frame</i>";
  paint();};

/* section 12: link the per-frame/per-camera panels for the current frame */
function panelLinks(){
  const el=document.getElementById("panels");
  const all=D.render_panels||[];
  if(!all.length){el.innerHTML="<i>not generated (use --debug-render-panels)</i>";return;}
  const tag=`frame_${String(frame*(M.frame_stride||1)).padStart(4,"0")}_`;
  const here=all.filter(p=>p.includes(tag));
  el.innerHTML=(here.length?here:all.slice(0,4)).map(p=>
    `<a href="${p}" target="_blank" style="color:var(--accent)">${p.split("/").pop()}</a>`
  ).join("<br>")+(here.length?"":"<br><i>(no panel for this frame; showing the first few)</i>");
}

let last=0;
function loop(ts){
  requestAnimationFrame(loop);
  const sp=+document.getElementById("speed").value;
  if(playing && ts-last>100/sp){
    last=ts; frame++;
    if(frame>=NF){ if(document.getElementById("loop").checked) frame=0;
                   else {frame=NF-1;playing=false;playBtn.textContent="play";} }
    slider.value=frame; document.getElementById("fnum").textContent=`${frame+1}/${NF}`;
    paint(); panelLinks();
  }
  renderer.render(scene,cam);
}
resize(); place(); paint(); panelLinks();
document.getElementById("fnum").textContent=`1/${NF}`;
requestAnimationFrame(loop);
</script>
</body>
</html>
"""
