"""Mandatory artifacts, ground-truth bookkeeping and debug visualizations.

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md sections 2, 22, 23.

Section 23's closing line is the requirement this module exists for:

    "No critical result may exist only in terminal scrollback."

and section 2.1's, which is about honesty rather than storage:

    "Never call imposed motion a real articulated-data result."

So every run declares its observation source as one of
``synthetic_renderer_generated`` / ``dataset_rgbd``, and the synthetic ground
truth is written with the ``gt_`` names section 2.2 reserves, then RELOADED
from disk and used to regenerate the state -- section 2.2's mandatory
round-trip. Metrics are computed from the reloaded artifact, so an in-memory
variable that drifted from what actually generated the observations cannot
quietly produce a good number.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

SYNTHETIC = "synthetic_renderer_generated"
DATASET_RGBD = "dataset_rgbd"
VALID_SOURCES = (SYNTHETIC, DATASET_RGBD)

EPS = 1e-12


def jsonable(value):
    """Non-finite floats become null: a NaN in a debug file is
    indistinguishable from a parse error downstream."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    return str(value)


# ---------------------------------------------------------------------------
# section 2: observation source and immutable ground truth
# ---------------------------------------------------------------------------
@dataclass
class ObservationSource:
    """Section 2.1's mandatory declaration."""

    kind: str
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in VALID_SOURCES:
            raise ValueError(
                f"observation source must be one of {VALID_SOURCES}, got "
                f"{self.kind!r}. Section 2.1 requires every report and artifact "
                f"to state which generated the observations.")

    @property
    def is_synthetic(self) -> bool:
        return self.kind == SYNTHETIC

    def to_dict(self) -> dict:
        d = {"observation_source": self.kind, **self.detail}
        if self.is_synthetic:
            d["description"] = ("synthetic imposed motion on real reconstructed "
                                "geometry")
            d["is_not"] = ("joint discovery from a real RGB-D articulated "
                           "sequence")
        else:
            d["description"] = ("real synchronized dataset RGB-D frames; no "
                                "synthetic motion was applied")
            d["is_not"] = ("a ground-truth-verified joint measurement, unless "
                           "the dataset itself provides joint GT")
        return d


@dataclass
class GroundTruthMotion:
    """Section 2.2's reserved ``gt_`` names, written once and never reused for
    an estimate."""

    gt_joint_type: str
    gt_axis_direction: list
    gt_axis_point: list | None
    gt_axis_moment: list | None
    gt_q: list
    gt_qdot: list
    gt_dt: float
    gt_centroid: list | None = None
    gt_seed: int = 0
    gt_num_gaussians: int = 0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def write(self, path: Path) -> Path:
        path.write_text(json.dumps(jsonable(self.to_dict()), indent=2))
        return path

    @staticmethod
    def load(path: Path) -> "GroundTruthMotion":
        d = json.loads(Path(path).read_text())
        return GroundTruthMotion(**{k: d[k] for k in
                                    GroundTruthMotion.__dataclass_fields__})

    def transform(self, mu0: torch.Tensor, R0: torch.Tensor, index: int):
        """Regenerate gt_T[t] applied to the canonical state.

        This is the ONLY place the synthetic transform is defined, so the
        round-trip below genuinely re-derives the observation geometry rather
        than comparing two copies of the same in-memory tensor.
        """
        from estimator.manifold import so3_exp

        dev, dt = mu0.device, mu0.dtype
        e = torch.tensor(self.gt_axis_direction, device=dev, dtype=dt)
        e = e / e.norm().clamp_min(EPS)
        q = float(self.gt_q[index])
        if self.gt_joint_type == "prismatic":
            return mu0 + e.reshape(1, 3) * q, R0
        Rq = so3_exp((e * q).reshape(1, 3)).reshape(3, 3)
        if self.gt_axis_point is not None:
            a = torch.tensor(self.gt_axis_point, device=dev, dtype=dt)
        elif self.gt_centroid is not None:
            a = torch.tensor(self.gt_centroid, device=dev, dtype=dt)
        else:
            a = torch.zeros(3, device=dev, dtype=dt)
        mu = (mu0 - a.reshape(1, 3)) @ Rq.T + a.reshape(1, 3)
        return mu, (Rq.unsqueeze(0) @ R0 if R0 is not None else None)


def ground_truth_round_trip(gt: GroundTruthMotion, path: Path,
                            mu0: torch.Tensor, R0: torch.Tensor,
                            generated_states: list) -> dict:
    """Section 2.2's mandatory round-trip test.

    1. save, 2. reload from disk, 3. regenerate, 4. assert equality with the
    state that generated the observations, 5. compute errors from the RELOADED
    artifact. "Any mismatch invalidates the synthetic metric."
    """
    gt.write(path)
    reloaded = GroundTruthMotion.load(path)
    worst_mu, worst_R = 0.0, 0.0
    for t, (mu_used, R_used) in enumerate(generated_states):
        mu_re, R_re = reloaded.transform(mu0, R0, t)
        worst_mu = max(worst_mu, float((mu_re - mu_used).abs().max()))
        if R_used is not None and R_re is not None:
            worst_R = max(worst_R, float((R_re - R_used).abs().max()))
    ok = worst_mu < 1e-5 and worst_R < 1e-5
    return {
        "round_trip_performed": True,
        "artifact": str(path),
        "max_mu_mismatch": worst_mu,
        "max_R_mismatch": worst_R,
        "ok": bool(ok),
        "frames_checked": len(generated_states),
        "note": ("section 2.2: all axis/type errors are computed from this "
                 "reloaded artifact, not from in-memory variables"
                 if ok else
                 "MISMATCH -- section 2.2: the synthetic metric is invalid"),
    }


def axis_error_from_artifact(path: Path, estimated_axis: torch.Tensor,
                             estimated_kind: str,
                             estimated_point: torch.Tensor | None = None) -> dict:
    """Section 2.2 step 5: errors computed from the reloaded GT artifact."""
    gt = GroundTruthMotion.load(path)
    e_gt = torch.tensor(gt.gt_axis_direction, dtype=torch.float64)
    e_gt = e_gt / e_gt.norm().clamp_min(EPS)
    e_est = estimated_axis.detach().cpu().double()
    e_est = e_est / e_est.norm().clamp_min(EPS)
    cos = float(torch.clamp((e_gt @ e_est).abs(), -1.0, 1.0))
    out = {"gt_joint_type": gt.gt_joint_type,
           "estimated_joint_type": estimated_kind,
           "joint_type_correct": gt.gt_joint_type == estimated_kind,
           "axis_angle_error_deg": math.degrees(math.acos(cos)),
           "gt_axis_direction": gt.gt_axis_direction,
           "estimated_axis_direction": e_est.tolist(),
           "computed_from": str(path)}
    if estimated_point is not None and gt.gt_axis_point is not None:
        from joint_reduction.line_geometry import PluckerLine
        gl = PluckerLine.from_point_direction(
            torch.tensor(gt.gt_axis_point, dtype=torch.float64), e_gt)
        el = PluckerLine.from_point_direction(
            estimated_point.detach().cpu().double(), e_est)
        d = el.distance_to(gl)
        out.update({"axis_line_distance": d["line_distance"],
                    "axis_point_distance": d["point_distance"]})
    return out


# ---------------------------------------------------------------------------
# section 23: the mandatory artifact set
# ---------------------------------------------------------------------------
REQUIRED_ARTIFACTS = (
    "config.json", "observation_source.json", "jacobian_chain_audit.json",
    "adjoint_test.json", "trust_region_history.json",
    "linear_solver_history.json", "observability.json", "supernodes.json",
    "support_iteration_history.json", "support_hard.pt",
    "support_probability.pt", "joint_hypotheses.json", "joint_selected.json",
    "joint_axis_geometry.json", "joint_window_refinement.json",
    "handover_refit.json", "q_trace.csv", "sequence.json",
    "convergence_status.json", "state_final.pt",
)

SYNTHETIC_ARTIFACTS = ("ground_truth_motion.json", "ground_truth_support.pt",
                       "metrics.json")


class ArtifactWriter:
    """Writes section 23's set and reports what is present or missing."""

    def __init__(self, out_dir: str | Path):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.written: list[str] = []

    def json(self, name: str, payload) -> Path:
        p = self.dir / (name if name.endswith(".json") else f"{name}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(jsonable(payload), indent=2))
        self.written.append(str(p.relative_to(self.dir)))
        return p

    def tensor(self, name: str, obj) -> Path:
        p = self.dir / (name if name.endswith(".pt") else f"{name}.pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, p)
        self.written.append(str(p.relative_to(self.dir)))
        return p

    def csv(self, name: str, header: list[str], rows: list[list]) -> Path:
        p = self.dir / (name if name.endswith(".csv") else f"{name}.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        self.written.append(str(p.relative_to(self.dir)))
        return p

    def text(self, name: str, content: str) -> Path:
        p = self.dir / name
        p.write_text(content)
        self.written.append(str(p.relative_to(self.dir)))
        return p

    def audit(self, synthetic: bool) -> dict:
        """Which of section 23's artifacts exist, and which do not.

        Reported rather than silently tolerated: a missing artifact is a
        missing result, and the manifest says so explicitly.
        """
        want = list(REQUIRED_ARTIFACTS)
        if synthetic:
            want += list(SYNTHETIC_ARTIFACTS)
        present = {p.name for p in self.dir.iterdir() if p.is_file()}
        missing = [w for w in want if w not in present]
        return {"out_dir": str(self.dir),
                "required": want,
                "present": sorted(present),
                "missing": missing,
                "complete": not missing,
                "extra": sorted(present - set(want))}

    def write_manifest(self, synthetic: bool) -> dict:
        a = self.audit(synthetic)
        self.json("artifact_manifest.json", a)
        return a


# ---------------------------------------------------------------------------
# section 22: debug visualizations
# ---------------------------------------------------------------------------
def write_ply(path: Path, xyz: torch.Tensor, rgb: torch.Tensor) -> Path:
    """A plain binary-free PLY so it opens anywhere without a dependency."""
    xyz = xyz.detach().cpu().double()
    rgb = (rgb.detach().cpu().clamp(0, 1) * 255).round().to(torch.int64)
    lines = ["ply", "format ascii 1.0", f"element vertex {xyz.shape[0]}",
             "property float x", "property float y", "property float z",
             "property uchar red", "property uchar green", "property uchar blue",
             "end_header"]
    for p, c in zip(xyz.tolist(), rgb.tolist()):
        lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}")
    path.write_text("\n".join(lines) + "\n")
    return path


def support_colors(prob: torch.Tensor) -> torch.Tensor:
    """Blue = static, red = moving, through grey."""
    p = prob.detach().cpu().clamp(0, 1).reshape(-1, 1)
    return torch.cat([p, 0.25 * torch.ones_like(p), 1.0 - p], dim=1)


def save_residual_png(path: Path, image: torch.Tensor,
                      vmax: float | None = None) -> Path | None:
    """Per-camera residual / precision image (section 22 items 6-8).

    Falls back to a .pt dump if matplotlib is unavailable, rather than silently
    skipping a required artifact.
    """
    arr = image.detach().cpu()
    if arr.dim() == 3:
        arr = arr.abs().mean(dim=0)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4), dpi=110)
        im = ax.imshow(arr.numpy(), cmap="inferno",
                       vmax=vmax if vmax else None)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:
        alt = path.with_suffix(".pt")
        torch.save(arr, alt)
        return alt


def save_plot(path: Path, series: dict, title: str, xlabel: str = "iteration",
              logy: bool = False) -> Path | None:
    """Trust radius / rho / PCG residual and q/qdot plots (section 22 items
    10-11)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.6), dpi=110)
        for name, values in series.items():
            vals = [v if (v is not None and math.isfinite(v)) else float("nan")
                    for v in values]
            ax.plot(range(len(vals)), vals, marker="o", ms=3, label=name)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
    except Exception:
        torch.save(series, path.with_suffix(".pt"))
        return path.with_suffix(".pt")


def axis_polyline(line_or_axis, point=None, length: float = 1.0, n: int = 64):
    """Points along the fitted joint axis, for the 3D view (section 22 items
    3-4)."""
    if hasattr(line_or_axis, "direction"):
        e = line_or_axis.direction.detach().cpu().double()
        a = line_or_axis.closest_point.detach().cpu().double()
    else:
        e = line_or_axis.detach().cpu().double()
        e = e / e.norm().clamp_min(EPS)
        a = (point.detach().cpu().double() if point is not None
             else torch.zeros(3, dtype=torch.float64))
    t = torch.linspace(-length, length, n, dtype=torch.float64).reshape(-1, 1)
    return a.reshape(1, 3) + t * e.reshape(1, 3)


def write_scene_html(path: Path, xyz: torch.Tensor, colors: torch.Tensor,
                     *, axis_points: torch.Tensor | None = None,
                     motion_vectors: tuple | None = None,
                     cameras: list | None = None,
                     title: str = "IEKF scene", extra: dict | None = None,
                     max_points: int = 40000) -> Path:
    """Section 22's interactive 3D view.

    Self-contained: the viewer is inlined, so the file opens from disk with no
    network access and no CDN. Section 22 asks for an interactive HTML for the
    3D geometry; the static PNGs cover the image-space quantities.
    """
    xyz = xyz.detach().cpu().double()
    colors = colors.detach().cpu().clamp(0, 1)
    if xyz.shape[0] > max_points:
        idx = torch.linspace(0, xyz.shape[0] - 1, max_points).long()
        xyz, colors = xyz[idx], colors[idx]
    pts = [[round(v, 5) for v in p] for p in xyz.tolist()]
    cols = [[round(v, 4) for v in c] for c in colors.tolist()]
    ax = ([[round(v, 5) for v in p] for p in axis_points.detach().cpu().tolist()]
          if axis_points is not None else [])
    mv = []
    if motion_vectors is not None:
        starts, ends = motion_vectors
        s = starts.detach().cpu().double()
        e = ends.detach().cpu().double()
        step = max(1, s.shape[0] // 2000)
        mv = [[[round(x, 5) for x in a], [round(x, 5) for x in b]]
              for a, b in zip(s[::step].tolist(), e[::step].tolist())]
    cams = ([[round(v, 5) for v in c] for c in cameras] if cameras else [])

    data = json.dumps({"points": pts, "colors": cols, "axis": ax,
                       "motion": mv, "cameras": cams,
                       "extra": jsonable(extra or {})})
    html = _HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)
    path.write_text(html)
    return path


_HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{margin:0;background:#0d1117;color:#c9d1d9;font:13px system-ui,sans-serif}
 canvas{display:block}
 #hud{position:fixed;top:10px;left:10px;background:rgba(13,17,23,.86);
      padding:10px 12px;border:1px solid #30363d;border-radius:6px;max-width:330px}
 #hud b{color:#58a6ff} .k{color:#8b949e}
</style>
<div id="hud"><b>__TITLE__</b><div id="info"></div>
<div class="k" style="margin-top:6px">drag = rotate &middot; wheel = zoom &middot;
shift+drag = pan</div></div>
<canvas id="c"></canvas>
<script>
const D = __DATA__;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
let W, H; function size(){W=cv.width=innerWidth;H=cv.height=innerHeight;draw();}
addEventListener('resize', size);
let yaw=0.5, pitch=-0.25, dist=0, cx=0, cy=0, cz=0, panx=0, pany=0;
(function init(){
  const P=D.points; if(!P.length){return;}
  let mnx=1e9,mny=1e9,mnz=1e9,mxx=-1e9,mxy=-1e9,mxz=-1e9;
  for(const p of P){mnx=Math.min(mnx,p[0]);mxx=Math.max(mxx,p[0]);
    mny=Math.min(mny,p[1]);mxy=Math.max(mxy,p[1]);
    mnz=Math.min(mnz,p[2]);mxz=Math.max(mxz,p[2]);}
  cx=(mnx+mxx)/2; cy=(mny+mxy)/2; cz=(mnz+mxz)/2;
  dist=Math.max(mxx-mnx,mxy-mny,mxz-mnz)*1.8||5;
  const e=D.extra||{};
  document.getElementById('info').innerHTML =
    `${P.length} points${D.axis.length?' &middot; axis shown':''}` +
    Object.entries(e).map(([k,v])=>`<br><span class="k">${k}:</span> ${
      typeof v==='number'?(+v).toPrecision(4):v}`).join('');
})();
function proj(p){
  let x=p[0]-cx, y=p[1]-cy, z=p[2]-cz;
  let c=Math.cos(yaw), s=Math.sin(yaw); let X=c*x+s*z, Z=-s*x+c*z;
  c=Math.cos(pitch); s=Math.sin(pitch); let Y=c*y-s*Z; Z=s*y+c*Z;
  const f=(Math.min(W,H)*0.9)/dist, d=Z+dist*1.6;
  if(d<=0.01) return null;
  return [W/2+panx+X*f*dist/d/1.6, H/2+pany+Y*f*dist/d/1.6, d];
}
function draw(){
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);
  const P=D.points, C=D.colors;
  const order=[]; for(let i=0;i<P.length;i++){const q=proj(P[i]); if(q) order.push([q,i]);}
  order.sort((a,b)=>b[0][2]-a[0][2]);
  for(const [q,i] of order){
    const c=C[i]||[1,1,1];
    ctx.fillStyle=`rgb(${c[0]*255|0},${c[1]*255|0},${c[2]*255|0})`;
    const r=Math.max(0.7, 2.6*dist/q[2]/1.6);
    ctx.fillRect(q[0]-r/2,q[1]-r/2,r,r);
  }
  if(D.motion.length){ ctx.strokeStyle='rgba(255,210,60,.55)'; ctx.lineWidth=1;
    ctx.beginPath();
    for(const [a,b] of D.motion){const p=proj(a),q=proj(b);
      if(p&&q){ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);}}
    ctx.stroke(); }
  if(D.axis.length){ ctx.strokeStyle='#ff4d6d'; ctx.lineWidth=2.5;
    ctx.beginPath(); let started=false;
    for(const a of D.axis){const p=proj(a);
      if(p){ if(!started){ctx.moveTo(p[0],p[1]);started=true;}
             else ctx.lineTo(p[0],p[1]); }}
    ctx.stroke(); }
  if(D.cameras.length){ ctx.strokeStyle='#3fb950'; ctx.lineWidth=1.2;
    for(const c of D.cameras){const p=proj(c);
      if(p){ctx.strokeRect(p[0]-4,p[1]-4,8,8);}}}
}
let drag=false,lx=0,ly=0,sh=false;
cv.addEventListener('mousedown',e=>{drag=true;lx=e.clientX;ly=e.clientY;sh=e.shiftKey;});
addEventListener('mouseup',()=>drag=false);
addEventListener('mousemove',e=>{ if(!drag) return;
  const dx=e.clientX-lx, dy=e.clientY-ly; lx=e.clientX; ly=e.clientY;
  if(sh){panx+=dx;pany+=dy;} else {yaw+=dx*0.006; pitch+=dy*0.006;} draw(); });
cv.addEventListener('wheel',e=>{e.preventDefault();
  dist*=Math.exp(e.deltaY*0.0012); draw();},{passive:false});
size();
</script>
"""
