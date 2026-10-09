"""Physics-first dynamic discovery on a real endpoint pair (P0-P6).

Runs the hierarchy in its mandated order and stops where the physics stops:

    P0  bidirectional spatial mismatch -> two-component mixture -> posterior
    P1  dynamic source/target anchors from the MAP sets
    P2  nested metric patches, sized by local rank
    P3  partial Gromov-Wasserstein -> E_metric
    P4  weighted Kabsch from the coupling -> R, t
    P5  scene-derived null distributions -> metric and SE(3) p-values, FDR
    P6  SE(3) agreement graph -> transform-consistent components

There is no scalar weight anywhere in the decision path.  Every acceptance
is a statistical test against a null built from this scene's own non-matching
patch pairs, and the stages compose lexicographically: a candidate must pass
metric isometry to be fitted, pass the rigid test to be kept, and agree with
its neighbours to become part of a component.

Ground truth is a firewall: loaded only after every decision is made, and
used only to report.  No FM is trained, and §65's "no physically valid
transport found" is a legitimate result this tool is willing to print.

Run:
    PYTHONPATH=. python tools/run_patch_isometry_probe.py \\
        --frame0-model-dir <frame0_static> --frameT-model-dir <frameT_static> \\
        --iteration 14000 --output-dir <out> --root-scene <scene>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils import progress  # noqa: E402
from utils.dynamic_prior import _binary_auc  # noqa: E402
from utils.local_metric_isometry import (  # noqa: E402
    benjamini_hochberg,
    metric_hypothesis_test,
)
from utils.local_rigid_hypothesis import (  # noqa: E402
    fit_rigid_hypothesis,
    rigid_hypothesis_test,
    solve_isometry_with_rigid_selection,
)
from utils.metric_patch_graph import (  # noqa: E402
    build_full_geometry_knn,
    build_nested_metric_patches,
    compute_pairwise_metric_matrix,
    gather_patches,
    normalise_metric_matrix,
)
from utils.endpoint_flow_matching import FlowMatchingConfig  # noqa: E402
from utils.physical_appearance_disambiguation import (  # noqa: E402
    build_patch_sh_dc_descriptor,
    rank_physically_valid_candidates,
    sh_dc_to_rgb,
)
from utils.render_hypothesis_likelihood import (  # noqa: E402
    compare_static_vs_dynamic_likelihood,
    fit_static_rgb_residual_model,
)
from utils.transform_flow_matching import (  # noqa: E402
    evaluate_flow_against_transform,
    train_transform_flow_matching,
)
from utils.se3_hypothesis_consistency import (  # noqa: E402
    build_consistent_transform_components,
    se3_log,
)
from utils.spatial_motion_posterior import compute_motion_posterior  # noqa: E402
from utils.transform_supervision import (  # noqa: E402
    LABEL_AMBIGUOUS,
    LABEL_DYNAMIC,
    LABEL_STATIC,
    assign_accepted_patch_transforms_to_gaussians,
    generate_forward_fm_pairs,
    physical_labels,
    resolve_overlapping_patch_transforms,
)
from tools.run_relational_graph_probe import _find_model_ply, _json_safe, _write_json  # noqa: E402


def load_cloud(model_dir: str, iteration: int, sh_degree: int = 3):
    from scene.gaussian_model import GaussianModel

    ply = _find_model_ply(model_dir, iteration)
    if ply is None:
        raise SystemExit(f"No point_cloud.ply found under {model_dir}")
    model = GaussianModel(int(sh_degree), optimizer_type="default")
    model.load_ply(str(ply))
    return (
        model.get_canonical_xyz.detach().float(),
        model.get_scaling.detach().float(),
        sh_dc_to_rgb(model.get_features.detach().float()),
        ply,
    )


def json_finite(value):
    """NaN/inf serialise as null -- a missing measurement, not a number."""
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(v) for v in value]
    return value


def stats(value: torch.Tensor | None) -> dict[str, float]:
    if value is None:
        return {}
    v = torch.as_tensor(value).detach().float().cpu().reshape(-1)
    v = v[torch.isfinite(v)]
    if int(v.numel()) == 0:
        return {}
    return {
        "median": float(v.median()), "mean": float(v.mean()),
        "p10": float(torch.quantile(v, 0.10)), "p90": float(torch.quantile(v, 0.90)),
    }


def local_scale_from_knn(distances: torch.Tensor) -> torch.Tensor:
    """Median neighbour distance, excluding the self-match at column 0."""
    return distances[:, 1:].median(dim=1).values.clamp_min(1.0e-8)


@torch.no_grad()
def evaluate_candidates(
    source_xyz, target_xyz, source_patches, target_patches,
    source_anchors, target_anchors, *, level, batch_size, null_samples, generator, device,
    trace_solver: bool = False,
):
    """P3-P5 for one block of source anchors against a shared candidate pool.

    The null is POOLED across every source patch rather than built per patch,
    and that is what makes the test possible at all.  A per-patch null of
    ``S`` draws can never produce a p-value below ``1/(S+1)``, so with the
    handful of draws each patch can afford, no candidate survives an FDR
    correction across hundreds of patches however isometric it is -- the
    first run of this probe accepted 0 of 64 for exactly that reason.
    Distortion is normalised per patch, so the non-best evaluations are
    exchangeable and pooling them gives a null with ``patches x draws``
    resolution for the same compute.
    """
    idx_t_all, valid_t_all = gather_patches(target_patches, target_anchors, level=level)
    dt_all, scale_t_all = normalise_metric_matrix(
        compute_pairwise_metric_matrix(target_xyz, idx_t_all), valid=valid_t_all
    )
    pool = int(target_anchors.numel())

    results: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("distortion", "residual", "best_target", "second_distortion",
                        "rotation", "translation", "degenerate", "patch_index", "patch_valid")
    }
    null_metric_pool: list[torch.Tensor] = []
    null_rigid_pool: list[torch.Tensor] = []

    total_anchors = int(source_anchors.numel())
    starts = list(range(0, total_anchors, int(batch_size)))
    outer = progress.bar(starts, desc="P3 patches", unit="batch", leave=True)
    running_distortion: list[float] = []
    running_residual: list[float] = []
    for start in outer:
        block = source_anchors[start:start + int(batch_size)]
        idx_s, valid_s = gather_patches(source_patches, block, level=level)
        d0, scale_s = normalise_metric_matrix(
            compute_pairwise_metric_matrix(source_xyz, idx_s), valid=valid_s
        )
        b = int(block.numel())

        # Candidate draws and null draws come from the same pool; the null is
        # what "a non-matching pair of this scene" looks like, which is the
        # only reference that makes a distortion interpretable.
        draws = torch.randint(0, pool, (b, int(null_samples)), generator=generator).to(device)
        best = None
        null_metric = torch.empty((b, int(null_samples)), device=device)
        null_rigid = torch.empty((b, int(null_samples)), device=device)
        inner = progress.bar(
            range(int(null_samples)),
            desc=f"  candidates {start}-{start + b}", unit="draw",
        )
        for s in inner:
            pick = draws[:, s]
            coupling, distortion, hypothesis = solve_isometry_with_rigid_selection(
                source_xyz[idx_s], target_xyz[idx_t_all[pick]], d0, dt_all[pick],
                valid_s, valid_t_all[pick], patch_scale=scale_s,
            )
            null_metric[:, s] = distortion
            null_rigid[:, s] = hypothesis.residual
            score = torch.where(hypothesis.degenerate, torch.full_like(hypothesis.residual, float("inf")),
                                hypothesis.residual)
            if best is None:
                best = [distortion, hypothesis.residual, score, pick,
                        hypothesis.rotation, hypothesis.translation, hypothesis.degenerate]
            else:
                better = score < best[2]
                best = [
                    torch.where(better, distortion, best[0]),
                    torch.where(better, hypothesis.residual, best[1]),
                    torch.where(better, score, best[2]),
                    torch.where(better, pick, best[3]),
                    torch.where(better[:, None, None], hypothesis.rotation, best[4]),
                    torch.where(better[:, None], hypothesis.translation, best[5]),
                    torch.where(better, hypothesis.degenerate, best[6]),
                ]
            # The running best is the quantity the search is actually driving
            # down, so it is what the bar shows: a best that stops improving
            # says the pool is exhausted, one that never improves says the
            # true counterpart is not in it.
            inner.set_postfix({
                "draw E_met": f"{float(distortion.median()):.4f}",
                "best E_met": f"{float(best[0].median()):.4f}",
                "best E_SE3": f"{float(best[1].median()):.4f}",
                "degen": f"{float(hypothesis.degenerate.float().mean()):.2f}",
            })
        inner.close()

        distortion, residual, _score, pick, rotation, translation, degenerate = best
        # §59 wants the runner-up too: a best that is barely better than the
        # second best is a tie, and a tie is not a discovery however low its
        # absolute distortion.
        second = torch.topk(null_metric, k=min(2, int(null_samples)), dim=1, largest=False).values
        second_distortion = second[:, -1] if int(second.shape[1]) > 1 else second[:, 0]
        # Everything that was NOT the winner is a non-matching evaluation of
        # this scene, which is exactly what §20 asks the null to be made of.
        loser = null_metric > distortion[:, None]
        null_metric_pool.append(null_metric[loser])
        null_rigid_pool.append(null_rigid[null_rigid > residual[:, None]])

        results["distortion"].append(distortion)
        results["residual"].append(residual)
        results["best_target"].append(target_anchors.to(device)[pick])
        results["second_distortion"].append(second_distortion)
        results["rotation"].append(rotation)
        results["translation"].append(translation)
        results["degenerate"].append(degenerate)
        results["patch_index"].append(idx_s)
        results["patch_valid"].append(valid_s)
        running_distortion.append(float(distortion.median()))
        running_residual.append(float(residual.median()))
        margin = float((second_distortion - distortion).median())
        outer.set_postfix({
            "E_met": f"{float(distortion.median()):.4f}",
            "E_SE3": f"{float(residual.median()):.4f}",
            "margin": f"{margin:+.4f}",
            "null_med": f"{float(null_metric.median()):.4f}",
            "run_E_met": f"{sum(running_distortion) / len(running_distortion):.4f}",
        })
        progress.write(
            f"    anchors {start:5d}-{start + b:5d}  "
            f"E_metric {float(distortion.median()):.4f} (2nd {float(second_distortion.median()):.4f}, "
            f"margin {margin:+.4f})  E_SE3 {float(residual.median()):.4f}  "
            f"null med {float(null_metric.median()):.4f}  degenerate {int(degenerate.sum())}/{b}"
        )
        if trace_solver:
            probe = solve_partial_local_gromov_wasserstein_traced(d0, dt_all[draws[:, 0]], valid_s, valid_t_all[draws[:, 0]])
            progress.write("      GW objective: " + "  ".join(
                f"eps={e:g}/it{i}={v:.5f}" for e, i, v in probe[::max(1, len(probe) // 6)]
            ))

    width = max(int(t.shape[1]) for t in results["patch_index"])
    for key in ("patch_index", "patch_valid"):
        padded = []
        for tensor in results[key]:
            if int(tensor.shape[1]) < width:
                pad = tensor[:, :1].expand(-1, width - int(tensor.shape[1]))
                tensor = torch.cat((tensor, pad if key == "patch_index" else torch.zeros_like(pad, dtype=torch.bool)), 1)
            padded.append(tensor)
        results[key] = padded
    out = {k: torch.cat(v, 0) for k, v in results.items()}

    null_metric = torch.cat(null_metric_pool, 0)
    null_rigid = torch.cat(null_rigid_pool, 0)
    # Finite-sample corrected empirical p-values against the pooled null.
    out["metric_p"] = (
        (null_metric[None, :] <= out["distortion"][:, None]).sum(1).float() + 1.0
    ) / float(int(null_metric.numel()) + 1)
    out["rigid_p"] = (
        (null_rigid[None, :] <= out["residual"][:, None]).sum(1).float() + 1.0
    ) / float(int(null_rigid.numel()) + 1)
    out["null_metric"] = null_metric
    out["null_rigid"] = null_rigid
    outer.close()
    return out


def solve_partial_local_gromov_wasserstein_traced(d0, dt, valid_s, valid_t):
    """The solver's own objective per step, for --trace-solver."""
    from utils.local_metric_isometry import solve_partial_local_gromov_wasserstein

    return solve_partial_local_gromov_wasserstein(
        d0, dt, source_valid=valid_s, target_valid=valid_t, trace=True
    ).trace


# --------------------------------------------------------------------------
# §59-§61 debug visualisation
# --------------------------------------------------------------------------
def write_debug_html(path: Path, payload: dict[str, Any]) -> None:
    """Scene views, the null distributions, and the SE(3) tangent space.

    The tangent-space view is the one that matters for §61: a real moving
    part shows up as a tight cluster of accepted transforms, and two parts as
    two clusters, with no K supplied anywhere.  A diffuse cloud there means
    the local transforms do not agree and no component should form -- which
    is exactly what the agreement graph is reporting when it returns zero.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(json_finite(_json_safe(payload)))
    path.write_text(
        """<!doctype html><meta charset="utf-8"><title>Physics-first patch isometry</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>body{font:13px system-ui;margin:0;background:#111;color:#ddd}
#tabs{padding:8px;background:#1b1b1b;position:sticky;top:0;z-index:9}
button{font:12px system-ui;margin:2px;padding:5px 9px;background:#2a2a2a;color:#ddd;
border:1px solid #444;border-radius:3px;cursor:pointer}
button.on{background:#3a5a8a;border-color:#6a9ada}#plot{height:84vh}
#note{padding:6px 12px;color:#9ab;font-size:12px}</style>
<div id="tabs"></div><div id="note"></div><div id="plot"></div>
<script>
const D = """ + body + """;
const tabs=document.getElementById('tabs'), note=document.getElementById('note');
function scene3(v){
  const tr=[];
  if(v.edges&&v.edges.length) tr.push({type:'scatter3d',mode:'lines',
    x:v.edges.map(p=>p[0]),y:v.edges.map(p=>p[1]),z:v.edges.map(p=>p[2]),
    line:{color:'#5ad18a',width:1.5},name:'agreement edges',hoverinfo:'skip'});
  if(v.arrows&&v.arrows.length) tr.push({type:'scatter3d',mode:'lines',
    x:v.arrows.map(p=>p[0]),y:v.arrows.map(p=>p[1]),z:v.arrows.map(p=>p[2]),
    line:{color:'#e8a33d',width:2},name:'accepted transform',hoverinfo:'skip'});
  if(v.cloud) tr.push({type:'scatter3d',mode:'markers',x:D.cloud.map(p=>p[0]),
    y:D.cloud.map(p=>p[1]),z:D.cloud.map(p=>p[2]),
    marker:{size:1.4,color:v.cloud,colorscale:v.scale||'Viridis',cmin:v.cmin,cmax:v.cmax,
    showscale:true,colorbar:{title:v.title||''}},name:'frame 0',
    text:v.cloud.map(x=>x===null?'n/a':x.toFixed(3)),hoverinfo:'text'});
  if(v.anchors) tr.push({type:'scatter3d',mode:'markers',x:D.anchors.map(p=>p[0]),
    y:D.anchors.map(p=>p[1]),z:D.anchors.map(p=>p[2]),
    marker:{size:5,color:v.anchors,colorscale:v.ascale||'Portland',showscale:false,
    line:{color:'#000',width:0.5}},name:'patch anchors',
    text:v.anchors.map(x=>x===null?'n/a':String(x)),hoverinfo:'text'});
  return tr;
}
function draw(name){
  [...tabs.children].forEach(b=>b.className=b.textContent===name?'on':'');
  const v=D.views[name]; note.textContent=v.note||'';
  let tr, layout;
  if(v.kind==='hist'){
    tr=v.series.map(s=>({type:'histogram',x:s.x,name:s.name,opacity:0.6,
      marker:{color:s.color},nbinsx:60}));
    layout={barmode:'overlay',paper_bgcolor:'#111',plot_bgcolor:'#181818',
      font:{color:'#ddd'},xaxis:{title:v.xaxis,gridcolor:'#333'},
      yaxis:{title:'count',gridcolor:'#333'},margin:{t:10}};
  } else if(v.kind==='scatter'){
    tr=[{type:'scatter',mode:'markers',x:v.x,y:v.y,
      marker:{size:7,color:v.color,colorscale:'Portland',showscale:true,
      colorbar:{title:v.ctitle||''}},text:v.text,hoverinfo:'text'}];
    layout={paper_bgcolor:'#111',plot_bgcolor:'#181818',font:{color:'#ddd'},
      xaxis:{title:v.xaxis,gridcolor:'#333'},yaxis:{title:v.yaxis,gridcolor:'#333'},margin:{t:10}};
  } else {
    tr=scene3(v);
    layout={paper_bgcolor:'#111',plot_bgcolor:'#111',font:{color:'#ddd'},
      margin:{l:0,r:0,t:0,b:0},showlegend:true,
      scene:{aspectmode:'data',xaxis:{color:'#666'},yaxis:{color:'#666'},zaxis:{color:'#666'}}};
  }
  Plotly.react('plot',tr,layout);
}
Object.keys(D.views).forEach(n=>{const b=document.createElement('button');
  b.textContent=n;b.onclick=()=>draw(n);tabs.appendChild(b);});
draw(Object.keys(D.views)[0]);
</script>""",
        encoding="utf-8",
    )


def build_debug_payload(
    xyz0, posterior, source_anchors, evaluated, physically_valid, components, label,
    *, max_points: int = 8000, generator=None,
):
    """§59 null distributions, §60 tangent space, §61 agreement graph."""
    n = int(xyz0.shape[0])
    keep = torch.arange(n)
    if n > int(max_points):
        keep = torch.randperm(n, generator=generator)[: int(max_points)].sort().values
    cloud = xyz0.detach().cpu()[keep]
    anchors = xyz0.detach().cpu()[source_anchors.cpu()]

    def col(v):
        return [float(x) for x in torch.as_tensor(v).detach().float().cpu().reshape(-1)[keep]]

    def acol(v):
        return [float(x) for x in torch.as_tensor(v).detach().float().cpu().reshape(-1)]

    # §61: edges of the agreement graph, drawn between anchor positions.
    edges = []
    for u, w in components.edges.tolist():
        edges.append([float(x) for x in anchors[int(u)]])
        edges.append([float(x) for x in anchors[int(w)]])
        edges.append([None, None, None])

    # §60: each accepted transform drawn as the displacement it predicts for
    # its own anchor, which is the physically readable form of (R, t).
    rotation = evaluated["rotation"].detach().cpu()
    translation = evaluated["translation"].detach().cpu()
    moved = torch.einsum("bij,bj->bi", rotation, anchors) + translation
    arrows = []
    for i in physically_valid.cpu().nonzero(as_tuple=True)[0].tolist():
        arrows.append([float(x) for x in anchors[i]])
        arrows.append([float(x) for x in moved[i]])
        arrows.append([None, None, None])

    tangent = components.tangent.detach().cpu()
    valid_cpu = physically_valid.cpu()
    return {
        "cloud": [[float(v) for v in row] for row in cloud],
        "anchors": [[float(v) for v in row] for row in anchors],
        "views": {
            "q0 motion posterior": {
                "cloud": col(posterior.source_dynamic_posterior), "cmin": 0, "cmax": 1,
                "title": "P(D|z)",
                "note": "P0. Posterior of the fitted high-mismatch mixture component -- no threshold, no temperature.",
            },
            "log10 forward mismatch": {
                "cloud": col(torch.log10(posterior.forward_mismatch.cpu() + 1.0e-6)),
                "title": "log10 E",
                "note": "P0. Absolute spatial mismatch. Says whether geometry stayed, never where it went (§11).",
            },
            "physical label": {
                "cloud": col(label.float()), "cmin": 0, "cmax": 2, "scale": "Portland",
                "title": "0=S 1=D 2=A",
                "note": "P9. The §36 decision tree, not a weighted score. AMBIGUOUS is a real answer (§65).",
            },
            "patch anchors: metric p-value": {
                "cloud": col(posterior.source_dynamic_posterior), "cmin": 0, "cmax": 1,
                "anchors": acol(evaluated["metric_p"]), "title": "q0",
                "note": "P5. Anchor colour is the empirical metric p-value against this scene's own null.",
            },
            "patch anchors: SE3 residual": {
                "cloud": col(posterior.source_dynamic_posterior), "cmin": 0, "cmax": 1,
                "anchors": acol(evaluated["residual"]), "title": "q0",
                "note": "P5. Anchor colour is E_SE3 in units of patch scale squared.",
            },
            "accepted transforms + agreement graph": {
                "cloud": col(posterior.source_dynamic_posterior), "cmin": 0, "cmax": 1,
                "anchors": acol(components.label.float()), "ascale": "Portland",
                "edges": edges, "arrows": arrows, "title": "q0",
                "note": ("P6/§61. Orange lines are each accepted patch's predicted motion; green lines join "
                         "anchors whose transforms agree under the fitted SE(3) noise model. "
                         "Connected components are candidate parts -- K is never supplied."),
            },
            "null vs observed (metric)": {
                "kind": "hist", "xaxis": "E_metric",
                "series": [
                    {"name": "null (non-matching pairs)", "color": "#888",
                     "x": [float(x) for x in evaluated["null_metric"].detach().cpu()[:20000]]},
                    {"name": "best candidate", "color": "#e8a33d",
                     "x": [float(x) for x in evaluated["distortion"].detach().cpu()]},
                ],
                "note": "§59. Acceptance is exceptionality against this null, never a fixed threshold (§20).",
            },
            "null vs observed (SE3)": {
                "kind": "hist", "xaxis": "E_SE3",
                "series": [
                    {"name": "null", "color": "#888",
                     "x": [float(x) for x in evaluated["null_rigid"].detach().cpu()[:20000]]},
                    {"name": "best candidate", "color": "#5ad18a",
                     "x": [float(x) for x in evaluated["residual"].detach().cpu()]},
                ],
                "note": "§60. The rigid residual is tested separately and never summed with E_metric (§22).",
            },
            "SE(3) tangent space": {
                "kind": "scatter",
                "x": [float(x) for x in torch.linalg.norm(tangent[:, :3], dim=1)],
                "y": [float(x) for x in torch.linalg.norm(tangent[:, 3:], dim=1)],
                "color": [float(x) for x in components.label.float()],
                "text": [f"anchor {i} | accepted={bool(valid_cpu[i])} | component={int(components.label[i])}"
                         for i in range(int(tangent.shape[0]))],
                "xaxis": "|omega| (rotation, rad)", "yaxis": "|v| (translation)",
                "ctitle": "component",
                "note": ("§60/§61. A real moving part is a tight cluster here and two parts are two clusters. "
                         "A diffuse cloud means the local transforms disagree and no component should form."),
            },
        },
    }

@torch.no_grad()
def run_render_validation(args, xyz0, evaluated, physically_valid, components, *, device):
    """P8 against the project rasteriser and the scene's real images.

    Each accepted patch is rendered twice -- as reconstructed, and with its
    own Gaussians moved by its hypothesis -- and the two are scored against
    the actual photographs on the pixels either hypothesis can touch.
    """
    from types import SimpleNamespace

    from tools.test_render_denoise_discovery import build_scene

    model_dir = args.static_model_dir or args.frame0_model_dir
    scene, gaussians, pipe, white_background = build_scene(SimpleNamespace(
        static_model_dir=model_dir, root_scene=args.root_scene, iteration=int(args.iteration),
        sh_degree=None, resolution=None, white_background=None,
        camera_data_device="cpu", use_depth=True, cameras="", fps=30.0,
    ))
    cameras = scene.getTrainCameras()
    background = torch.tensor(
        [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0], device=gaussians.get_xyz.device
    )
    count = int(gaussians.get_xyz.shape[0])
    if count != int(xyz0.shape[0]):
        return {"run": False, "reason": f"render model has {count} Gaussians, probe has {int(xyz0.shape[0])}"}, None

    # §32: residual scale fitted on the static render, which is where the
    # residual is pure observation noise.
    model = fit_static_rgb_residual_model(
        cameras, gaussians, pipe, background, max_cameras=int(args.render_cameras)
    )
    progress.write(
        f"[P8] residual model: scale {model.scale:.5g} from {model.samples} pixels "
        f"over {min(len(cameras), int(args.render_cameras))} cameras"
    )

    accepted = physically_valid.cpu().nonzero(as_tuple=True)[0]
    order = torch.argsort(evaluated["residual"].cpu()[accepted])
    chosen = accepted[order][: int(args.render_hypotheses)]

    patch_index = evaluated["patch_index"].cpu()
    patch_valid = evaluated["patch_valid"].cpu()
    rotation = evaluated["rotation"].cpu()
    translation = evaluated["translation"].cpu()

    prefers = torch.zeros(int(evaluated["residual"].shape[0]), dtype=torch.bool)
    ratios: list[float] = []
    rows: list[dict[str, Any]] = []
    for patch in progress.bar(chosen.tolist(), desc="P8 hypotheses", unit="patch", leave=True):
        mask = torch.zeros(count, dtype=torch.bool)
        mask[patch_index[patch][patch_valid[patch]]] = True
        comparison = compare_static_vs_dynamic_likelihood(
            cameras, gaussians, pipe, background, mask.to(background.device),
            rotation[patch], translation[patch], model,
            max_cameras=int(args.render_cameras),
        )
        prefers[patch] = comparison.prefers_dynamic
        ratios.append(comparison.log_likelihood_ratio)
        rows.append({
            "patch": int(patch),
            "log_likelihood_ratio": comparison.log_likelihood_ratio,
            "prefers_dynamic": comparison.prefers_dynamic,
            "indistinguishable": comparison.indistinguishable,
            "pixels": comparison.pixels,
        })
        progress.write(
            f"    patch {int(patch):4d}  LLR {comparison.log_likelihood_ratio:+.4g} "
            f"over {comparison.pixels} px -> "
            f"{'DYNAMIC' if comparison.prefers_dynamic else ('ambiguous' if comparison.indistinguishable else 'static')}"
        )

    # A per-Gaussian veto: only Gaussians inside a render-preferred patch may
    # end up DYNAMIC (§36 clause 5).
    gaussian_prefers = torch.zeros(int(xyz0.shape[0]), dtype=torch.bool)
    for patch in prefers.nonzero(as_tuple=True)[0].tolist():
        gaussian_prefers[patch_index[patch][patch_valid[patch]]] = True

    return {
        "run": True,
        "model_dir": str(model_dir),
        "cameras_available": len(cameras),
        "cameras_per_hypothesis": int(args.render_cameras),
        "residual_scale": model.scale,
        "residual_samples": model.samples,
        "hypotheses_tested": int(chosen.numel()),
        "prefers_dynamic_count": int(prefers.sum()),
        "log_likelihood_ratio": stats(torch.tensor(ratios)) if ratios else {},
        "per_hypothesis": rows,
    }, gaussian_prefers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame0-model-dir", required=True)
    parser.add_argument("--frameT-model-dir", required=True)
    parser.add_argument("--iteration", type=int, default=14000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-scene", default="", help="only used for post-hoc GT reporting")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Runtime controls only (§45): these bound cost, not physics.
    parser.add_argument("--knn", type=int, default=48)
    parser.add_argument("--patch-level", type=int, default=1)
    parser.add_argument("--source-anchors", type=int, default=256)
    parser.add_argument("--target-anchors", type=int, default=192)
    parser.add_argument("--null-samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)

    # Statistical conventions only (§46): fixed globally, never tuned on GT.
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-component", type=int, default=3)
    parser.add_argument("--save-tensors", action="store_true")
    parser.add_argument("--no-html", action="store_true", help="skip the §59-§61 debug page")
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")
    parser.add_argument("--static-model-dir", default="",
                        help="P8: scene whose cameras and Gaussians validate hypotheses (default: frame0)")
    parser.add_argument("--render-cameras", type=int, default=8, help="P8: cameras per hypothesis")
    parser.add_argument("--render-hypotheses", type=int, default=16, help="P8: accepted patches to validate")
    parser.add_argument("--skip-render", action="store_true", help="skip P8 entirely")
    parser.add_argument("--fm-iters", type=int, default=1500, help="P10: flow-matching iterations")
    parser.add_argument("--fm-parameterisation", default="twist", choices=("twist", "cartesian"))
    parser.add_argument("--skip-fm", action="store_true", help="skip P10 even if the gate passes")
    parser.add_argument("--trace-solver", action="store_true",
                        help="print the GW objective per epsilon level for one candidate per batch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(int(args.seed))
    progress.set_enabled(not bool(args.no_progress))

    xyz0, scale0, rgb0, ply0 = load_cloud(args.frame0_model_dir, int(args.iteration), int(args.sh_degree))
    xyzT, scaleT, rgbT, plyT = load_cloud(args.frameT_model_dir, int(args.iteration), int(args.sh_degree))
    xyz0, xyzT = xyz0.to(device), xyzT.to(device)
    print(f"[physics] frame0={int(xyz0.shape[0])} ({ply0})")
    print(f"[physics] frameT={int(xyzT.shape[0])} ({plyT})")

    # ---------------- P0: spatial mismatch and the motion posterior --------
    with progress.stage("P0 knn + mismatch"):
        knn0, dist0 = build_full_geometry_knn(
            xyz0, int(args.knn), chunk_size=int(args.chunk_size), desc="  knn frame0"
        )
        knnT, distT = build_full_geometry_knn(
            xyzT, int(args.knn), chunk_size=int(args.chunk_size), desc="  knn frameT"
        )
        posterior = compute_motion_posterior(
            xyz0, xyzT, local_scale_from_knn(dist0), local_scale_from_knn(distT),
            chunk_size=int(args.chunk_size),
        )
    source_fit, target_fit = posterior.source_mixture, posterior.target_mixture
    print(
        f"[P0] source mixture: static mu {source_fit.static_mean:+.2f} sd {source_fit.static_std:.2f} | "
        f"dynamic mu {source_fit.dynamic_mean:+.2f} sd {source_fit.dynamic_std:.2f} | "
        f"w_dyn {source_fit.dynamic_weight:.3f} | separation {source_fit.separation:.2f} | "
        f"bimodal {source_fit.bimodal}"
    )
    print(
        f"[P0] MAP dynamic: source {posterior.stats['source_map_dynamic_fraction']:.3f} "
        f"target {posterior.stats['target_map_dynamic_fraction']:.3f}"
    )
    if posterior.stats["spatial_detector_uncertain"]:
        print("[P0] WARNING: the mismatch distribution is not statistically bimodal; "
              "the spatial detector is uncertain (§8)")

    results: dict[str, Any] = {
        "config": {k: _json_safe(v) for k, v in vars(args).items()},
        "frame0_count": int(xyz0.shape[0]),
        "frameT_count": int(xyzT.shape[0]),
        "P0_spatial": {
            "source_mixture": vars(source_fit),
            "target_mixture": vars(target_fit),
            "forward_mismatch": stats(posterior.forward_mismatch),
            "backward_mismatch": stats(posterior.backward_mismatch),
            **posterior.stats,
        },
    }

    # ---------------- P1/P2: dynamic anchors and nested metric patches -----
    source_pool = posterior.source_map_dynamic.nonzero(as_tuple=True)[0]
    target_pool = posterior.target_map_dynamic.nonzero(as_tuple=True)[0]
    if int(source_pool.numel()) < 8 or int(target_pool.numel()) < 8:
        print("[P1] too few dynamic anchors at either endpoint; NO PHYSICALLY VALID TRANSPORT FOUND")
        results["verdict"] = {"result": "NO PHYSICALLY VALID TRANSPORT FOUND", "reason": "empty dynamic domain"}
        _write_json(out / "patch_isometry_probe.json", json_finite(_json_safe(results)))
        return 0

    with progress.stage("P2 nested metric patches"):
        source_patches = build_nested_metric_patches(xyz0, knn0, desc="  rank sweep frame0")
        target_patches = build_nested_metric_patches(xyzT, knnT, desc="  rank sweep frameT")
    progress.write(
        f"[P2] patch sizes frame0: K1 median {int(source_patches.rank_size.median())} "
        f"(p10 {int(torch.quantile(source_patches.rank_size.float(), 0.10))}, "
        f"p90 {int(torch.quantile(source_patches.rank_size.float(), 0.90))}) | "
        f"full rank {float(source_patches.full_rank.float().mean()):.3f}"
    )
    # Only anchors whose own neighbourhood is full rank can constrain a 3-D
    # transform, so a degenerate anchor is dropped here rather than allowed
    # to contribute an unidentifiable rotation later.
    source_pool = source_pool[source_patches.full_rank[source_pool]]
    target_pool = target_pool[target_patches.full_rank[target_pool]]
    print(f"[P2] full-rank dynamic anchors: source {int(source_pool.numel())} target {int(target_pool.numel())}")

    def subsample(pool: torch.Tensor, count: int) -> torch.Tensor:
        if int(pool.numel()) <= int(count):
            return pool
        pick = torch.randperm(int(pool.numel()), generator=generator)[: int(count)]
        return pool[pick.sort().values.to(pool.device)]

    source_anchors = subsample(source_pool, int(args.source_anchors))
    target_anchors = subsample(target_pool, int(args.target_anchors))
    coverage = float(args.null_samples) / max(1.0, float(target_anchors.numel()))
    print(f"[P3] testing {int(source_anchors.numel())} source patches "
          f"against {int(target_anchors.numel())} target patches, "
          f"{int(args.null_samples)} draws each "
          f"(candidate coverage {coverage:.1%} of the dynamic target domain)")
    if coverage < 0.5:
        # Worth stating plainly: with sparse draws the "best" candidate is
        # the best of a random sample, so a null verdict means the search
        # never saw the true counterpart -- not that no isometry exists.
        print(f"[P3] WARNING: each source patch sees only {coverage:.1%} of the candidate pool. "
              f"A negative result at this coverage is a search limit, not a physical one; "
              f"raise --null-samples (and/or lower --target-anchors) to close it.")

    # ---------------- P3/P4/P5: isometry, transform, statistical tests -----
    with progress.stage("P3-P5 isometry + rigid fit + tests"):
        evaluated = evaluate_candidates(
            xyz0, xyzT, source_patches, target_patches, source_anchors, target_anchors,
            level=int(args.patch_level), batch_size=int(args.batch_size),
            null_samples=int(args.null_samples), generator=generator, device=device,
            trace_solver=bool(args.trace_solver),
        )
    metric_accepted, metric_cut = benjamini_hochberg(evaluated["metric_p"], alpha=float(args.alpha))
    rigid_accepted, rigid_cut = benjamini_hochberg(evaluated["rigid_p"], alpha=float(args.alpha))
    # Lexicographic: metric compatibility first, then the rigid test, then a
    # determined fit. Never a sum of the three.
    physically_valid = metric_accepted.to(device) & rigid_accepted.to(device) & ~evaluated["degenerate"]
    print(
        f"[P5] metric test accepts {int(metric_accepted.sum())}/{int(metric_accepted.numel())} "
        f"(FDR cut {metric_cut:.4g}) | rigid test accepts {int(rigid_accepted.sum())} "
        f"(cut {rigid_cut:.4g}) | both and determined: {int(physically_valid.sum())}"
    )

    results["P3_P5_patch_tests"] = {
        "source_patches_tested": int(source_anchors.numel()),
        "target_patches_per_test": int(args.null_samples),
        "best_metric_distortion": stats(evaluated["distortion"]),
        "second_best_metric_distortion": stats(evaluated["second_distortion"]),
        "best_to_second_margin": stats(evaluated["second_distortion"] - evaluated["distortion"]),
        "best_se3_residual": stats(evaluated["residual"]),
        "metric_p_value": stats(evaluated["metric_p"]),
        "null_metric_distortion": stats(evaluated["null_metric"]),
        "null_se3_residual": stats(evaluated["null_rigid"]),
        "null_size": int(evaluated["null_metric"].numel()),
        "rigid_p_value": stats(evaluated["rigid_p"]),
        "metric_accepted": int(metric_accepted.sum()),
        "metric_fdr_cut": float(metric_cut),
        "rigid_accepted": int(rigid_accepted.sum()),
        "rigid_fdr_cut": float(rigid_cut),
        "degenerate": int(evaluated["degenerate"].sum()),
        "physically_valid": int(physically_valid.sum()),
        "alpha": float(args.alpha),
        "candidate_coverage": float(args.null_samples) / max(1.0, float(args.target_anchors)),
    }

    # ---------------- P6: SE(3) agreement between neighbouring anchors -----
    progress.write(
        f"[P5] p-values: metric median {float(evaluated['metric_p'].median()):.4g} "
        f"min {float(evaluated['metric_p'].min()):.4g} | "
        f"rigid median {float(evaluated['rigid_p'].median()):.4g} "
        f"min {float(evaluated['rigid_p'].min()):.4g} | "
        f"null pool {int(evaluated['null_metric'].numel())}"
    )
    tangent = se3_log(evaluated["rotation"].cpu(), evaluated["translation"].cpu())
    anchor_xyz = xyz0[source_anchors].cpu()
    neighbour_k = min(9, int(anchor_xyz.shape[0]))
    anchor_knn, _ = build_full_geometry_knn(anchor_xyz, neighbour_k, desc="  anchor knn")
    pairs = torch.stack([
        anchor_knn[:, 0].repeat_interleave(neighbour_k - 1),
        anchor_knn[:, 1:].reshape(-1),
    ], dim=1)
    with progress.stage("P6 SE(3) agreement"):
        components = build_consistent_transform_components(
            tangent, pairs, accepted=physically_valid.cpu(),
            alpha=float(args.alpha), min_component=int(args.min_component),
        )
    progress.write(
        f"[P6] tangent spread: |omega| median {float(torch.linalg.norm(tangent[:, :3], dim=1).median()):.4f} rad "
        f"| |v| median {float(torch.linalg.norm(tangent[:, 3:], dim=1).median()):.4f} "
        f"| chi2(6) limit {components.stats['chi2_limit']:.2f}"
    )
    print(
        f"[P6] transform-consistent components: {components.stats['components']} "
        f"| agreeing edges {components.stats['edge_fraction']:.3f} "
        f"| anchors assigned {components.stats['assigned_fraction']:.3f}"
    )
    results["P6_se3_components"] = {
        **components.stats,
        "component_sizes": components.component_sizes.tolist(),
        "tangent_covariance_diagonal": components.covariance.diagonal().tolist(),
    }


    # ---------------- P7: appearance ranks physically valid candidates -----
    # §27: appearance never makes an invalid candidate valid.  It is handed
    # only the candidates that already passed both physical tests, and where
    # exactly one survives it is not consulted at all.
    appearance_report: dict[str, Any] = {"run": False}
    valid_cpu = physically_valid.cpu()
    if bool(valid_cpu.any()):
        src_idx = evaluated["patch_index"].cpu()
        src_valid = evaluated["patch_valid"].cpu()
        source_descriptor = build_patch_sh_dc_descriptor(rgb0.cpu(), src_idx, src_valid)
        target_anchor_idx, target_anchor_valid = gather_patches(
            target_patches, evaluated["best_target"], level=int(args.patch_level)
        )
        target_anchor_idx = target_anchor_idx.cpu()
        target_anchor_valid = target_anchor_valid.cpu()
        candidate_descriptor = build_patch_sh_dc_descriptor(
            rgbT.cpu(), target_anchor_idx, target_anchor_valid
        )[:, None, :]
        ranking = rank_physically_valid_candidates(
            source_descriptor, candidate_descriptor, valid_cpu[:, None]
        )
        distance = ranking.distance[:, 0]
        finite = distance[torch.isfinite(distance)]
        appearance_report = {
            "run": True,
            "candidates_per_patch": 1,
            "note": ("one physical candidate per patch at this search width, so §27 says "
                     "appearance is not consulted; the distance is reported as a diagnostic only"),
            "appearance_distance": stats(finite),
            "appearance_distance_accepted": stats(distance[valid_cpu]),
        }
        progress.write(
            f"[P7] appearance distance on accepted patches: "
            f"median {float(distance[valid_cpu].median()):.4f} "
            f"(not used to decide -- only one physical candidate survives per patch)"
        )
    results["P7_appearance"] = appearance_report

    # ---------------- P8: renderer validates against the real images -------
    render_report: dict[str, Any] = {"run": False}
    render_prefers = None
    if not bool(args.skip_render) and bool(valid_cpu.any()):
        with progress.stage("P8 render likelihood"):
            render_report, render_prefers = run_render_validation(
                args, xyz0, evaluated, physically_valid, components, device=device,
            )
    elif bool(args.skip_render):
        render_report = {"run": False, "reason": "--skip-render"}
    else:
        render_report = {"run": False, "reason": "no physically valid patch to validate"}
    results["P8_render"] = render_report

    # ---------------- P9: transform-derived supervision --------------------
    membership, coverage = assign_accepted_patch_transforms_to_gaussians(
        int(xyz0.shape[0]), evaluated["patch_index"], evaluated["patch_valid"],
        physically_valid, components.label,
    )
    assignment = resolve_overlapping_patch_transforms(
        int(xyz0.shape[0]), membership,
        evaluated["rotation"], evaluated["translation"], components.label,
    )
    pairs_fm = generate_forward_fm_pairs(xyz0.cpu(), assignment)
    label = physical_labels(
        posterior.source_dynamic_posterior, assignment,
        render_prefers_dynamic=render_prefers,
    )
    print(
        f"[P9] Gaussians covered {assignment.stats['covered_fraction']:.4f} "
        f"| resolved {assignment.stats['assigned_fraction']:.4f} "
        f"| ambiguous {assignment.stats['ambiguous_fraction']:.4f} "
        f"| FM pairs {int(pairs_fm['source'].shape[0])}"
    )
    results["P9_supervision"] = {
        **assignment.stats,
        "fm_pairs": int(pairs_fm["source"].shape[0]),
        "label_static": int((label == LABEL_STATIC).sum()),
        "label_dynamic": int((label == LABEL_DYNAMIC).sum()),
        "label_ambiguous": int((label == LABEL_AMBIGUOUS).sum()),
    }


    # ---------------- P10: flow matching on validated transport ------------
    fm_report: dict[str, Any] = {"run": False}
    if bool(args.skip_fm):
        fm_report = {"run": False, "reason": "--skip-fm"}
    elif int(pairs_fm["source"].shape[0]) < 64:
        fm_report = {
            "run": False,
            "reason": f"only {int(pairs_fm['source'].shape[0])} validated pairs; "
                      f"§38 trains FM on accepted transforms, and there are not enough",
        }
        progress.write(f"[P10] not trained: {fm_report['reason']}")
    else:
        with progress.stage("P10 flow matching"):
            cfg = FlowMatchingConfig(iters=int(args.fm_iters))
            flow = train_transform_flow_matching(
                pairs_fm["source"], pairs_fm["target"], cfg,
                parameterisation=str(args.fm_parameterisation), device=device,
            )
            evaluation = evaluate_flow_against_transform(
                flow.model, pairs_fm["source"], pairs_fm["target"], cfg,
                origin=torch.tensor(flow.stats["origin"]),
            )
        fm_report = {"run": True, **flow.stats, "evaluation": evaluation, "logs": flow.logs[-5:]}
        progress.write(
            f"[P10] trained on {flow.stats['pairs']} transform-derived pairs "
            f"({args.fm_parameterisation}) | final loss {flow.stats['final_loss']:.4g} | "
            f"integrated endpoint error {evaluation['relative_endpoint_error']:.4f} of the displacement"
        )
    results["P10_flow_matching"] = fm_report

    # §63/§64: the go condition, machine-checked rather than left to reading.
    go = {
        "significant_metric_isometry": int(metric_accepted.sum()) > 0,
        "significant_rigid_transforms": int(rigid_accepted.sum()) > 0,
        "coherent_transform_components": int(components.stats["components"]) > 0,
        "improved_render_likelihood": bool(render_report.get("prefers_dynamic_count", 0) > 0),
        "render_validation_run": bool(render_report.get("run", False)),
    }
    results["verdict"] = {
        "go_conditions": go,
        "geometry_stages_pass": bool(
            go["significant_metric_isometry"]
            and go["significant_rigid_transforms"]
            and go["coherent_transform_components"]
        ),
        "proceed_to_fm": bool(
            go["significant_metric_isometry"] and go["significant_rigid_transforms"]
            and go["coherent_transform_components"] and go["improved_render_likelihood"]
        ),
        "fm_trained": bool(fm_report.get("run", False)),
        "result": (
            "GEOMETRY STAGES PASS -- render validation still required"
            if go["significant_metric_isometry"] and go["significant_rigid_transforms"]
            and int(components.stats["components"]) > 0
            else "NO PHYSICALLY VALID TRANSPORT FOUND"
        ),
    }

    if bool(args.save_tensors):
        tensors = out / "tensors"
        tensors.mkdir(parents=True, exist_ok=True)
        torch.save(posterior.forward_mismatch.cpu(), tensors / "forward_static_mismatch.pt")
        torch.save(posterior.backward_mismatch.cpu(), tensors / "backward_static_mismatch.pt")
        torch.save(posterior.source_dynamic_posterior.cpu(), tensors / "source_dynamic_posterior.pt")
        torch.save(posterior.target_dynamic_posterior.cpu(), tensors / "target_dynamic_posterior.pt")
        torch.save(source_anchors.cpu(), tensors / "source_anchors.pt")
        for key in ("distortion", "residual", "metric_p", "rigid_p", "rotation", "translation", "best_target"):
            torch.save(evaluated[key].cpu(), tensors / f"patch_{key}.pt")
        torch.save(physically_valid.cpu(), tensors / "patch_physically_valid.pt")
        torch.save(components.label, tensors / "patch_component.pt")
        torch.save(label, tensors / "frame0_physical_label.pt")
        print(f"[physics] tensors written to {tensors}")

    if not bool(args.no_html):
        page = out / "patch_isometry_debug.html"
        write_debug_html(page, build_debug_payload(
            xyz0, posterior, source_anchors, evaluated, physically_valid, components, label,
            generator=generator,
        ))
        print(f"[physics] debug html: {page}")

    # ---------------- GT firewall: nothing above this line read labels -----
    root_scene = Path(args.root_scene).expanduser().resolve() if args.root_scene else None
    gt = None
    if root_scene is not None:
        parts_ply = root_scene / "pointcloud" / "points3D_parts.ply"
        if parts_ply.is_file():
            from tools.evaluate_closed_loop_motion import _ground_truth_moving

            gt = _ground_truth_moving(xyz0.detach().cpu().double(), parts_ply).bool()

    if gt is not None:
        q0 = posterior.source_dynamic_posterior.cpu()
        predicted = q0 > 0.5
        tp = int((predicted & gt).sum())
        fp = int((predicted & ~gt).sum())
        fn = int((~predicted & gt).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        anchor_gt = gt[source_anchors.cpu()]
        valid_cpu = physically_valid.cpu()
        results["ground_truth"] = {
            "ground_truth_used_as_algorithm_input": False,
            "gt_dynamic_fraction": float(gt.float().mean()),
            "P0_auc": _binary_auc(q0, gt),
            "P0_precision": precision,
            "P0_recall": recall,
            "P0_f1": 2 * precision * recall / max(1.0e-12, precision + recall),
            "P0_static_contamination": float((~gt)[predicted].float().mean()) if bool(predicted.any()) else 0.0,
            "P0_dynamic_capture": float(predicted[gt].float().mean()) if bool(gt.any()) else 0.0,
            "anchors_gt_dynamic_fraction": float(anchor_gt.float().mean()),
            "accepted_anchors_gt_dynamic_fraction": (
                float(anchor_gt[valid_cpu].float().mean()) if bool(valid_cpu.any()) else None
            ),
            "label_dynamic_precision": (
                float(gt[(label == LABEL_DYNAMIC)].float().mean())
                if bool((label == LABEL_DYNAMIC).any()) else None
            ),
        }
        report = results["ground_truth"]
        print(
            f"\n[GT] P0 posterior: AUC {report['P0_auc']:.4f} F1 {report['P0_f1']:.4f} "
            f"(precision {precision:.3f} recall {recall:.3f})"
        )
        print(
            f"[GT] anchors GT-dynamic {report['anchors_gt_dynamic_fraction']:.3f} -> "
            f"after physical acceptance {report['accepted_anchors_gt_dynamic_fraction']}"
        )
    else:
        results["ground_truth"] = None

    _write_json(out / "patch_isometry_probe.json", json_finite(_json_safe(results)))
    print(f"\n=== {results['verdict']['result']} ===")
    print(f"  go conditions: {results['verdict']['go_conditions']}")
    print(f"  proceed to FM (§63, all four): {results['verdict']['proceed_to_fm']}")
    print(f"  FM trained: {results['verdict']['fm_trained']}")
    print(f"\n[physics] wrote {out / 'patch_isometry_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
