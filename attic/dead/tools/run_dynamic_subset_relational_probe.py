"""Stage A.6 probe: does restricting the candidate space BEFORE matching help?

Stage A.5 matched the whole scene and suppressed static pairs afterwards.
That leaves every static Gaussian competing in the retrieval, so a moving
Gaussian is scored against ~11k mostly-static targets and the winner is
decided by whichever of them happens to sit closest in descriptor space.
The result was that relational confidence carried almost no information.

This probe tests whether that was candidate-space contamination rather than
a failure of relative geometry, by running the corrected order:

    forward + reverse spatial mismatch -> q0, qT -> restricted candidates
    -> relational coupling -> confidence / rigidity / reverse consistency

The ablation isolates each ingredient:

    A0  full-scene geometry only            (the Stage A.5 baseline)
    A1  full-scene geometry + SH-DC
    A2  source restriction only
    A3  target restriction only
    A4  bidirectional restriction q0*qT
    A5  A4 + SH-DC
    A6  A5 + relational compatibility
    A7  A6 + reverse candidate union

Per §20 this is NOT judged on binary classification AUC.  The question is
"where did the likely-dynamic geometry go", so the criterion (§28) is that
dynamic restriction improves at least three of: margin, entropy, rigidity
error, reverse-neighbourhood consistency, target-occupancy collapse,
unmatched fraction.

Ground truth is a firewall: loaded only after every coupling exists, used
only to report, never to build a graph, weight, gate, threshold or coupling.
No FM is trained here.

Run:
    PYTHONPATH=. python tools/run_dynamic_subset_relational_probe.py \
        --frame0-model-dir <frame0_static> --frameT-model-dir <frameT_static> \
        --iteration 14000 --output-dir <out> --root-scene <scene>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.bidirectional_spatial_mismatch import compute_bidirectional_spatial_mismatch  # noqa: E402
from utils.dynamic_prior import _binary_auc  # noqa: E402
from utils.dynamic_subset_relational_coupling import (  # noqa: E402
    build_dynamic_descriptor_candidates,
    build_dynamic_subset_coupling,
    compute_dynamic_candidate_confidence,
    compute_relational_rigidity_error,
    compute_reverse_neighborhood_consistency,
    peak_target,
)
from utils.dynamic_subset_selection import (  # noqa: E402
    build_hard_dynamic_subset,
    build_soft_dynamic_weights,
    summarise_dynamic_subset,
)
from utils.relational_graph import build_relational_graph  # noqa: E402
from tools.run_relational_graph_probe import _find_model_ply, _json_safe, _write_json  # noqa: E402

SH_C0 = 0.28209479177387814


def load_cloud(model_dir: str, iteration: int, sh_degree: int = 3):
    """Positions, scales and view-independent colour of one endpoint."""
    from scene.gaussian_model import GaussianModel

    ply = _find_model_ply(model_dir, iteration)
    if ply is None:
        raise SystemExit(f"No point_cloud.ply found under {model_dir}")
    model = GaussianModel(int(sh_degree), optimizer_type="default")
    model.load_ply(str(ply))
    # Only the DC band. Higher SH bands are view-dependent, so they change
    # with the camera rather than with the Gaussian, and two independently
    # trained endpoint models will not agree on them.
    dc = model.get_features.detach().float()[:, 0, :]
    return (
        model.get_canonical_xyz.detach().float(),
        model.get_scaling.detach().float(),
        (SH_C0 * dc + 0.5).clamp(0.0, 1.0),
        ply,
    )


def json_finite(value):
    """Recursively turn NaN/inf into null.

    A bin that contained nothing measurable must serialise as ``null``, not
    as a number: writing NaN would either break a strict reader or, worse,
    be silently read back as a value and averaged into a summary.
    """
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
        "median": float(v.median()),
        "mean": float(v.mean()),
        "p10": float(torch.quantile(v, 0.10)),
        "p90": float(torch.quantile(v, 0.90)),
    }


# --------------------------------------------------------------------------
# The A0-A7 ladder
# --------------------------------------------------------------------------
def variant_specs(args) -> dict[str, dict[str, Any]]:
    """Each rung adds exactly one ingredient to the rung below it."""
    appearance = float(args.lambda_appearance_descriptor)
    return {
        "A0_full_geometry": dict(source=False, target=False, appearance=0.0, sweeps=0, union=False),
        "A1_full_geometry_sh": dict(source=False, target=False, appearance=appearance, sweeps=0, union=False),
        "A2_source_restricted": dict(source=True, target=False, appearance=0.0, sweeps=0, union=False),
        "A3_target_restricted": dict(source=False, target=True, appearance=0.0, sweeps=0, union=False),
        "A4_bidirectional": dict(source=True, target=True, appearance=0.0, sweeps=0, union=False),
        "A5_bidirectional_sh": dict(source=True, target=True, appearance=appearance, sweeps=0, union=False),
        "A6_relational": dict(source=True, target=True, appearance=appearance,
                              sweeps=int(args.relational_sweeps), union=False),
        "A7_reverse_union": dict(source=True, target=True, appearance=appearance,
                                 sweeps=int(args.relational_sweeps), union=True),
    }


def build_variant(source_graph, target_graph, weights, spec: dict[str, Any], args):
    """One rung of the ladder.

    A variant that restricts neither side is the Stage A.5 baseline and must
    say so explicitly -- the dynamic path refuses to run without weights
    precisely so that a full-scene run cannot happen by omission.
    """
    q0 = weights.source_weight if spec["source"] else None
    qT = weights.target_weight if spec["target"] else None
    graph_src = source_graph if spec["appearance"] <= 0.0 else source_graph
    return build_dynamic_subset_coupling(
        graph_src, target_graph,
        source_weight=q0, target_weight=qT,
        max_candidates=int(args.descriptor_candidates),
        lambda_distance=float(args.lambda_distance_descriptor),
        lambda_covariance=float(args.lambda_covariance_descriptor),
        lambda_appearance=float(spec["appearance"]),
        lambda_relational=float(args.lambda_relational),
        relational_sweeps=int(spec["sweeps"]),
        relational_bandwidth=float(args.relational_bandwidth),
        target_mode=str(args.target_mode),
        lambda_q=float(args.lambda_q),
        source_threshold=float(args.source_threshold),
        bidirectional_union=bool(spec["union"]),
        unmatched_quantile=float(args.unmatched_quantile),
        column_capacity=float(args.column_capacity),
        chunk_size=int(args.chunk_size),
        edge_chunk=int(args.edge_chunk),
        allow_full_scene=not (spec["source"] or spec["target"]),
    )


def measure_variant(
    source_graph,
    target_graph,
    coupling,
    weights,
    reverse_idx: torch.Tensor,
    *,
    dynamic_population: torch.Tensor,
) -> dict[str, Any]:
    """§23 / §47 metrics. Every one of these is computed without ground truth.

    ``dynamic_population`` is fixed across the whole ladder on purpose: it is
    derived from the gates once, not from each variant's own restriction, so
    that "rigidity improved" cannot mean "we measured an easier subset".
    """
    confidence = compute_dynamic_candidate_confidence(coupling.total_cost, coupling.mass, coupling.valid)
    tgt, matched = peak_target(coupling)
    scale = source_graph.effective_scale.clamp_min(1.0e-8)
    displacement = torch.linalg.norm(target_graph.xyz[tgt] - source_graph.xyz, dim=-1) / scale
    displacement = torch.where(matched, displacement, torch.full_like(displacement, float("nan")))

    # Two fixed populations, identical for every rung of the ladder.
    rigidity_all = compute_relational_rigidity_error(source_graph, target_graph.xyz, coupling)
    rigidity_dynamic = compute_relational_rigidity_error(
        source_graph, target_graph.xyz, coupling, evaluation_mask=dynamic_population,
    )
    rigidity = rigidity_dynamic
    reverse = compute_reverse_neighborhood_consistency(source_graph, coupling, reverse_idx)
    reverse_dynamic = reverse[dynamic_population.to(reverse.device)]

    occupancy = coupling.col_mass
    covered = occupancy > 0.05
    return {
        "candidate_concentration": {
            "median_top1_mass": confidence.stats["median_peak_mass"],
            "median_peak_ratio": confidence.stats["median_peak_ratio"],
            "median_cost_margin": confidence.stats["median_cost_margin"],
            "median_entropy": confidence.stats["median_entropy"],
            "unmatched_fraction": float(coupling.unmatched_src.mean()),
            "active_row_fraction": float((coupling.valid.any(dim=1)).float().mean()),
        },
        "dynamic_transport": {
            "median_peak_displacement": stats(displacement).get("median", float("nan")),
            "p90_peak_displacement": stats(displacement).get("p90", float("nan")),
            "matched_fraction": float(matched.float().mean()),
            "source_coverage": float((coupling.row_mass > 0.05).float().mean()),
            "target_coverage": float(covered.float().mean()),
            "max_target_occupancy": float(occupancy.max()),
            "occupancy_p99": float(torch.quantile(occupancy.float(), 0.99)),
        },
        "relative_rigidity": {
            **{f"error_{k}": v for k, v in stats(rigidity["per_point_error"]).items()},
            "median_inlier_fraction": float(rigidity["inlier_fraction"].median()),
            "measured_fraction": float(rigidity["measured_fraction"]),
        },
        "relative_rigidity_all_points": {
            **{f"error_{k}": v for k, v in stats(rigidity_all["per_point_error"]).items()},
            "measured_fraction": float(rigidity_all["measured_fraction"]),
        },
        "reverse_neighborhood": {
            **{f"error_{k}": v for k, v in stats(reverse_dynamic).items()},
            "measured_fraction": float(torch.isfinite(reverse_dynamic).float().mean()),
        },
        "reverse_neighborhood_all_points": {
            **{f"error_{k}": v for k, v in stats(reverse).items()},
            "measured_fraction": float(torch.isfinite(reverse).float().mean()),
        },
        "_fields": {
            "cost_margin": confidence.cost_margin,
            "peak_ratio": confidence.peak_ratio,
            "entropy": confidence.entropy,
            "peak_mass": confidence.peak_mass,
            "peak_target": tgt,
            "peak_displacement": displacement,
            "rigidity_error": rigidity["per_point_error"],
            "reverse_error": reverse,
            "matched": matched,
        },
    }


# --------------------------------------------------------------------------
# §28 go / no-go
# --------------------------------------------------------------------------
CRITERIA = (
    # (label, path, "higher is better")
    ("top1/top2 margin", ("candidate_concentration", "median_cost_margin"), True),
    ("peak ratio", ("candidate_concentration", "median_peak_ratio"), True),
    ("entropy", ("candidate_concentration", "median_entropy"), False),
    ("rigidity error", ("relative_rigidity", "error_median"), False),
    ("reverse-neighbourhood error", ("reverse_neighborhood", "error_median"), False),
    ("max target occupancy", ("dynamic_transport", "max_target_occupancy"), False),
)


def stop_condition(report: dict[str, Any], *, args) -> dict[str, Any]:
    """§49: is the coupling coherent in ABSOLUTE terms, not just relative ones?

    §28 asks whether restriction improved things and §49 asks whether the
    result is usable, and the two can disagree -- a transport can improve on
    every axis and still be incoherent.  Only §49 can stop the FM stack,
    because you cannot supervise flow matching with a transport whose
    neighbour distances change by many local scales however much better that
    is than before.

    The thresholds are in scene-free units (normalised entropy and margin in
    [0, 1], errors in local-scale units) so they carry across datasets.
    """
    conc = report["candidate_concentration"]
    rigid = report["relative_rigidity"]
    reverse = report["reverse_neighborhood"]
    checks = {
        "entropy_remains_high": {
            "value": conc["median_entropy"], "limit": float(args.stop_entropy), "triggered": None,
        },
        "margin_remains_near_zero": {
            "value": conc["median_cost_margin"], "limit": float(args.stop_margin), "triggered": None,
        },
        "rigidity_error_stays_large": {
            "value": rigid.get("error_median"), "limit": float(args.stop_rigidity), "triggered": None,
        },
        "rigid_inlier_fraction_too_low": {
            "value": rigid.get("median_inlier_fraction"), "limit": float(args.stop_inlier), "triggered": None,
        },
        "reverse_consistency_stays_poor": {
            "value": reverse.get("error_median"), "limit": float(args.stop_reverse), "triggered": None,
        },
    }
    checks["entropy_remains_high"]["triggered"] = checks["entropy_remains_high"]["value"] > float(args.stop_entropy)
    checks["margin_remains_near_zero"]["triggered"] = checks["margin_remains_near_zero"]["value"] < float(args.stop_margin)
    for key, limit, worse_when_above in (
        ("rigidity_error_stays_large", float(args.stop_rigidity), True),
        ("reverse_consistency_stays_poor", float(args.stop_reverse), True),
        ("rigid_inlier_fraction_too_low", float(args.stop_inlier), False),
    ):
        value = checks[key]["value"]
        checks[key]["triggered"] = None if value is None or value != value else (
            value > limit if worse_when_above else value < limit
        )
    triggered = [k for k, v in checks.items() if v["triggered"]]
    return {"checks": checks, "triggered": triggered, "stop": len(triggered) >= 3}


def compare_to_baseline(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """§28: count how many concentration/coherence measures actually improved."""
    rows = {}
    improved = 0
    for label, (section, key), higher_better in CRITERIA:
        before = baseline[section].get(key, float("nan"))
        after = candidate[section].get(key, float("nan"))
        if before != before or after != after:      # NaN
            rows[label] = {"before": before, "after": after, "improved": None}
            continue
        better = after > before if higher_better else after < before
        rows[label] = {"before": before, "after": after, "improved": bool(better),
                       "delta": float(after - before)}
        improved += int(better)
    return {"criteria": rows, "improved_count": improved, "required": 3, "passes": improved >= 3}


# --------------------------------------------------------------------------
# §45 debug HTML
# --------------------------------------------------------------------------
def write_debug_html(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(_json_safe(payload))
    path.write_text(
        """<!doctype html><meta charset="utf-8"><title>Stage A.6 dynamic subset transport</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>body{font:13px system-ui;margin:0;background:#111;color:#ddd}
#tabs{padding:8px;background:#1b1b1b;position:sticky;top:0}
button{font:12px system-ui;margin:2px;padding:5px 9px;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px;cursor:pointer}
button.on{background:#3a5a8a;border-color:#6a9ada}#plot{height:88vh}</style>
<div id="tabs"></div><div id="plot"></div>
<script>
const D = """ + body + """;
const views = Object.keys(D.views);
const tabs = document.getElementById('tabs');
function draw(name){
  [...tabs.children].forEach(b=>b.className = b.textContent===name?'on':'');
  const v = D.views[name];
  const traces = [];
  if (v.lines) traces.push({type:'scatter3d',mode:'lines',x:v.lines.map(p=>p[0]),
    y:v.lines.map(p=>p[1]),z:v.lines.map(p=>p[2]),line:{color:'#e8a33d',width:1.5},
    name:'top-1 candidate',hoverinfo:'skip'});
  traces.push({type:'scatter3d',mode:'markers',x:D.source.map(p=>p[0]),y:D.source.map(p=>p[1]),
    z:D.source.map(p=>p[2]),marker:{size:1.8,color:v.value,colorscale:v.scale||'Viridis',
    cmin:v.cmin,cmax:v.cmax,showscale:true,colorbar:{title:name}},name:'frame 0',
    text:v.value.map(x=>x.toFixed(3)),hoverinfo:'text'});
  if (v.target) traces.push({type:'scatter3d',mode:'markers',x:D.target.map(p=>p[0]),
    y:D.target.map(p=>p[1]),z:D.target.map(p=>p[2]),marker:{size:1.6,color:v.target,
    colorscale:v.scale||'Viridis',showscale:false,opacity:0.55},name:'frame T'});
  Plotly.react('plot',traces,{paper_bgcolor:'#111',plot_bgcolor:'#111',
    font:{color:'#ddd'},margin:{l:0,r:0,t:0,b:0},showlegend:true,
    scene:{aspectmode:'data',xaxis:{color:'#666'},yaxis:{color:'#666'},zaxis:{color:'#666'}}});
}
views.forEach(n=>{const b=document.createElement('button');b.textContent=n;b.onclick=()=>draw(n);tabs.appendChild(b);});
draw(views[0]);
</script>""",
        encoding="utf-8",
    )


def build_html_payload(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    mismatch,
    weights,
    fields: dict[str, torch.Tensor],
    *,
    max_points: int = 6000,
    max_lines: int = 1200,
) -> dict[str, Any]:
    n = int(source_xyz.shape[0])
    generator = torch.Generator().manual_seed(0)
    keep = torch.arange(n) if n <= max_points else torch.randperm(n, generator=generator)[:max_points].sort().values
    tkeep = torch.arange(int(target_xyz.shape[0]))
    if int(target_xyz.shape[0]) > max_points:
        tkeep = torch.randperm(int(target_xyz.shape[0]), generator=generator)[:max_points].sort().values

    src = source_xyz.detach().cpu()[keep]
    tgt = target_xyz.detach().cpu()[tkeep]

    def column(value: torch.Tensor) -> list[float]:
        v = torch.as_tensor(value).detach().float().cpu().reshape(-1)[keep]
        return [float(x) if torch.isfinite(torch.tensor(float(x))) else 0.0 for x in v]

    # Top-1 candidate lines, drawn only where the row actually matched: a line
    # from an unmatched row would show a correspondence that was never made.
    matched = fields["matched"].detach().cpu()
    line_ids = keep[matched[keep]]
    if int(line_ids.numel()) > max_lines:
        line_ids = line_ids[torch.randperm(int(line_ids.numel()), generator=generator)[:max_lines]]
    peak = fields["peak_target"].detach().cpu()
    lines: list[list[float]] = []
    for i in line_ids.tolist():
        lines.append([float(x) for x in source_xyz[i].detach().cpu()])
        lines.append([float(x) for x in target_xyz[int(peak[i])].detach().cpu()])
        lines.append([None, None, None])

    log_forward = torch.log10(mismatch.forward.detach().cpu() + 1.0e-6)
    log_backward = torch.log10(mismatch.backward.detach().cpu() + 1.0e-6)
    return {
        "source": [[float(v) for v in row] for row in src],
        "target": [[float(v) for v in row] for row in tgt],
        "views": {
            "q0 (source dynamic weight)": {"value": column(weights.source_weight), "cmin": 0, "cmax": 1,
                                           "target": [float(v) for v in weights.target_weight.detach().cpu()[tkeep]]},
            "log10 forward mismatch": {"value": column(log_forward),
                                       "target": [float(v) for v in log_backward[tkeep]]},
            "hard dynamic subset": {"value": column(build_hard_dynamic_subset(weights.source_weight).float()),
                                    "cmin": 0, "cmax": 1},
            "top-1 candidate lines": {"value": column(fields["peak_displacement"]), "lines": lines},
            "peak candidate mass": {"value": column(fields["peak_mass"])},
            "cost margin": {"value": column(fields["cost_margin"]), "cmin": 0, "cmax": 1},
            "candidate entropy": {"value": column(fields["entropy"]), "cmin": 0, "cmax": 1},
            "relational rigidity error": {"value": column(fields["rigidity_error"]), "scale": "Inferno"},
            "reverse-neighbourhood error": {"value": column(fields["reverse_error"]), "scale": "Inferno"},
        },
    }


# --------------------------------------------------------------------------
# Everything below reads ground truth. Nothing above it may.
# --------------------------------------------------------------------------
def load_ground_truth(source_xyz: torch.Tensor, root_scene: Path | None) -> torch.Tensor | None:
    if root_scene is None:
        return None
    parts_ply = root_scene / "pointcloud" / "points3D_parts.ply"
    if not parts_ply.is_file():
        print(f"[A6] no ground truth at {parts_ply}; GT sections will be omitted")
        return None
    from tools.evaluate_closed_loop_motion import _ground_truth_moving

    return _ground_truth_moving(source_xyz.detach().cpu().double(), parts_ply).bool()


def gt_report(
    weights,
    fields: dict[str, torch.Tensor],
    gt: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """§25: what the dynamic subset actually caught, measured after the fact."""
    q0 = weights.source_weight.detach().cpu()
    active = q0 > float(threshold)
    rigidity = fields["rigidity_error"].detach().cpu()
    finite = torch.isfinite(rigidity)
    return {
        "ground_truth_used_as_algorithm_input": False,
        "source_active_fraction": float(active.float().mean()),
        "source_active_gt_dynamic_fraction": float(gt[active].float().mean()) if bool(active.any()) else 0.0,
        "gt_dynamic_captured_by_subset": float(active[gt].float().mean()) if bool(gt.any()) else 0.0,
        "gt_static_contamination": float((~gt)[active].float().mean()) if bool(active.any()) else 0.0,
        "q0_auc": _binary_auc(q0, gt),
        "rigidity_median_on_gt_dynamic": float(rigidity[finite & gt].median()) if bool((finite & gt).any()) else float("nan"),
        "rigidity_median_on_gt_static": float(rigidity[finite & ~gt].median()) if bool((finite & ~gt).any()) else float("nan"),
    }


def binned_by_q0(weights, fields: dict[str, torch.Tensor], gt: torch.Tensor | None, *, bins: int = 5) -> list[dict[str, Any]]:
    """§26: every diagnostic sliced by the source dynamic weight."""
    q0 = weights.source_weight.detach().cpu().float()
    chunks = torch.chunk(torch.argsort(q0), int(bins))
    rows = []
    for index, chunk in enumerate(chunks):
        row: dict[str, Any] = {
            "bin": f"{index * 100 // bins}-{(index + 1) * 100 // bins}%",
            "count": int(chunk.numel()),
            "q0_range": [float(q0[chunk].min()), float(q0[chunk].max())],
        }
        for name in ("cost_margin", "entropy", "peak_mass", "peak_displacement", "rigidity_error", "reverse_error"):
            v = fields[name].detach().cpu().float()[chunk]
            v = v[torch.isfinite(v)]
            row[f"{name}_median"] = float(v.median()) if int(v.numel()) else float("nan")
        if gt is not None:
            row["gt_dynamic_fraction"] = float(gt.cpu()[chunk].float().mean())
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame0-model-dir", required=True)
    parser.add_argument("--frameT-model-dir", required=True)
    parser.add_argument("--iteration", type=int, default=14000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-scene", default="", help="only used for post-hoc GT reporting")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--edge-chunk", type=int, default=1024)

    parser.add_argument("--knn-rig", type=int, default=16)
    parser.add_argument("--knn-ctx", type=int, default=64)
    parser.add_argument("--descriptor-candidates", type=int, default=32)
    parser.add_argument("--lambda-distance-descriptor", type=float, default=1.0)
    parser.add_argument("--lambda-covariance-descriptor", type=float, default=0.2)
    parser.add_argument("--lambda-appearance-descriptor", type=float, default=1.0)
    parser.add_argument("--lambda-relational", type=float, default=1.0)
    parser.add_argument("--relational-sweeps", type=int, default=2)
    parser.add_argument("--relational-bandwidth", type=float, default=1.0)
    parser.add_argument("--unmatched-quantile", type=float, default=0.90)
    parser.add_argument("--column-capacity", type=float, default=4.0)

    parser.add_argument("--gate-mode", default="otsu", choices=("otsu", "median", "quantile"))
    parser.add_argument("--gate-center-quantile", type=float, default=0.50)
    parser.add_argument("--gate-temperature-scale", type=float, default=1.0)
    parser.add_argument("--target-mode", default="soft", choices=("soft", "hard", "none"))
    parser.add_argument("--lambda-q", type=float, default=1.0)
    parser.add_argument("--source-threshold", type=float, default=0.05)
    # §49 absolute coherence limits, in scene-free units.
    parser.add_argument("--stop-entropy", type=float, default=0.90,
                        help="normalised row entropy above this is 'remains high'")
    parser.add_argument("--stop-margin", type=float, default=0.10,
                        help="cost margin below this is 'near zero'")
    parser.add_argument("--stop-rigidity", type=float, default=1.0,
                        help="median neighbour-distance change above this many local scales is 'large'")
    parser.add_argument("--stop-inlier", type=float, default=0.50,
                        help="median rigid-inlier fraction below this is incoherent")
    parser.add_argument("--stop-reverse", type=float, default=3.0,
                        help="reverse-neighbourhood error above this many local scales is 'poor'")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--save-tensors", action="store_true")
    parser.add_argument("--no-html", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    xyz0, scale0, rgb0, ply0 = load_cloud(args.frame0_model_dir, int(args.iteration), int(args.sh_degree))
    xyzT, scaleT, rgbT, plyT = load_cloud(args.frameT_model_dir, int(args.iteration), int(args.sh_degree))
    xyz0, scale0, rgb0 = xyz0.to(device), scale0.to(device), rgb0.to(device)
    xyzT, scaleT, rgbT = xyzT.to(device), scaleT.to(device), rgbT.to(device)
    print(f"[A6] frame0={int(xyz0.shape[0])} ({ply0})")
    print(f"[A6] frameT={int(xyzT.shape[0])} ({plyT})")

    graph_kwargs = dict(knn_rig=int(args.knn_rig), knn_ctx=int(args.knn_ctx), chunk_size=max(512, int(args.chunk_size)))
    source_geom = build_relational_graph(xyz0, scaling=scale0, **graph_kwargs)
    target_geom = build_relational_graph(xyzT, scaling=scaleT, **graph_kwargs)
    # Appearance variants attach colour to the SAME geometry; rebuilding the
    # graph would make them differ by more than the cue under test.
    source = replace(source_geom, appearance=rgb0)
    target = replace(target_geom, appearance=rgbT)

    # ---- Step A: bidirectional raw spatial mismatch, then the soft gates ----
    mismatch = compute_bidirectional_spatial_mismatch(
        xyz0, xyzT, source.effective_scale, target.effective_scale, chunk_size=int(args.chunk_size),
    )
    weights = build_soft_dynamic_weights(
        mismatch, mode=str(args.gate_mode),
        center_quantile=float(args.gate_center_quantile),
        temperature_scale=float(args.gate_temperature_scale),
    )
    print(
        f"[A6] q0 active {weights.source_gate['active_fraction']:.3f} "
        f"(mu {weights.source_gate['mu']:.2f} tau {weights.source_gate['tau']:.2f}) | "
        f"qT active {weights.target_gate['active_fraction']:.3f} "
        f"(mu {weights.target_gate['mu']:.2f} tau {weights.target_gate['tau']:.2f})"
    )

    # Reverse candidate list, used for the §24 neighbourhood test. Built once
    # and shared, so every variant is judged against the same reverse view.
    reverse_idx, _reverse_cost, _active = build_dynamic_descriptor_candidates(
        target, source, target_weight=weights.source_weight, source_weight=weights.target_weight,
        max_candidates=max(4, int(args.descriptor_candidates) // 4),
        lambda_distance=float(args.lambda_distance_descriptor),
        lambda_covariance=float(args.lambda_covariance_descriptor),
        lambda_appearance=float(args.lambda_appearance_descriptor),
        target_mode=str(args.target_mode), lambda_q=float(args.lambda_q),
        source_threshold=float(args.source_threshold), chunk_size=int(args.chunk_size),
    )

    results: dict[str, Any] = {
        "config": {k: _json_safe(v) for k, v in vars(args).items()},
        "frame0_count": int(xyz0.shape[0]),
        "frameT_count": int(xyzT.shape[0]),
        "gates": {"source": weights.source_gate, "target": weights.target_gate},
        "subsets": {
            "source": summarise_dynamic_subset(weights.source_weight, mismatch.forward),
            "target": summarise_dynamic_subset(weights.target_weight, mismatch.backward),
        },
        "variants": {},
    }

    specs = variant_specs(args)
    # One population, fixed before the ladder runs, used to evaluate every
    # variant's rigidity and reverse consistency.
    dynamic_population = build_hard_dynamic_subset(weights.source_weight)
    print(f"[A6] fixed evaluation population: {int(dynamic_population.sum())} points "
          f"({float(dynamic_population.float().mean()):.3f})")

    measured: dict[str, dict[str, Any]] = {}
    fields_by_variant: dict[str, dict[str, torch.Tensor]] = {}
    for name, spec in specs.items():
        coupling = build_variant(source, target, weights, spec, args)
        report = measure_variant(
            source, target, coupling, weights, reverse_idx,
            dynamic_population=dynamic_population,
        )
        fields_by_variant[name] = report.pop("_fields")
        report["spec"] = {k: _json_safe(v) for k, v in spec.items()}
        measured[name] = report
        results["variants"][name] = report
        conc, trans = report["candidate_concentration"], report["dynamic_transport"]
        print(
            f"[A6] {name:24s} margin {conc['median_cost_margin']:.4f} "
            f"entropy {conc['median_entropy']:.4f} "
            f"rigid {report['relative_rigidity'].get('error_median', float('nan')):7.3f} "
            f"rev {report['reverse_neighborhood'].get('error_median', float('nan')):7.3f} "
            f"occ {trans['max_target_occupancy']:7.2f} "
            f"unmatched {conc['unmatched_fraction']:.3f}"
        )

    baseline = measured["A0_full_geometry"]
    results["gate_vs_A0"] = {
        name: compare_to_baseline(baseline, report)
        for name, report in measured.items() if name != "A0_full_geometry"
    }
    # The question Stage A.6 asks is whether RESTRICTION helps, so each
    # restricted variant is also compared against the control that differs
    # from it only by the restriction.  Against A0, a variant that merely
    # added the appearance cue would be credited to the restriction.
    CONTROLS = {
        "A2_source_restricted": "A0_full_geometry",
        "A3_target_restricted": "A0_full_geometry",
        "A4_bidirectional": "A0_full_geometry",
        "A5_bidirectional_sh": "A1_full_geometry_sh",
        "A6_relational": "A1_full_geometry_sh",
        "A7_reverse_union": "A1_full_geometry_sh",
    }
    results["restriction_vs_matched_control"] = {
        name: dict(compare_to_baseline(measured[control], measured[name]), control=control)
        for name, control in CONTROLS.items()
    }
    best_name = max(
        results["restriction_vs_matched_control"],
        key=lambda k: results["restriction_vs_matched_control"][k]["improved_count"],
    )
    results["verdict"] = {
        "isolating_baseline": CONTROLS[best_name],
        "best_restricted_variant": best_name,
        "improved_count": results["restriction_vs_matched_control"][best_name]["improved_count"],
        "required": 3,
        # The literal §28 reading, kept for reference, and the isolating one
        # that actually answers the stage's question.
        "stage_a6_passes_vs_A0": bool(
            max(c["improved_count"] for c in results["gate_vs_A0"].values()) >= 3
        ),
        "stage_a6_passes_isolating_restriction": bool(
            results["restriction_vs_matched_control"][best_name]["passes"]
        ),
        "appearance_only_improved_count": results["gate_vs_A0"]["A1_full_geometry_sh"]["improved_count"],
    }
    results["stop_conditions"] = {name: stop_condition(report, args=args) for name, report in measured.items()}
    # §49 governs.  A relative improvement on three of six axes does not make
    # a transport usable if it is still incoherent in absolute terms, and it
    # is the FM stack downstream that would pay for the difference.
    best_stop = results["stop_conditions"][best_name]
    results["verdict"]["stop_condition_triggered"] = best_stop["triggered"]
    results["verdict"]["proceed_to_dynamic_fm"] = bool(
        results["verdict"]["stage_a6_passes_isolating_restriction"] and not best_stop["stop"]
    )

    if bool(args.save_tensors):
        tensors = out / "tensors"
        tensors.mkdir(parents=True, exist_ok=True)
        torch.save(mismatch.forward.cpu(), tensors / "forward_static_mismatch.pt")
        torch.save(mismatch.backward.cpu(), tensors / "backward_static_mismatch.pt")
        torch.save(weights.source_weight.cpu(), tensors / "source_dynamic_weight.pt")
        torch.save(weights.target_weight.cpu(), tensors / "target_dynamic_weight.pt")
        for name, fields in fields_by_variant.items():
            for key, value in fields.items():
                torch.save(value.detach().cpu(), tensors / f"{name}__{key}.pt")
        print(f"[A6] tensors written to {tensors}")

    if not bool(args.no_html):
        html = out / "dynamic_subset_transport_debug.html"
        write_debug_html(html, build_html_payload(xyz0, xyzT, mismatch, weights, fields_by_variant[best_name]))
        print(f"[A6] debug html: {html}")

    # ---------------- GT firewall: nothing above this line read labels -----
    root_scene = Path(args.root_scene).expanduser().resolve() if args.root_scene else None
    gt = load_ground_truth(xyz0, root_scene)
    if gt is not None:
        results["ground_truth"] = {
            name: gt_report(weights, fields, gt) for name, fields in fields_by_variant.items()
        }
        results["binned_by_q0"] = {
            name: binned_by_q0(weights, fields, gt, bins=int(args.bins))
            for name, fields in fields_by_variant.items()
        }
    else:
        results["ground_truth"] = None
        results["binned_by_q0"] = {
            name: binned_by_q0(weights, fields, None, bins=int(args.bins))
            for name, fields in fields_by_variant.items()
        }

    _write_json(out / "dynamic_subset_relational_probe.json", json_finite(_json_safe(results)))

    print("\n=== Stage A.6 acceptance table ===")
    header = (
        f"  {'variant':24s} {'srcAct':>7s} {'tgtAct':>7s} {'top1':>7s} {'margin':>7s} {'entropy':>8s} "
        f"{'disp':>8s} {'rigid':>8s} {'rig p90':>8s} {'revNbr':>8s} {'tgtCov':>7s} {'maxOcc':>7s} {'unmat':>6s}"
    )
    print(header)
    src_active = weights.source_gate["active_fraction"]
    tgt_active = weights.target_gate["active_fraction"]
    for name, report in measured.items():
        conc, trans = report["candidate_concentration"], report["dynamic_transport"]
        rig, rev = report["relative_rigidity"], report["reverse_neighborhood"]
        restricted = report["spec"]["source"] or report["spec"]["target"]
        print(
            f"  {name:24s} {src_active if report['spec']['source'] else 1.0:7.3f} "
            f"{tgt_active if report['spec']['target'] else 1.0:7.3f} "
            f"{conc['median_top1_mass']:7.4f} {conc['median_cost_margin']:7.4f} {conc['median_entropy']:8.4f} "
            f"{trans['median_peak_displacement']:8.2f} {rig.get('error_median', float('nan')):8.3f} "
            f"{rig.get('error_p90', float('nan')):8.3f} {rev.get('error_median', float('nan')):8.3f} "
            f"{trans['target_coverage']:7.3f} {trans['max_target_occupancy']:7.2f} "
            f"{conc['unmatched_fraction']:6.3f}"
        )
        if not restricted:
            continue

    def _marks(comparison) -> str:
        return "".join(
            "+" if row["improved"] else ("." if row["improved"] is not None else "?")
            for row in comparison["criteria"].values()
        )

    print(f"\n  §28 literal reading -- everything vs A0 (needs 3 of {len(CRITERIA)}):")
    for name, comparison in results["gate_vs_A0"].items():
        print(f"    {name:24s} {comparison['improved_count']}/{len(CRITERIA)}  [{_marks(comparison)}]  "
              f"{'PASS' if comparison['passes'] else 'fail'}")
    print(f"    criteria order: {', '.join(label for label, _, _ in CRITERIA)}")

    print("\n  isolating the RESTRICTION -- each variant vs its appearance-matched control:")
    for name, comparison in results["restriction_vs_matched_control"].items():
        print(f"    {name:24s} vs {comparison['control']:20s} {comparison['improved_count']}/{len(CRITERIA)}  "
              f"[{_marks(comparison)}]  {'PASS' if comparison['passes'] else 'fail'}")

    verdict = results["verdict"]
    print(
        f"\n  appearance alone (A1 vs A0) improves {verdict['appearance_only_improved_count']}/{len(CRITERIA)}"
    )
    print(
        f"  §28 (relative): {'PASS' if verdict['stage_a6_passes_isolating_restriction'] else 'fail'} "
        f"on the isolating comparison (best {verdict['best_restricted_variant']} vs "
        f"{verdict['isolating_baseline']}, {verdict['improved_count']}/{len(CRITERIA)} improved)"
    )

    print(f"\n  §49 absolute coherence -- is the result usable, not just better?")
    print(f"    {'variant':24s} {'entropy':>9s} {'margin':>8s} {'rigidity':>9s} {'inlier':>8s} {'revNbr':>8s}   stop")
    for name, condition in results["stop_conditions"].items():
        checks = condition["checks"]
        row = (
            f"    {name:24s} "
            f"{checks['entropy_remains_high']['value']:9.4f}"
            f"{checks['margin_remains_near_zero']['value']:8.4f}"
            f"{(checks['rigidity_error_stays_large']['value'] or float('nan')):9.2f}"
            f"{(checks['rigid_inlier_fraction_too_low']['value'] or 0.0):8.3f}"
            f"{(checks['reverse_consistency_stays_poor']['value'] or float('nan')):8.2f}"
            f"   {len(condition['triggered'])}/5 {'STOP' if condition['stop'] else ''}"
        )
        print(row)
    print(
        f"    limits: entropy>{args.stop_entropy} margin<{args.stop_margin} "
        f"rigidity>{args.stop_rigidity} inlier<{args.stop_inlier} reverse>{args.stop_reverse}"
    )
    print(
        f"\n  PROCEED TO DYNAMIC FM: {'YES' if verdict['proceed_to_dynamic_fm'] else 'NO'}"
        + ("" if verdict["proceed_to_dynamic_fm"]
           else f"  (§49 triggered: {', '.join(verdict['stop_condition_triggered'])})")
    )

    if gt is not None:
        print("\n  GT debug (post-hoc only)")
        report = results["ground_truth"][best_name]
        print(
            f"    q0 active {report['source_active_fraction']:.3f} | "
            f"of which GT-dynamic {report['source_active_gt_dynamic_fraction']:.3f} | "
            f"GT-dynamic captured {report['gt_dynamic_captured_by_subset']:.3f} | "
            f"static contamination {report['gt_static_contamination']:.3f}"
        )
        print(
            f"    q0 AUC {report['q0_auc']:.4f} | rigidity on GT-dynamic "
            f"{report['rigidity_median_on_gt_dynamic']:.3f} vs GT-static "
            f"{report['rigidity_median_on_gt_static']:.3f}"
        )
        print("\n  binned by q0 (best variant)")
        print(f"    {'bin':>10s} {'GTdyn':>7s} {'margin':>7s} {'entropy':>8s} {'disp':>8s} {'rigid':>8s} {'revNbr':>8s}")
        for row in results["binned_by_q0"][best_name]:
            print(
                f"    {row['bin']:>10s} {row.get('gt_dynamic_fraction', float('nan')):7.3f} "
                f"{row['cost_margin_median']:7.4f} {row['entropy_median']:8.4f} "
                f"{row['peak_displacement_median']:8.2f} {row['rigidity_error_median']:8.3f} "
                f"{row['reverse_error_median']:8.3f}"
            )

    print(f"\n[A6] wrote {out / 'dynamic_subset_relational_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
