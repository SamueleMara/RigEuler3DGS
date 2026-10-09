"""Debug artifacts for progressive rigid discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from utils.progressive_rigid_state import ProgressiveRigidState


def save_progressive_state_artifacts(
    state: ProgressiveRigidState,
    output_dir: str | Path,
    *,
    prefix: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_" if prefix else ""
    state.save(output / f"{name}state.pt")
    torch.save(state.part_transforms_T.detach().cpu(), output / f"{name}transforms.pt")
    torch.save(state.part_twists_xi.detach().cpu(), output / f"{name}twists.pt")
    torch.save(state.hard_labels.detach().cpu(), output / f"{name}labels.pt")
    torch.save(state.responsibilities.detach().cpu(), output / f"{name}responsibilities.pt")
    torch.save(state.status.detach().cpu(), output / f"{name}status.pt")
    torch.save(state.provenance.detach().cpu(), output / f"{name}provenance.pt")
    torch.save(state.fm_rigid_residual.detach().cpu(), output / f"{name}fm_rigid_residual.pt")
    torch.save(state.pcd_residual_per_gaussian.detach().cpu(), output / f"{name}pcd_residual_per_gaussian.pt")
    torch.save(state.render_residual_per_gaussian.detach().cpu(), output / f"{name}render_residual_per_gaussian.pt")
    torch.save(state.depth_residual_per_gaussian.detach().cpu(), output / f"{name}depth_residual_per_gaussian.pt")
    torch.save(state.candidate_component_id.detach().cpu(), output / f"{name}candidate_components.pt")
    summary = {
        "schema_version": 1,
        "pipeline": "progressive_rigid_discovery",
        "outer_iteration": int(state.outer_iteration),
        "label_counts": state.label_counts(),
        "component_scores": state.component_scores,
        "accepted_components": state.accepted_components,
        "rejected_components": state.rejected_components,
        "deferred_components": state.deferred_components,
        "extra": extra or {},
        "ground_truth_used_as_algorithm_input": False,
    }
    (output / f"{name}summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return output


def write_component_score_files(
    root: str | Path,
    component_id: int,
    scores: dict[int, dict[str, float]],
    decision: dict[str, Any],
) -> None:
    root = Path(root).expanduser().resolve() / f"candidate_{int(component_id):04d}"
    for label, score in scores.items():
        directory = root / ("static" if int(label) == 0 else f"part_{int(label):02d}")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "scores.json").write_text(json.dumps(score, indent=2, allow_nan=False), encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    (root / "decision.json").write_text(json.dumps(decision, indent=2, allow_nan=False), encoding="utf-8")


def write_progressive_html(
    state: ProgressiveRigidState,
    path: str | Path,
    *,
    max_points: int = 20000,
) -> Path:
    """Self-contained Plotly scatter for labels/status/residuals."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    n = state.num_foreground
    if n > int(max_points):
        stride = max(1, n // int(max_points))
        ids = torch.arange(0, n, stride, device=state.device)[: int(max_points)]
    else:
        ids = torch.arange(n, device=state.device)
    xyz = state.canonical_fg[ids].detach().cpu()
    labels = state.hard_labels[ids].detach().cpu()
    status = state.status[ids].detach().cpu()
    prob = state.moving_probability()[ids].detach().cpu()
    residual = state.fm_rigid_residual.min(0).values[ids].detach().cpu()
    data = {
        "x": xyz[:, 0].tolist(),
        "y": xyz[:, 1].tolist(),
        "z": xyz[:, 2].tolist(),
        "label": labels.tolist(),
        "status": status.tolist(),
        "p_move": prob.tolist(),
        "residual": residual.tolist(),
    }
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>progressive rigid discovery</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body>
<h2>Progressive rigid discovery</h2>
<p>Foreground points shown: {int(ids.numel())} / {n}. Color = label, hover includes status/prob/residual.</p>
<div id="plot" style="width:100%;height:88vh;"></div>
<script>
const d = {json.dumps(data)};
Plotly.newPlot('plot', [{{
  type: 'scatter3d',
  mode: 'markers',
  x: d.x, y: d.y, z: d.z,
  marker: {{size: 2, color: d.label, colorscale: 'Viridis', opacity: 0.85}},
  text: d.label.map((v,i) => `label=${{v}} status=${{d.status[i]}} p=${{d.p_move[i].toFixed(3)}} r=${{d.residual[i].toFixed(4)}}`),
  hoverinfo: 'text'
}}], {{scene: {{aspectmode: 'data'}}, margin: {{l:0,r:0,b:0,t:30}}}});
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path
