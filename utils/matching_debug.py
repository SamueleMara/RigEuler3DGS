"""Debug helpers for the standalone bidirectional matcher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def binary_metrics(probability: torch.Tensor, truth: torch.Tensor, *, threshold: float = 0.5) -> dict[str, float]:
    pred = torch.as_tensor(probability).detach().cpu().float() >= float(threshold)
    truth = torch.as_tensor(truth).detach().cpu().bool()
    if pred.numel() != truth.numel():
        raise ValueError("probability and truth must have the same length")
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(1, pred.numel()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "ground_truth_used_as_algorithm_input": False,
    }


def tensor_summary(value: torch.Tensor) -> dict[str, float]:
    x = torch.as_tensor(value).detach().cpu().float().reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p90": float(torch.quantile(x, 0.9)) if x.numel() > 1 else float(x[0]),
    }


def _sample(ids: torch.Tensor, max_points: int, seed: int) -> torch.Tensor:
    ids = torch.as_tensor(ids).detach().cpu().long().reshape(-1)
    if ids.numel() <= int(max_points):
        return ids
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(ids.numel(), generator=gen)[: int(max_points)]
    return ids[order.sort().values]


def write_matching_html(
    path: str | Path,
    *,
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    transformed_source_xyz: torch.Tensor,
    forward_endpoint: torch.Tensor,
    backward_endpoint: torch.Tensor,
    active_source_mask: torch.Tensor,
    active_target_mask: torch.Tensor,
    match_source_ids: torch.Tensor,
    match_target_ids: torch.Tensor,
    match_energy: torch.Tensor | None = None,
    max_points: int = 3000,
    seed: int = 0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    src = torch.as_tensor(source_xyz).detach().cpu().float()
    tgt = torch.as_tensor(target_xyz).detach().cpu().float()
    moved = torch.as_tensor(transformed_source_xyz).detach().cpu().float()
    fep = torch.as_tensor(forward_endpoint).detach().cpu().float()
    bep = torch.as_tensor(backward_endpoint).detach().cpu().float()
    active_src_ids = _sample(torch.nonzero(active_source_mask.detach().cpu().bool(), as_tuple=False).reshape(-1), max_points, seed)
    active_tgt_ids = _sample(torch.nonzero(active_target_mask.detach().cpu().bool(), as_tuple=False).reshape(-1), max_points, seed + 1)
    ms = torch.as_tensor(match_source_ids).detach().cpu().long()
    mt = torch.as_tensor(match_target_ids).detach().cpu().long()
    match_sample = _sample(torch.arange(ms.numel()), max(1, max_points // 2), seed + 2)
    ms = ms[match_sample]
    mt = mt[match_sample]
    energy = (
        torch.zeros(ms.numel())
        if match_energy is None or torch.as_tensor(match_energy).numel() == 0
        else torch.as_tensor(match_energy).detach().cpu().float()[match_sample]
    )

    def pts(arr: torch.Tensor) -> list[list[float]]:
        return [[float(v) for v in row] for row in arr]

    line_xyz = []
    for s, t in zip(ms.tolist(), mt.tolist()):
        line_xyz += [src[s].tolist(), tgt[t].tolist(), [None, None, None]]
    fm_arrows = []
    for s in active_src_ids[: max(1, max_points // 3)].tolist():
        fm_arrows += [src[s].tolist(), fep[s].tolist(), [None, None, None]]
    backward_arrows = []
    for t in active_tgt_ids[: max(1, max_points // 3)].tolist():
        backward_arrows += [tgt[t].tolist(), bep[t].tolist(), [None, None, None]]
    payload = {
        "source": pts(src[active_src_ids]),
        "target": pts(tgt[active_tgt_ids]),
        "moved": pts(moved[active_src_ids]),
        "matches": line_xyz,
        "fm_arrows": fm_arrows,
        "backward_arrows": backward_arrows,
        "match_energy": [float(v) for v in energy],
    }
    html = """<!doctype html><meta charset="utf-8"><title>Bidirectional FM matching</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
const p = PAYLOAD;
function splitLines(rows){ return {x:rows.map(v=>v[0]), y:rows.map(v=>v[1]), z:rows.map(v=>v[2])}; }
const m = splitLines(p.matches), f = splitLines(p.fm_arrows), b = splitLines(p.backward_arrows);
Plotly.newPlot('plot', [
 {type:'scatter3d', mode:'markers', name:'frame0 dynamic sources', x:p.source.map(v=>v[0]), y:p.source.map(v=>v[1]), z:p.source.map(v=>v[2]), marker:{size:2,color:'#1f77b4'}},
 {type:'scatter3d', mode:'markers', name:'frameT backward-dynamic targets', x:p.target.map(v=>v[0]), y:p.target.map(v=>v[1]), z:p.target.map(v=>v[2]), marker:{size:2,color:'#ff7f0e'}},
 {type:'scatter3d', mode:'markers', name:'rigid transformed sources', x:p.moved.map(v=>v[0]), y:p.moved.map(v=>v[1]), z:p.moved.map(v=>v[2]), marker:{size:2,color:'#2ca02c'}},
 {type:'scatter3d', mode:'lines', name:'reciprocal matches', x:m.x, y:m.y, z:m.z, line:{width:2,color:'rgba(20,20,20,0.35)'}},
 {type:'scatter3d', mode:'lines', name:'forward FM arrows', x:f.x, y:f.y, z:f.z, line:{width:1,color:'rgba(31,119,180,0.25)'}},
 {type:'scatter3d', mode:'lines', name:'backward FM arrows', x:b.x, y:b.y, z:b.z, line:{width:1,color:'rgba(255,127,14,0.25)'}}
], {title:'Bidirectional FM + diffusion reciprocal matching', scene:{aspectmode:'data'}, margin:{l:0,r:0,t:40,b:0}});
</script>"""
    path.write_text(html.replace("PAYLOAD", json.dumps(payload)), encoding="utf-8")
