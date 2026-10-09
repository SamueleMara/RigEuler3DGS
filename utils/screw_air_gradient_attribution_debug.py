from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F


def _cpu(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.as_tensor(x).detach().cpu()


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if v != v or v == float('inf') or v == -float('inf'):
            return default
        return v
    except Exception:
        return default


def _stats_1d(x: torch.Tensor) -> Dict[str, float]:
    x = _cpu(x).float().reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    qs = torch.quantile(x, torch.tensor([0.50, 0.75, 0.90, 0.95, 0.99])) if x.numel() > 1 else x.repeat(5)
    return {
        "count": int(x.numel()),
        "mean": _safe_float(x.mean().item()),
        "std": _safe_float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "min": _safe_float(x.min().item()),
        "p50": _safe_float(qs[0].item()),
        "p75": _safe_float(qs[1].item()),
        "p90": _safe_float(qs[2].item()),
        "p95": _safe_float(qs[3].item()),
        "p99": _safe_float(qs[4].item()),
        "max": _safe_float(x.max().item()),
    }


def _top_mass(x: torch.Tensor, frac: float) -> float:
    x = _cpu(x).float().reshape(-1).clamp_min(0.0)
    if x.numel() == 0:
        return 0.0
    s = x.sum().clamp_min(1e-12)
    k = max(1, int(round(float(frac) * int(x.numel()))))
    return _safe_float(torch.topk(x, k=min(k, int(x.numel())), largest=True).values.sum().div(s).item())


def compute_cluster_quality(
    xyz: torch.Tensor,
    foreground_mask: torch.Tensor,
    raw_cluster_id: torch.Tensor,
    dino_features: Optional[torch.Tensor] = None,
    semantic_probability: Optional[torch.Tensor] = None,
    k_graph: int = 8,
) -> Dict[str, Any]:
    """Measure whether Euclidean+DINO+semantic groups are spatial/feature coherent.

    This does not change labels.  It is intentionally diagnostic: cluster size,
    connectedness over a local KNN graph, compactness, DINO variance, and semantic
    purity are reported before any dynamic/static training decision is trusted.
    """
    xyz = _cpu(xyz).float()
    n = int(xyz.shape[0])
    fg = _cpu(foreground_mask).bool().reshape(-1) if foreground_mask is not None else torch.ones(n, dtype=torch.bool)
    raw = _cpu(raw_cluster_id).long().reshape(-1) if raw_cluster_id is not None else torch.full((n,), -1, dtype=torch.long)
    if raw.numel() != n:
        tmp = torch.full((n,), -1, dtype=torch.long)
        m = min(n, raw.numel())
        tmp[:m] = raw[:m]
        raw = tmp
    sem = _cpu(semantic_probability).float().reshape(-1) if semantic_probability is not None else torch.ones(n)
    if sem.numel() != n:
        sem = torch.ones(n)
    din = _cpu(dino_features).float() if dino_features is not None else None
    if din is not None and (din.ndim != 2 or din.shape[0] != n or din.shape[1] == 0):
        din = None
    if din is not None:
        din = F.normalize(din, dim=-1, eps=1e-6)

    fg_ids = torch.where(fg & (raw >= 0))[0]
    out: Dict[str, Any] = {
        "num_gaussians": n,
        "foreground_count": int(fg_ids.numel()),
        "k_graph": int(k_graph),
        "clusters": [],
    }
    if fg_ids.numel() == 0:
        return out

    # Local graph over foreground for connectedness of each cluster.
    k_graph = max(1, int(k_graph))
    X = xyz[fg_ids]
    # This can be O(N^2), but it is debug-only and N_fg is usually <~10k.
    try:
        D = torch.cdist(X.float(), X.float())
        kk = min(k_graph + 1, int(X.shape[0]))
        nn = torch.topk(D, k=kk, largest=False).indices[:, 1:]
    except Exception:
        nn = torch.empty((int(fg_ids.numel()), 0), dtype=torch.long)
    local_raw = raw[fg_ids]
    vals = torch.unique(local_raw[local_raw >= 0])
    for c in vals.tolist():
        c = int(c)
        loc = torch.where(local_raw == c)[0]
        ids = fg_ids[loc]
        if ids.numel() == 0:
            continue
        pts = xyz[ids]
        ctr = pts.mean(dim=0)
        dist = torch.linalg.norm(pts - ctr, dim=-1)
        bbox = pts.max(dim=0).values - pts.min(dim=0).values
        # Components using fg KNN graph restricted to this cluster.
        comp_count = 0
        largest_comp = 0
        if nn.numel() > 0:
            in_c = torch.zeros(int(fg_ids.numel()), dtype=torch.bool)
            in_c[loc] = True
            seen = torch.zeros(int(fg_ids.numel()), dtype=torch.bool)
            for seed in loc.tolist():
                if seen[seed]:
                    continue
                comp_count += 1
                stack = [seed]
                seen[seed] = True
                size = 0
                while stack:
                    u = stack.pop()
                    size += 1
                    for v in nn[u].tolist():
                        if (not seen[v]) and bool(in_c[v]):
                            seen[v] = True
                            stack.append(v)
                largest_comp = max(largest_comp, size)
        else:
            comp_count = 1
            largest_comp = int(ids.numel())
        dino_intra = 0.0
        dino_nn_other = 0.0
        if din is not None:
            fc = din[ids]
            fmean = F.normalize(fc.mean(dim=0), dim=0, eps=1e-6)
            dino_intra = _safe_float((1.0 - (fc @ fmean).clamp(-1, 1)).mean().item())
            other = fg_ids[local_raw != c]
            if other.numel() > 0:
                # approximate nearest other centroid by raw cluster centroids
                others = []
                for oc in vals.tolist():
                    oc = int(oc)
                    if oc == c:
                        continue
                    oids = fg_ids[local_raw == oc]
                    if oids.numel() > 0:
                        others.append(F.normalize(din[oids].mean(dim=0), dim=0, eps=1e-6))
                if others:
                    O = torch.stack(others, dim=0)
                    dino_nn_other = _safe_float((1.0 - (O @ fmean).max()).item())
        rec = {
            "cluster_id": c,
            "count": int(ids.numel()),
            "fraction_of_foreground": _safe_float(ids.numel() / max(1, int(fg_ids.numel()))),
            "center": [_safe_float(v) for v in ctr.tolist()],
            "bbox_extent": [_safe_float(v) for v in bbox.tolist()],
            "radius_mean": _safe_float(dist.mean().item()),
            "radius_p90": _safe_float(torch.quantile(dist, 0.90).item()) if dist.numel() > 1 else _safe_float(dist.max().item()),
            "semantic_mean": _safe_float(sem[ids].mean().item()),
            "semantic_std": _safe_float(sem[ids].std(unbiased=False).item()) if ids.numel() > 1 else 0.0,
            "dino_intra_mean_cosine_distance": dino_intra,
            "dino_nearest_other_centroid_distance": dino_nn_other,
            "connected_components": int(comp_count),
            "largest_component_fraction": _safe_float(largest_comp / max(1, int(ids.numel()))),
        }
        out["clusters"].append(rec)
    return out


def summarize_gradient_attribution(
    xyz: torch.Tensor,
    foreground_mask: torch.Tensor,
    raw_cluster_id: Optional[torch.Tensor],
    selected_moving_mask: Optional[torch.Tensor],
    stats: Dict[str, torch.Tensor],
    samples: list,
    sample_records: list,
    cluster_quality: Optional[Dict[str, Any]] = None,
    top_k: int = 128,
) -> Dict[str, Any]:
    xyz = _cpu(xyz).float()
    n = int(xyz.shape[0])
    fg = _cpu(foreground_mask).bool().reshape(-1) if foreground_mask is not None else torch.ones(n, dtype=torch.bool)
    raw = _cpu(raw_cluster_id).long().reshape(-1) if raw_cluster_id is not None else torch.full((n,), -1, dtype=torch.long)
    moving = _cpu(selected_moving_mask).bool().reshape(-1) if selected_moving_mask is not None else torch.zeros(n, dtype=torch.bool)
    # Tensors
    total = _cpu(stats.get("grad_total_mean", torch.zeros(n))).float().reshape(-1)
    f0 = _cpu(stats.get("grad_static_frame0_mean", stats.get("grad_frame0_mean", torch.zeros(n)))).float().reshape(-1)
    dyn = _cpu(stats.get("grad_motion_future_mean", stats.get("grad_nonzero_mean", torch.zeros(n)))).float().reshape(-1)
    excess = _cpu(stats.get("grad_motion_excess", dyn - f0)).float().reshape(-1)
    excess_pos = _cpu(stats.get("grad_motion_excess_positive", excess.clamp_min(0.0))).float().reshape(-1)
    ratio = _cpu(stats.get("grad_motion_ratio", dyn / f0.clamp_min(1e-6))).float().reshape(-1)
    raw_future = _cpu(stats.get("raw_future_evidence", dyn)).float().reshape(-1)
    selected_evidence = _cpu(stats.get("selected_motion_evidence", excess_pos)).float().reshape(-1)
    if total.numel() != n:
        total = torch.zeros(n)
    if f0.numel() != n:
        f0 = torch.zeros(n)
    if dyn.numel() != n:
        dyn = torch.zeros(n)
    if excess.numel() != n:
        excess = dyn - f0
    if excess_pos.numel() != n:
        excess_pos = excess.clamp_min(0.0)
    if ratio.numel() != n:
        ratio = dyn / f0.clamp_min(1e-6)
    if raw_future.numel() != n:
        raw_future = dyn
    if selected_evidence.numel() != n:
        selected_evidence = excess_pos
    fg_total = total[fg]
    fg_excess = excess[fg]
    fg_excess_pos = excess_pos[fg]
    fg_ratio = ratio[fg]
    top_k = max(1, min(int(top_k), n))
    top_total_vals, top_total_idx = torch.topk(total.clamp_min(0), k=top_k, largest=True)
    top_excess_vals, top_excess_idx = torch.topk(excess_pos.clamp_min(0), k=top_k, largest=True)
    top_ratio_vals, top_ratio_idx = torch.topk(ratio.clamp_min(0), k=top_k, largest=True)

    def _mask_mass(x, mask):
        x = x.float().clamp_min(0)
        denom = x[fg].sum().clamp_min(1e-12)
        return _safe_float(x[mask & fg].sum().div(denom).item())

    out: Dict[str, Any] = {
        "num_gaussians": n,
        "foreground_count": int(fg.sum().item()),
        "selected_dynamic_count": int((moving & fg).sum().item()),
        "samples_count": len(samples),
        "loss_samples_count": len(sample_records),
        "measurement_definition": {
            "static_responsibility": "render canonical/frame-0 Gaussians against frame-0 targets for every camera and back-project residual responsibility through differentiable Gaussian channels",
            "motion_responsibility": "render the same canonical/frame-0 Gaussians against future-frame targets and back-project residual responsibility",
            "motion_excess": "motion_responsibility - static_responsibility, per Gaussian",
            "motion_ratio": "motion_responsibility / static_responsibility, per Gaussian",
            "channels": "combined, xyz, screen/viewspace, opacity, appearance, dino feature, shape",
        },
        "gradient_total_foreground_stats": _stats_1d(fg_total),
        "gradient_static_frame0_foreground_stats": _stats_1d(f0[fg]),
        "gradient_motion_future_foreground_stats": _stats_1d(dyn[fg]),
        "gradient_motion_excess_foreground_stats": _stats_1d(fg_excess),
        "gradient_motion_excess_positive_foreground_stats": _stats_1d(fg_excess_pos),
        "gradient_motion_ratio_foreground_stats": _stats_1d(fg_ratio),
        "dominance_future_top_1pct_mass": _top_mass(dyn[fg], 0.01),
        "dominance_future_top_5pct_mass": _top_mass(dyn[fg], 0.05),
        "dominance_static_top_1pct_mass": _top_mass(f0[fg], 0.01),
        "dominance_static_top_5pct_mass": _top_mass(f0[fg], 0.05),
        "dominance_excess_positive_top_1pct_mass": _top_mass(fg_excess_pos, 0.01),
        "dominance_excess_positive_top_5pct_mass": _top_mass(fg_excess_pos, 0.05),
        "selected_dynamic_mass_fraction_total_grad": _mask_mass(total, moving),
        "selected_dynamic_mass_fraction_static_frame0_grad": _mask_mass(f0, moving),
        "selected_dynamic_mass_fraction_motion_future_grad": _mask_mass(dyn, moving),
        "selected_dynamic_mass_fraction_motion_excess_positive": _mask_mass(excess_pos, moving),
        "selected_dynamic_mass_fraction_selected_motion_evidence": _mask_mass(selected_evidence, moving),
        "sample_records": sample_records,
        "top_gaussians_by_total_grad": [
            {"gaussian_id": int(i.item()), "value": _safe_float(v.item()), "xyz": [_safe_float(q) for q in xyz[int(i.item())].tolist()], "foreground": bool(fg[int(i.item())]), "moving_selected": bool(moving[int(i.item())]), "raw_cluster": int(raw[int(i.item())].item()) if raw.numel() == n else -1}
            for v, i in zip(top_total_vals, top_total_idx)
        ],
        "top_gaussians_by_motion_excess_positive": [
            {"gaussian_id": int(i.item()), "value": _safe_float(v.item()), "xyz": [_safe_float(q) for q in xyz[int(i.item())].tolist()], "foreground": bool(fg[int(i.item())]), "moving_selected": bool(moving[int(i.item())]), "raw_cluster": int(raw[int(i.item())].item()) if raw.numel() == n else -1}
            for v, i in zip(top_excess_vals, top_excess_idx)
        ],
        "top_gaussians_by_motion_ratio": [
            {"gaussian_id": int(i.item()), "value": _safe_float(v.item()), "xyz": [_safe_float(q) for q in xyz[int(i.item())].tolist()], "foreground": bool(fg[int(i.item())]), "moving_selected": bool(moving[int(i.item())]), "raw_cluster": int(raw[int(i.item())].item()) if raw.numel() == n else -1}
            for v, i in zip(top_ratio_vals, top_ratio_idx)
        ],
        "cluster_quality": cluster_quality or {},
        "cluster_gradient_stats": [],
    }
    if raw.numel() == n:
        for c in torch.unique(raw[(raw >= 0) & fg]).tolist():
            c = int(c)
            m = (raw == c) & fg
            out["cluster_gradient_stats"].append({
                "cluster_id": c,
                "count": int(m.sum().item()),
                "selected_dynamic_fraction": _safe_float((moving[m].float().mean().item()) if m.any() else 0.0),
                "total_grad_mean": _safe_float(total[m].mean().item()) if m.any() else 0.0,
                "motion_future_grad_mean": _safe_float(dyn[m].mean().item()) if m.any() else 0.0,
                "static_frame0_grad_mean": _safe_float(f0[m].mean().item()) if m.any() else 0.0,
                "motion_excess_mean": _safe_float(excess[m].mean().item()) if m.any() else 0.0,
                "motion_excess_positive_mean": _safe_float(excess_pos[m].mean().item()) if m.any() else 0.0,
                "motion_ratio_mean": _safe_float(ratio[m].mean().item()) if m.any() else 0.0,
                "total_grad_mass_fraction": _mask_mass(total, m),
                "static_frame0_grad_mass_fraction": _mask_mass(f0, m),
                "motion_future_grad_mass_fraction": _mask_mass(dyn, m),
                "motion_excess_positive_mass_fraction": _mask_mass(excess_pos, m),
            })
    return out


def write_gradient_attribution_debug(
    out_dir,
    xyz: torch.Tensor,
    foreground_mask: torch.Tensor,
    raw_cluster_id: Optional[torch.Tensor],
    selected_moving_mask: Optional[torch.Tensor],
    stats: Dict[str, torch.Tensor],
    samples: list,
    sample_records: list,
    cluster_quality: Optional[Dict[str, Any]] = None,
    max_points: int = 30000,
    top_k: int = 128,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz_cpu = _cpu(xyz).float()
    n = int(xyz_cpu.shape[0])
    fg = _cpu(foreground_mask).bool().reshape(-1) if foreground_mask is not None else torch.ones(n, dtype=torch.bool)
    raw = _cpu(raw_cluster_id).long().reshape(-1) if raw_cluster_id is not None else torch.full((n,), -1, dtype=torch.long)
    moving = _cpu(selected_moving_mask).bool().reshape(-1) if selected_moving_mask is not None else torch.zeros(n, dtype=torch.bool)
    # Ensure lengths
    def _fix(x, fill=0.0):
        x = _cpu(x).float().reshape(-1)
        if x.numel() == n:
            return x
        y = torch.full((n,), float(fill), dtype=torch.float32)
        m = min(n, x.numel())
        y[:m] = x[:m]
        return y
    total = _fix(stats.get("grad_total_mean", torch.zeros(n)))
    f0 = _fix(stats.get("grad_static_frame0_mean", stats.get("grad_frame0_mean", torch.zeros(n))))
    dyn = _fix(stats.get("grad_motion_future_mean", stats.get("grad_nonzero_mean", torch.zeros(n))))
    excess = _fix(stats.get("grad_motion_excess", dyn - f0))
    excess_pos = _fix(stats.get("grad_motion_excess_positive", excess.clamp_min(0.0)))
    ratio = _fix(stats.get("grad_motion_ratio", dyn / f0.clamp_min(1e-6)))
    raw_future = _fix(stats.get("raw_future_evidence", dyn))
    selected_evidence = _fix(stats.get("selected_motion_evidence", excess_pos))
    # Optional v25 residual-responsibility channel maps.
    attr_maps = {}
    for _name, _value in (stats or {}).items():
        if isinstance(_name, str) and _name.startswith("attr_"):
            try:
                attr_maps[_name] = _fix(_value)
            except Exception:
                pass
    counts = _fix(stats.get("grad_count", torch.zeros(n)))
    summary = summarize_gradient_attribution(
        xyz=xyz_cpu,
        foreground_mask=fg,
        raw_cluster_id=raw,
        selected_moving_mask=moving,
        stats={
            "grad_total_mean": total,
            "grad_static_frame0_mean": f0,
            "grad_motion_future_mean": dyn,
            "grad_motion_excess": excess,
            "grad_motion_excess_positive": excess_pos,
            "grad_motion_ratio": ratio,
            "raw_future_evidence": raw_future,
            "selected_motion_evidence": selected_evidence,
            "grad_count": counts,
        },
        samples=samples,
        sample_records=sample_records,
        cluster_quality=cluster_quality,
        top_k=top_k,
    )
    meta = {k[5:]: v for k, v in (stats or {}).items() if isinstance(k, str) and k.startswith("meta_") and isinstance(v, (str, int, float, bool))}
    if meta:
        summary["metadata"] = meta
    if attr_maps:
        summary["v25_attribution_channel_stats"] = {k: _stats_1d(v[fg]) for k, v in attr_maps.items()}
        for k, v in attr_maps.items():
            if k.endswith("_motion_excess_positive"):
                summary[f"dominance_{k}_top_1pct_mass"] = _top_mass(v[fg], 0.01)
                summary[f"dominance_{k}_top_5pct_mass"] = _top_mass(v[fg], 0.05)
    with open(out_dir / "static_dynamic_gradient_attribution.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({
        "xyz": xyz_cpu,
        "foreground_mask": fg,
        "raw_cluster_id": raw,
        "selected_moving_mask": moving,
        "grad_total_mean": total,
        "grad_static_frame0_mean": f0,
        "grad_motion_future_mean": dyn,
        "grad_motion_excess": excess,
        "grad_motion_excess_positive": excess_pos,
        "grad_motion_ratio": ratio,
        "raw_future_evidence": raw_future,
        "selected_motion_evidence": selected_evidence,
        **attr_maps,
        "grad_count": counts,
        "sample_records": sample_records,
        "cluster_quality": cluster_quality or {},
        "summary": summary,
    }, out_dir / "static_dynamic_gradient_attribution.pt")
    # Sample points for HTML, preserving foreground and high-gradient points.
    ids = torch.arange(n)
    keep = torch.where(fg | moving)[0]
    top_extra = torch.topk(excess.clamp_min(0), k=min(max(1, max_points // 5), n), largest=True).indices
    keep = torch.unique(torch.cat([keep, top_extra], dim=0))
    if keep.numel() > int(max_points):
        gen = torch.Generator(device="cpu"); gen.manual_seed(12345)
        keep = keep[torch.randperm(keep.numel(), generator=gen)[:int(max_points)]]
    elif n > int(max_points):
        rest = ids[~torch.isin(ids, keep)] if hasattr(torch, "isin") else ids[~fg]
        need = max(0, int(max_points) - int(keep.numel()))
        if rest.numel() > need:
            gen = torch.Generator(device="cpu"); gen.manual_seed(23456)
            rest = rest[torch.randperm(rest.numel(), generator=gen)[:need]]
        keep = torch.unique(torch.cat([keep, rest], dim=0))
    X = xyz_cpu[keep]
    data = {
        "title": "ScrewAIR frame0-vs-future residual-responsibility debug",
        "points": X.tolist(),
        "foreground": fg[keep].int().tolist(),
        "moving": moving[keep].int().tolist(),
        "raw_cluster": raw[keep].tolist(),
        "grad_total": total[keep].tolist(),
        "grad_static_frame0": f0[keep].tolist(),
        "grad_motion_future": dyn[keep].tolist(),
        "grad_motion_excess": excess[keep].tolist(),
        "grad_motion_excess_positive": excess_pos[keep].tolist(),
        "grad_motion_ratio": ratio[keep].tolist(),
        "raw_future_evidence": raw_future[keep].tolist(),
        "selected_motion_evidence": selected_evidence[keep].tolist(),
        "attr_maps": {k: v[keep].tolist() for k, v in attr_maps.items()},
        "grad_count": counts[keep].tolist(),
        "summary": summary,
    }
    json_data = json.dumps(data)
    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'/><title>ScrewAIR gradient attribution debug</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<style>
body {{ margin:0; font-family:Arial,sans-serif; color:#203550; }}
#top {{ height:10vh; padding:8px 12px; box-sizing:border-box; }}
#plot {{ width:100vw; height:90vh; }}
</style></head>
<body>
<div id='top'><b>ScrewAIR frame0-vs-future residual-responsibility debug</b><br/>
Views compare residual responsibility of canonical frame-0 Gaussians. Static = render canonical Gaussians vs frame-0 targets. Motion = render the same canonical Gaussians vs later-frame targets. v25 back-projects residuals through xyz, screen-space, opacity, appearance, DINO, and shape channels; motion_excess = motion_future - static_frame0.</div>
<div id='plot'></div>
<script>
const S={json_data};
const pts=S.points; const x=pts.map(p=>p[0]), y=pts.map(p=>p[1]), z=pts.map(p=>p[2]);
function ids(fn){{const out=[]; for(let i=0;i<pts.length;i++) if(fn(i)) out.push(i); return out;}}
function sub(a,I){{return I.map(i=>a[i]);}}
const idBg=ids(i=>S.foreground[i]===0);
const idSta=ids(i=>S.foreground[i]===1 && S.moving[i]===0);
const idDyn=ids(i=>S.foreground[i]===1 && S.moving[i]===1);
const traces=[];
traces.push({{type:'scatter3d',mode:'markers',name:'selected static foreground',x:sub(x,idSta),y:sub(y,idSta),z:sub(z,idSta),marker:{{size:2,color:'rgba(50,120,255,0.55)'}}}});
traces.push({{type:'scatter3d',mode:'markers',name:'selected dynamic foreground',x:sub(x,idDyn),y:sub(y,idDyn),z:sub(z,idDyn),marker:{{size:2,color:'rgba(230,30,30,0.70)'}}}});
traces.push({{type:'scatter3d',mode:'markers',name:'background',x:sub(x,idBg),y:sub(y,idBg),z:sub(z,idBg),marker:{{size:2,color:'rgba(60,60,60,0.15)'}}}});
function heat(name, vals, visible){{traces.push({{type:'scatter3d',mode:'markers',name:name,visible:visible,x:x,y:y,z:z,marker:{{size:2,color:vals,colorscale:'Turbo',showscale:true,colorbar:{{title:name,x:-0.08}}}}}});}}
heat('grad_total_mean', S.grad_total, 'legendonly');
heat('static_frame0_grad', S.grad_static_frame0, 'legendonly');
heat('motion_future_static_pose_grad', S.grad_motion_future, true);
heat('motion_excess_future_minus_static', S.grad_motion_excess, 'legendonly');
heat('motion_excess_positive', S.grad_motion_excess_positive, 'legendonly');
heat('motion_ratio_future_over_static', S.grad_motion_ratio, 'legendonly');
heat('selected_motion_evidence_used_by_init', S.selected_motion_evidence, 'legendonly');
if (S.attr_maps) {{
  const attrNames = Object.keys(S.attr_maps).sort();
  for (const name of attrNames) {{
    const show = name === 'attr_combined_motion_excess_positive' ? true : 'legendonly';
    heat(name, S.attr_maps[name], show);
  }}
}}
heat('grad_count_visibility_proxy', S.grad_count, 'legendonly');
const layout={{title:S.title, scene:{{domain:{{x:[0.08,0.82],y:[0,1]}},aspectmode:'data'}}, margin:{{l:120,r:280,t:60,b:30}}, legend:{{x:0.85,y:1.0}} }};
Plotly.newPlot('plot', traces, layout, {{responsive:true}});
</script></body></html>"""
    (out_dir / "static_dynamic_gradient_attribution.html").write_text(html)
    # Cluster quality separately for easy grep.
    if cluster_quality is not None:
        with open(out_dir / "cluster_quality.json", "w") as f:
            json.dump(cluster_quality, f, indent=2)
    return out_dir / "static_dynamic_gradient_attribution.html"
