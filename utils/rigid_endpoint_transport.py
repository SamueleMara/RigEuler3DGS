"""Rigid-conditioned endpoint transport evidence for K-part discovery."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.rigid_killing_motion import integrate_rigid_mode_exact


@dataclass
class EndpointTransportConfig:
    enabled: bool = False
    knn: int = 16
    radius: float = 0.05
    temperature: float = 0.02
    geometry_weight: float = 1.0
    normal_weight: float = 0.0
    sam3_weight: float = 0.0
    confidence_min: float = 0.05
    robust_delta: float = 0.02
    metric: str = "point"
    mode: str = "local_soft"
    source_chunk_size: int = 1024
    target_chunk_size: int = 16384


@dataclass
class EndpointTransportResult:
    energy: torch.Tensor
    confidence: torch.Tensor
    barycentric_target: torch.Tensor
    fm_velocity: torch.Tensor
    nearest_distance: torch.Tensor
    entropy: torch.Tensor
    target_indices: torch.Tensor
    transport_weights: torch.Tensor
    delta_vs_static: torch.Tensor
    valid: torch.Tensor


def _coordinates(speed, frame_times):
    speed = torch.as_tensor(speed).detach().cpu().double()
    time = torch.as_tensor(frame_times).detach().cpu().double()
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.float64),
            torch.cumsum(speed * torch.diff(time), 0),
        )
    )


@torch.no_grad()
def integrate_endpoint_hypotheses(canonical_xyz, fields, frame_times):
    """Return exact endpoint centers for class 0 identity and class 1..K fields."""
    source_input = torch.as_tensor(canonical_xyz)
    if source_input.requires_grad:
        raise RuntimeError("Endpoint transport received trainable canonical XYZ")
    source = source_input.detach()
    columns = [source]
    for field in fields:
        if not bool(field.get("valid", False)):
            columns.append(source.clone())
            continue
        coord = _coordinates(field["temporal_speed"], frame_times)[-1]
        mode = field["mode"].to(device=source.device, dtype=source.dtype)
        moved = integrate_rigid_mode_exact(source, mode, coord.to(device=source.device, dtype=source.dtype))
        columns.append(moved["positions"].detach())
    return torch.stack(columns, dim=1)


class RigidEndpointTransport:
    def __init__(self, endpoint_model_or_state, cfg: EndpointTransportConfig):
        if cfg.mode != "local_soft":
            raise RuntimeError("Only endpoint_transport_mode=local_soft is implemented")
        self.cfg = cfg
        if isinstance(endpoint_model_or_state, dict):
            target_xyz = endpoint_model_or_state.get("xyz")
            target_normals = endpoint_model_or_state.get("normals")
            target_sam3 = endpoint_model_or_state.get("sam3_features")
            self.target_path = str(endpoint_model_or_state.get("point_cloud_path", ""))
            self.frame = int(endpoint_model_or_state.get("frame", -1))
        else:
            target_xyz = endpoint_model_or_state.get_xyz
            target_normals = None
            target_sam3 = getattr(endpoint_model_or_state, "get_sam3_feature_field", None)
            self.target_path = ""
            self.frame = -1
        if target_xyz is None:
            raise ValueError("Endpoint transport requires endpoint target xyz")
        self.target_xyz = torch.as_tensor(target_xyz).detach().float()
        self.target_normals = (
            None
            if target_normals is None
            else F.normalize(torch.as_tensor(target_normals).detach().float(), dim=-1)
        )
        self.target_sam3 = (
            None
            if target_sam3 is None or not torch.is_tensor(target_sam3) or target_sam3.numel() == 0
            else F.normalize(torch.as_tensor(target_sam3).detach().float(), dim=-1)
        )

    def _targets_on(self, device):
        xyz = self.target_xyz.to(device=device, dtype=torch.float32)
        normals = None if self.target_normals is None else self.target_normals.to(device=device, dtype=torch.float32)
        sam3 = None if self.target_sam3 is None else self.target_sam3.to(device=device, dtype=torch.float32)
        return xyz, normals, sam3

    @torch.no_grad()
    def build_target_index(self):
        return {
            "mode": "chunked_exact_knn",
            "target_count": int(self.target_xyz.shape[0]),
            "knn": int(self.cfg.knn),
            "radius": float(self.cfg.radius),
        }

    @torch.no_grad()
    def _knn(self, query, target):
        query = query.detach().float()
        target = target.detach().float()
        count = int(query.shape[0])
        k = min(int(self.cfg.knn), int(target.shape[0]))
        if count == 0 or k <= 0:
            return (
                torch.empty((count, 0), device=query.device),
                torch.empty((count, 0), device=query.device, dtype=torch.long),
            )
        best_d = torch.full((count, k), float("inf"), device=query.device)
        best_i = torch.full((count, k), -1, device=query.device, dtype=torch.long)
        target_chunk_size = max(k, int(self.cfg.target_chunk_size))
        for start in range(0, int(target.shape[0]), target_chunk_size):
            target_chunk = target[start : start + target_chunk_size]
            if target_chunk.numel() == 0:
                continue
            dist = torch.cdist(query, target_chunk).square()
            local_k = min(k, int(target_chunk.shape[0]))
            d, i = dist.topk(local_k, largest=False, dim=-1)
            i = i + int(start)
            merged_d = torch.cat((best_d, d), dim=-1)
            merged_i = torch.cat((best_i, i), dim=-1)
            best_d, order = merged_d.topk(k, largest=False, dim=-1)
            best_i = merged_i.gather(-1, order)
        return best_d, best_i

    @torch.no_grad()
    def evaluate(self, canonical_xyz, predicted_xyz_by_class, canonical_sam3=None):
        source_input = torch.as_tensor(canonical_xyz)
        predicted_input = torch.as_tensor(predicted_xyz_by_class)
        if source_input.requires_grad or predicted_input.requires_grad:
            raise RuntimeError("Endpoint transport must be read-only with respect to source geometry")
        source = source_input.detach().float()
        predicted = predicted_input.detach().float()
        if predicted.ndim != 3 or predicted.shape[0] != source.shape[0] or predicted.shape[-1] != 3:
            raise ValueError("predicted_xyz_by_class must be [N,K+1,3]")
        device = predicted.device
        source = source.to(device=device)
        target_xyz, target_normals, target_sam3 = self._targets_on(device)
        if target_xyz.numel() == 0:
            raise RuntimeError("Endpoint target model has no Gaussians")
        source_sam3 = None
        if canonical_sam3 is not None and torch.is_tensor(canonical_sam3) and canonical_sam3.numel() > 0:
            source_sam3 = F.normalize(torch.as_tensor(canonical_sam3).detach().float().to(device), dim=-1)

        n, classes = int(predicted.shape[0]), int(predicted.shape[1])
        k = min(int(self.cfg.knn), int(target_xyz.shape[0]))
        energy = torch.zeros((n, classes), device=device)
        confidence = torch.zeros_like(energy)
        nearest = torch.full_like(energy, float("inf"))
        entropy = torch.zeros_like(energy)
        valid = torch.zeros((n, classes), device=device, dtype=torch.bool)
        bary = torch.zeros((n, classes, 3), device=device)
        target_indices = torch.full((n, classes, k), -1, device=device, dtype=torch.long)
        weights = torch.zeros((n, classes, k), device=device)
        source_chunk = max(1, int(self.cfg.source_chunk_size))

        for label in range(classes):
            query_all = predicted[:, label].contiguous()
            for start in range(0, n, source_chunk):
                end = min(start + source_chunk, n)
                query = query_all[start:end]
                dist2, idx = self._knn(query, target_xyz)
                if idx.numel() == 0:
                    continue
                candidate_valid = idx >= 0
                if float(self.cfg.radius) > 0.0:
                    candidate_valid &= dist2.sqrt() <= float(self.cfg.radius)
                gathered_xyz = target_xyz[idx.clamp_min(0)]
                geom = torch.sqrt(dist2.clamp_min(0.0) + 1.0e-12)
                cap = max(float(self.cfg.robust_delta), 1.0e-8)
                geom = geom.clamp_max(4.0 * cap)
                cost = float(self.cfg.geometry_weight) * geom
                if (
                    float(self.cfg.normal_weight) > 0.0
                    and str(self.cfg.metric) == "point_plane"
                    and target_normals is not None
                ):
                    gathered_normals = target_normals[idx.clamp_min(0)]
                    plane = ((query[:, None] - gathered_xyz) * gathered_normals).sum(-1).square()
                    cost = cost + float(self.cfg.normal_weight) * plane
                if (
                    float(self.cfg.sam3_weight) > 0.0
                    and source_sam3 is not None
                    and target_sam3 is not None
                ):
                    dim = min(source_sam3.shape[1], target_sam3.shape[1])
                    gathered_sam3 = target_sam3[idx.clamp_min(0), :dim]
                    src = source_sam3[start:end, :dim]
                    sam_cost = 1.0 - (src[:, None] * gathered_sam3).sum(-1).clamp(-1.0, 1.0)
                    cost = cost + float(self.cfg.sam3_weight) * sam_cost
                large = torch.full_like(cost, 1.0e6)
                masked_cost = torch.where(candidate_valid, cost, large)
                row_valid = candidate_valid.any(-1)
                logit = -masked_cost / max(float(self.cfg.temperature), 1.0e-6)
                local_w = torch.softmax(logit, dim=-1)
                local_w = torch.where(candidate_valid, local_w, torch.zeros_like(local_w))
                local_w = local_w / local_w.sum(-1, keepdim=True).clamp_min(1.0e-8)
                local_energy = (local_w * torch.where(candidate_valid, cost, torch.zeros_like(cost))).sum(-1)
                local_bary = (local_w[..., None] * gathered_xyz).sum(-2)
                local_entropy = -(local_w * torch.log(local_w.clamp_min(1.0e-8))).sum(-1)
                if k > 1:
                    local_entropy = local_entropy / math.log(float(k))
                local_nearest = dist2[:, 0].sqrt()
                local_confidence = torch.exp(-local_nearest / cap) * (1.0 - local_entropy).clamp(0.0, 1.0)
                local_confidence = torch.where(row_valid, local_confidence, torch.zeros_like(local_confidence))

                energy[start:end, label] = torch.where(row_valid, local_energy, torch.zeros_like(local_energy))
                confidence[start:end, label] = local_confidence.clamp(0.0, 1.0)
                nearest[start:end, label] = local_nearest
                entropy[start:end, label] = local_entropy
                valid[start:end, label] = row_valid
                bary[start:end, label] = local_bary
                target_indices[start:end, label] = idx
                weights[start:end, label] = local_w
        fm_velocity = bary - source[:, None]
        delta_vs_static = energy[:, :1] - energy
        return EndpointTransportResult(
            energy=energy.detach(),
            confidence=confidence.detach(),
            barycentric_target=bary.detach(),
            fm_velocity=fm_velocity.detach(),
            nearest_distance=nearest.detach(),
            entropy=entropy.detach(),
            target_indices=target_indices.detach().cpu(),
            transport_weights=weights.detach().cpu(),
            delta_vs_static=delta_vs_static.detach(),
            valid=(valid & (confidence >= float(self.cfg.confidence_min))).detach(),
        )


def _sample_ids(score, max_points):
    count = int(score.shape[0])
    if count <= int(max_points):
        return torch.arange(count)
    order = torch.argsort(torch.as_tensor(score).detach().cpu().float(), descending=True)
    head = order[: min(count, int(max_points) // 2)]
    rest_count = max(0, int(max_points) - int(head.numel()))
    if rest_count == 0:
        return head.sort().values
    remaining = torch.linspace(0, count - 1, rest_count).round().long().unique()
    return torch.unique(torch.cat((head, remaining))).sort().values[: int(max_points)]


@torch.no_grad()
def write_endpoint_transport_diagnostics(
    result: EndpointTransportResult,
    source_xyz,
    predicted_xyz_by_class,
    output_dir,
    cycle,
    *,
    max_points=1000,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"cycle_{int(cycle):03d}"
    torch.save(result.energy.cpu(), f"{prefix}_endpoint_energy.pt")
    torch.save(result.confidence.cpu(), f"{prefix}_endpoint_confidence.pt")
    torch.save(result.delta_vs_static.cpu(), f"{prefix}_delta_vs_static.pt")
    torch.save(result.barycentric_target.cpu(), f"{prefix}_barycentric_targets.pt")
    torch.save(result.entropy.cpu(), f"{prefix}_transport_entropy.pt")
    torch.save(
        {
            "target_indices": result.target_indices,
            "transport_weights": result.transport_weights,
            "valid": result.valid.cpu(),
        },
        f"{prefix}_sparse_transport.pt",
    )
    _write_transport_vectors_html(
        output / f"cycle_{int(cycle):03d}_transport_vectors.html",
        source_xyz,
        predicted_xyz_by_class,
        result,
        max_points=max_points,
    )
    _write_endpoint_energy_html(
        output / f"cycle_{int(cycle):03d}_endpoint_energy.html",
        source_xyz,
        result,
        max_points=max_points,
    )


def _plotly_html(path, payload, script):
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
        "</head><body><div id='plot' style='width:100%;height:95vh'></div>"
        f"<script>const payload={json.dumps(payload)};{script}</script></body></html>"
    )
    Path(path).write_text(document, encoding="utf-8")


def _write_transport_vectors_html(path, source_xyz, predicted_xyz_by_class, result, *, max_points):
    source = torch.as_tensor(source_xyz).detach().cpu().float()
    predicted = torch.as_tensor(predicted_xyz_by_class).detach().cpu().float()
    confidence = result.confidence.detach().cpu().float()
    score = confidence.max(-1).values
    ids = _sample_ids(score, max_points)
    payload = {
        "source": source[ids].tolist(),
        "predicted": predicted[ids].tolist(),
        "bary": result.barycentric_target.detach().cpu().float()[ids].tolist(),
        "confidence": confidence[ids].tolist(),
        "delta": result.delta_vs_static.detach().cpu().float()[ids].tolist(),
        "classes": [f"static {i}" if i == 0 else f"part {i}" for i in range(predicted.shape[1])],
    }
    script = """
function tracesFor(k){
  const src=payload.source, pred=payload.predicted, bary=payload.bary, conf=payload.confidence, delta=payload.delta;
  const xs=[],ys=[],zs=[];
  for(let i=0;i<src.length;i++){ xs.push(pred[i][k][0],bary[i][k][0],null); ys.push(pred[i][k][1],bary[i][k][1],null); zs.push(pred[i][k][2],bary[i][k][2],null); }
  return [
    {type:'scatter3d',mode:'markers',name:'canonical source',x:src.map(p=>p[0]),y:src.map(p=>p[1]),z:src.map(p=>p[2]),marker:{size:2,color:'rgba(80,80,80,0.35)'}},
    {type:'scatter3d',mode:'markers',name:'rigid endpoint hypothesis',x:pred.map(p=>p[k][0]),y:pred.map(p=>p[k][1]),z:pred.map(p=>p[k][2]),marker:{size:3,color:conf.map(c=>c[k]),colorscale:'Viridis',colorbar:{title:'confidence'}}},
    {type:'scatter3d',mode:'markers',name:'soft endpoint barycenter',x:bary.map(p=>p[k][0]),y:bary.map(p=>p[k][1]),z:bary.map(p=>p[k][2]),marker:{size:3,color:delta.map(d=>d[k]),colorscale:'RdBu',colorbar:{title:'E_static-E_k'}}},
    {type:'scatter3d',mode:'lines',name:'transport residual',x:xs,y:ys,z:zs,line:{width:2,color:'rgba(200,60,60,0.45)'}}
  ];
}
const buttons=payload.classes.map((name,k)=>({label:name,method:'update',args:[{x:tracesFor(k).map(t=>t.x),y:tracesFor(k).map(t=>t.y),z:tracesFor(k).map(t=>t.z),'marker.color':[null,tracesFor(k)[1].marker.color,tracesFor(k)[2].marker.color,null]}, {title:'endpoint transport '+name}]}));
Plotly.newPlot('plot', tracesFor(0), {title:'endpoint transport static',scene:{aspectmode:'data'},updatemenus:[{buttons:buttons}]}, {responsive:true,scrollZoom:true});
"""
    _plotly_html(path, payload, script)


def _write_endpoint_energy_html(path, source_xyz, result, *, max_points):
    source = torch.as_tensor(source_xyz).detach().cpu().float()
    static = result.energy.detach().cpu().float()[:, 0]
    moving = result.energy.detach().cpu().float()[:, 1:].amin(-1) if result.energy.shape[1] > 1 else static
    delta = static - moving
    conf = result.confidence.detach().cpu().float().max(-1).values
    ids = _sample_ids(delta.abs() * conf, max_points)
    payload = {
        "xyz": source[ids].tolist(),
        "static_energy": static[ids].tolist(),
        "delta": delta[ids].tolist(),
        "confidence": conf[ids].tolist(),
    }
    script = """
const xyz=payload.xyz;
const mk=(name, color, colorscale)=>({type:'scatter3d',mode:'markers',name:name,x:xyz.map(p=>p[0]),y:xyz.map(p=>p[1]),z:xyz.map(p=>p[2]),marker:{size:3,color:color,colorscale:colorscale,colorbar:{title:name}}});
Plotly.newPlot('plot',[
 mk('endpoint static energy',payload.static_energy,'Magma'),
 mk('E_static - best_moving',payload.delta,'RdBu'),
 mk('endpoint confidence',payload.confidence,'Viridis')
],{title:'endpoint energy diagnostics',scene:{aspectmode:'data'}},{responsive:true,scrollZoom:true});
"""
    _plotly_html(path, payload, script)


def write_endpoint_target_summary(path, transport: RigidEndpointTransport):
    payload = {
        "schema_version": 1,
        "endpoint_frame": int(transport.frame),
        "target_path": str(transport.target_path),
        "target_count": int(transport.target_xyz.shape[0]),
        "has_normals": transport.target_normals is not None,
        "has_sam3": transport.target_sam3 is not None,
        "config": transport.cfg.__dict__,
        "ground_truth_used": False,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
