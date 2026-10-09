"""Moving-only candidate generation and scoring for bidirectional matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from utils.bidirectional_fm import chunked_knn
from utils.rigid_match_refinement import transform_points

FROM_FORWARD_FM = 1
FROM_RIGID = 2
FROM_DIFFUSION = 4
FROM_BACKWARD_FM = 8
FROM_BOTH = FROM_FORWARD_FM | FROM_RIGID


@dataclass
class CandidateSet:
    source_ids: torch.Tensor
    target_ids: torch.Tensor
    provenance: torch.Tensor
    forward_energy: torch.Tensor
    backward_energy: torch.Tensor
    rigid_energy: torch.Tensor
    target_occupancy_energy: torch.Tensor
    displacement_antisymmetry: torch.Tensor
    duplicate_stats: dict[str, Any]
    num_sources: int
    num_targets: int

    @property
    def base_energy(self) -> torch.Tensor:
        return self.forward_energy + self.backward_energy + self.rigid_energy + self.target_occupancy_energy


def _candidate_rows_to_flat(
    *,
    row_source_ids: torch.Tensor,
    target_rows_a: torch.Tensor,
    target_rows_b: torch.Tensor | None,
    target_rows_c: torch.Tensor | None = None,
    reverse_source_ids: torch.Tensor | None = None,
    reverse_target_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    src_out: list[int] = []
    tgt_out: list[int] = []
    prov_out: list[int] = []
    duplicates_before = 0
    duplicates_after = 0
    by_source: dict[int, dict[int, int]] = {}
    for row, src in enumerate(row_source_ids.detach().cpu().tolist()):
        seen: dict[int, int] = {}
        raw: list[tuple[int, int]] = []
        raw += [(int(v), FROM_FORWARD_FM) for v in target_rows_a[row].detach().cpu().tolist()]
        if target_rows_b is not None:
            raw += [(int(v), FROM_RIGID) for v in target_rows_b[row].detach().cpu().tolist()]
        if target_rows_c is not None:
            raw += [(int(v), FROM_DIFFUSION) for v in target_rows_c[row].detach().cpu().tolist()]
        duplicates_before += len(raw) - len({v for v, _ in raw})
        for target_id, provenance in raw:
            seen[target_id] = seen.get(target_id, 0) | provenance
        duplicates_after += len(seen) - len(set(seen))
        by_source[int(src)] = seen
    if reverse_source_ids is not None and reverse_target_ids is not None:
        for src, target_id in zip(reverse_source_ids.detach().cpu().tolist(), reverse_target_ids.detach().cpu().tolist()):
            seen = by_source.setdefault(int(src), {})
            seen[int(target_id)] = seen.get(int(target_id), 0) | FROM_BACKWARD_FM
    for src, seen in by_source.items():
        for target_id, provenance in seen.items():
            src_out.append(int(src))
            tgt_out.append(int(target_id))
            prov_out.append(int(provenance))
    stats = {
        "duplicates_before_dedup": int(duplicates_before),
        "duplicates_after_dedup": int(duplicates_after),
        "rows": int(row_source_ids.numel()),
        "flat_candidates": int(len(src_out)),
    }
    return (
        torch.tensor(src_out, dtype=torch.long),
        torch.tensor(tgt_out, dtype=torch.long),
        torch.tensor(prov_out, dtype=torch.long),
        stats,
    )


def _score_candidates(
    *,
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    forward_endpoint: torch.Tensor,
    backward_endpoint: torch.Tensor,
    forward_displacement: torch.Tensor,
    backward_displacement: torch.Tensor,
    source_scale: torch.Tensor,
    target_scale: torch.Tensor,
    T: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source = source_xyz[source_ids]
    target = target_xyz[target_ids]
    s0 = source_scale[source_ids].clamp_min(1.0e-8)
    sT = target_scale[target_ids].clamp_min(1.0e-8)
    sij = (s0 + sT) * 0.5
    forward = torch.linalg.norm(target - forward_endpoint[source_ids], dim=-1).square() / sT.square().clamp_min(1.0e-12)
    backward = torch.linalg.norm(source - backward_endpoint[target_ids], dim=-1).square() / s0.square().clamp_min(1.0e-12)
    rigid = torch.linalg.norm(target - transform_points(T, source), dim=-1).square() / sij.square().clamp_min(1.0e-12)
    disp = torch.linalg.norm(forward_displacement[source_ids] + backward_displacement[target_ids], dim=-1) / sij.clamp_min(1.0e-8)
    return forward, backward, rigid, disp


def build_motion_candidates(
    *,
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    forward_endpoint: torch.Tensor,
    backward_endpoint: torch.Tensor,
    forward_displacement: torch.Tensor,
    backward_displacement: torch.Tensor,
    source_scale: torch.Tensor,
    target_scale: torch.Tensor,
    active_source_mask: torch.Tensor,
    active_target_mask: torch.Tensor,
    T: torch.Tensor,
    k: int,
    chunk_size: int = 1024,
    include_rigid_branch: bool = True,
    include_reverse_branch: bool = True,
    diffusion_query_xyz: torch.Tensor | None = None,
    diffusion_knn: int = 0,
) -> CandidateSet:
    """Build candidates for moving matching only.

    There is intentionally no KNN(source_xyz, target_xyz) branch here.  That is
    the static/no-motion fallback and belongs outside this moving matcher.
    """
    source = torch.as_tensor(source_xyz).detach().float()
    target = torch.as_tensor(target_xyz, device=source.device, dtype=source.dtype).detach()
    active_sources = torch.nonzero(torch.as_tensor(active_source_mask, device=source.device).bool(), as_tuple=False).reshape(-1)
    active_targets = torch.nonzero(torch.as_tensor(active_target_mask, device=source.device).bool(), as_tuple=False).reshape(-1)
    if int(active_sources.numel()) == 0 or int(active_targets.numel()) == 0:
        empty = torch.empty(0, dtype=torch.long, device=source.device)
        empty_f = torch.empty(0, dtype=source.dtype, device=source.device)
        return CandidateSet(empty, empty, empty, empty_f, empty_f, empty_f, empty_f, empty_f, {"duplicates_before_dedup": 0, "duplicates_after_dedup": 0}, int(source.shape[0]), int(target.shape[0]))
    target_dyn = target[active_targets]
    _, idx_f = chunked_knn(forward_endpoint[active_sources], target_dyn, k, chunk_size=chunk_size)
    rows_f = active_targets[idx_f.to(active_targets.device)]
    rows_r = None
    if include_rigid_branch:
        _, idx_r = chunked_knn(transform_points(T, source[active_sources]), target_dyn, k, chunk_size=chunk_size)
        rows_r = active_targets[idx_r.to(active_targets.device)]
    rows_d = None
    if diffusion_query_xyz is not None and int(diffusion_knn) > 0:
        query = torch.as_tensor(diffusion_query_xyz, device=source.device, dtype=source.dtype)
        if query.shape != source.shape:
            raise ValueError("diffusion_query_xyz must be [N,3]")
        _, idx_d = chunked_knn(query[active_sources], target_dyn, int(diffusion_knn), chunk_size=chunk_size)
        rows_d = active_targets[idx_d.to(active_targets.device)]
    reverse_source_ids = reverse_target_ids = None
    if include_reverse_branch:
        source_dyn = source[active_sources]
        _, idx_b = chunked_knn(backward_endpoint[active_targets], source_dyn, k, chunk_size=chunk_size)
        reverse_source_ids = active_sources[idx_b.reshape(-1).to(active_sources.device)]
        reverse_target_ids = active_targets[:, None].expand_as(idx_b).reshape(-1)
    source_ids, target_ids, provenance, stats = _candidate_rows_to_flat(
        row_source_ids=active_sources,
        target_rows_a=rows_f,
        target_rows_b=rows_r,
        target_rows_c=rows_d,
        reverse_source_ids=reverse_source_ids,
        reverse_target_ids=reverse_target_ids,
    )
    source_ids = source_ids.to(source.device)
    target_ids = target_ids.to(source.device)
    provenance = provenance.to(source.device)
    forward, backward, rigid, disp = _score_candidates(
        source_xyz=source,
        target_xyz=target,
        source_ids=source_ids,
        target_ids=target_ids,
        forward_endpoint=forward_endpoint.to(source),
        backward_endpoint=backward_endpoint.to(source),
        forward_displacement=forward_displacement.to(source),
        backward_displacement=backward_displacement.to(source),
        source_scale=source_scale.to(source),
        target_scale=target_scale.to(source),
        T=T.to(source),
    )
    target_scale_t = torch.as_tensor(target_scale, device=source.device, dtype=source.dtype).clamp_min(1.0e-8)
    counts = torch.bincount(target_ids, minlength=int(target.shape[0])).float()
    nonzero = counts[counts > 0]
    median_hits = nonzero.median().clamp_min(1.0) if nonzero.numel() else source.new_tensor(1.0)
    reuse = torch.log((counts[target_ids].to(source) / median_hits.to(source)).clamp_min(1.0))
    density = torch.log((target_scale_t.median() / target_scale_t[target_ids]).clamp_min(1.0))
    occupancy = reuse + density
    stats["active_sources"] = int(active_sources.numel())
    stats["active_targets"] = int(active_targets.numel())
    stats["candidate_unique_targets"] = int(torch.unique(target_ids).numel()) if target_ids.numel() else 0
    stats["candidates_per_source_mean"] = float(target_ids.numel() / max(1, int(active_sources.numel())))
    stats["target_reuse_median_hits"] = float(median_hits)
    stats["target_occupancy_energy_median"] = float(occupancy.median()) if occupancy.numel() else 0.0
    stats["target_occupancy_energy_p90"] = float(torch.quantile(occupancy.float(), 0.9)) if occupancy.numel() else 0.0
    return CandidateSet(source_ids, target_ids, provenance, forward, backward, rigid, occupancy, disp, stats, int(source.shape[0]), int(target.shape[0]))


def save_candidate_set(candidates: CandidateSet, output_dir: str | Path, *, diffusion_forward: torch.Tensor | None = None, diffusion_backward: torch.Tensor | None = None, diffusion_correction: torch.Tensor | None = None) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(candidates.source_ids.detach().cpu(), output / "source_ids.pt")
    torch.save(candidates.target_ids.detach().cpu(), output / "target_ids.pt")
    torch.save(candidates.provenance.detach().cpu(), output / "provenance.pt")
    torch.save(candidates.forward_energy.detach().cpu(), output / "forward_energy.pt")
    torch.save(candidates.backward_energy.detach().cpu(), output / "backward_energy.pt")
    torch.save(candidates.rigid_energy.detach().cpu(), output / "rigid_energy.pt")
    torch.save(candidates.target_occupancy_energy.detach().cpu(), output / "target_occupancy_energy.pt")
    torch.save(torch.zeros_like(candidates.forward_energy).detach().cpu() if diffusion_forward is None else diffusion_forward.detach().cpu(), output / "forward_diffusion_energy.pt")
    torch.save(torch.zeros_like(candidates.forward_energy).detach().cpu() if diffusion_backward is None else diffusion_backward.detach().cpu(), output / "backward_diffusion_energy.pt")
    correction = (
        torch.zeros((candidates.source_ids.numel(), 3), dtype=candidates.forward_energy.dtype)
        if diffusion_correction is None
        else diffusion_correction.detach().cpu()
    )
    torch.save(correction, output / "diffusion_correction.pt")
