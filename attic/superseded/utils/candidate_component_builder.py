"""Build small uncertain components for progressive rigid discovery.

The graph is only used to choose a compact subset to test next.  It never
smooths labels and it never votes a point into static/dynamic by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from utils.progressive_rigid_state import ProgressiveRigidState, STATUS_UNCERTAIN


@dataclass
class CandidateComponentConfig:
    knn: int = 16
    max_components: int = 2
    max_gaussians: int = 4096
    min_size: int = 3
    distance_scale: float = 0.05
    fm_scale: float = 0.05
    allow_resolved_reconsideration: bool = False


def knn_indices(points: torch.Tensor, k: int, *, chunk_size: int = 2048) -> torch.Tensor:
    points = torch.as_tensor(points).float()
    n = int(points.shape[0])
    if n <= 1:
        return torch.zeros((n, 0), dtype=torch.long, device=points.device)
    kk = min(max(1, int(k)) + 1, n)
    out = []
    for start in range(0, n, int(chunk_size)):
        d = torch.cdist(points[start:start + int(chunk_size)], points)
        out.append(torch.topk(d, k=kk, largest=False, sorted=True).indices[:, 1:])
    return torch.cat(out, dim=0)


def _bfs_components(
    active: torch.Tensor,
    neigh: torch.Tensor,
    allowed: torch.Tensor,
    *,
    min_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    device = active.device
    n = int(active.numel())
    comp_id = torch.full((n,), -1, device=device, dtype=torch.long)
    components: list[torch.Tensor] = []
    active_ids = torch.nonzero(active, as_tuple=False).reshape(-1).tolist()
    active_set = set(active_ids)
    current = 0
    for seed in active_ids:
        if int(comp_id[seed]) >= 0:
            continue
        queue = [seed]
        comp = []
        comp_id[seed] = current
        while queue:
            node = queue.pop()
            comp.append(node)
            for slot, nb in enumerate(neigh[node].tolist()):
                if not bool(allowed[node, slot]):
                    continue
                if nb not in active_set or int(comp_id[nb]) >= 0:
                    continue
                comp_id[nb] = current
                queue.append(int(nb))
        if len(comp) >= int(min_size):
            components.append(torch.tensor(comp, device=device, dtype=torch.long))
            current += 1
        else:
            comp_id[torch.tensor(comp, device=device, dtype=torch.long)] = -1
    return comp_id, components


def build_candidate_components(
    state: ProgressiveRigidState,
    cfg: CandidateComponentConfig,
) -> dict[str, Any]:
    resolved = (
        torch.zeros(state.num_foreground, device=state.device, dtype=torch.bool)
        if state.resolved_once is None
        else state.resolved_once.to(device=state.device).bool()
    )
    if bool(cfg.allow_resolved_reconsideration):
        resolved[:] = False
    uncertain = (state.status == STATUS_UNCERTAIN) & state.observable & ~resolved
    if not bool(uncertain.any()):
        state.candidate_component_id[:] = -1
        state.component_scores = []
        return {"component_id": state.candidate_component_id, "components": [], "scores": [], "leaderboard": [], "history": []}

    neigh = knn_indices(state.canonical_fg, int(cfg.knn))
    active_ids = torch.nonzero(uncertain, as_tuple=False).reshape(-1)
    active_mask = torch.zeros(state.num_foreground, device=state.device, dtype=torch.bool)
    active_mask[active_ids] = True

    # Edges are allowed only when the current geometry/FM evidence is local
    # enough to describe a component.  This is grouping, not label smoothing.
    distance_scale = max(float(cfg.distance_scale), 1.0e-8)
    fm_scale = max(float(cfg.fm_scale), 1.0e-8)
    neighbour_xyz = state.canonical_fg[neigh]
    neighbour_fm = state.fm_displacement[neigh]
    local = torch.linalg.norm(neighbour_xyz - state.canonical_fg[:, None], dim=-1) <= distance_scale
    fm_local = torch.linalg.norm(neighbour_fm - state.fm_displacement[:, None], dim=-1) <= fm_scale
    allowed = (local | fm_local) & active_mask[neigh]

    comp_id, components = _bfs_components(uncertain, neigh, allowed, min_size=int(cfg.min_size))
    current_uncertainty = 1.0 - state.responsibilities.max(dim=-1).values.clamp(0.0, 1.0)
    entropy = -(state.responsibilities * torch.log(state.responsibilities.clamp_min(1.0e-12))).sum(-1)
    scored = []
    history = []
    for cid, ids in enumerate(components):
        overlap = int(resolved[ids].sum())
        if overlap:
            raise RuntimeError(f"candidate component {cid} contains {overlap} previously resolved points")
        uncertainty = current_uncertainty[ids]
        render = state.render_residual_per_gaussian[ids].detach()
        visibility = state.visibility_mass[ids].detach()
        fm_dyn = state.fm_probability[ids].detach()
        residual_mass = (visibility.clamp_min(0.0) * render.clamp_min(0.0)).sum()
        mean_uncertainty = uncertainty.mean()
        mean_fm_dyn = fm_dyn.mean()
        mean_visibility = visibility.mean()
        history.append({
            "id": int(cid),
            "num_points": int(ids.numel()),
            "overlap_resolved_points": overlap,
            "overlap_fraction": float(overlap / max(1, int(ids.numel()))),
        })
        scored.append({
            "id": int(cid),
            "count": int(ids.numel()),
            "residual_mass": float(residual_mass.detach().cpu()),
            "mean_residual": float(render.mean().detach().cpu()) if render.numel() else 0.0,
            "uncertainty": float(mean_uncertainty.detach().cpu()),
            "entropy": float(entropy[ids].mean().detach().cpu()),
            "fm_dynamic_probability": float(mean_fm_dyn.detach().cpu()),
            "visibility": float(mean_visibility.detach().cpu()) if visibility.numel() else 0.0,
            "status_ok": overlap == 0,
            "indices": ids.detach().cpu(),
        })
    scored.sort(
        key=lambda item: (
            -float(item["residual_mass"]),
            -float(item["uncertainty"]),
            -float(item["fm_dynamic_probability"]),
            -int(item["count"]),
        )
    )
    picked = []
    total = 0
    selected_ids = []
    for item in scored:
        if len(picked) >= int(cfg.max_components):
            break
        ids = item["indices"]
        if total + int(ids.numel()) > int(cfg.max_gaussians):
            ids = ids[: max(0, int(cfg.max_gaussians) - total)]
        if int(ids.numel()) < int(cfg.min_size):
            continue
        total += int(ids.numel())
        selected_ids.append(ids.to(state.device))
        item = dict(item)
        item["indices"] = ids
        picked.append(item)
        if total >= int(cfg.max_gaussians):
            break

    state.candidate_component_id[:] = -1
    for new_id, ids in enumerate(selected_ids):
        state.candidate_component_id[ids.to(state.device)] = int(new_id)
    state.component_scores = [
        {k: v for k, v in item.items() if k != "indices"}
        for item in picked
    ]
    return {
        "component_id": state.candidate_component_id,
        "components": selected_ids,
        "scores": picked,
        "leaderboard": [{k: v for k, v in item.items() if k != "indices"} for item in scored[:10]],
        "history": history,
    }
