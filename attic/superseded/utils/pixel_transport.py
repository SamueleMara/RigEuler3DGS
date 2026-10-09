"""R3 -- soft partial transport between endpoint images, on dynamic mass only.

    Gamma_c(p, q) = P(p_0 -> q_T | c)

Solved only on the dynamic/uncertain pixel mass, because the static pixels
already have their answer and including them would let the transport spend
its mass explaining geometry that never moved.

The cost is a calibrated negative log-likelihood, not a weighted sum (§11).
Displacement and appearance each arrive divided by a scale MEASURED on static
pixels, so both are already in units of their own noise and adding their log
densities is the correct combination rather than a choice. There is no
lambda to set because there is nothing left to trade off.

Dustbin mass on both sides handles occlusion and disocclusion (§30): a moved
surface that is hidden at the other endpoint has no counterpart, and forcing
one on it is how a transport invents motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

EPS = 1.0e-12


@dataclass
class PixelTransport:
    """Sparse endpoint pixel coupling for one camera."""

    camera_id: int
    source_pixel_indices: torch.Tensor   # [S] flat pixel index in frame 0
    target_pixel_indices: torch.Tensor   # [S, K] candidate targets per source
    coupling_values: torch.Tensor        # [S, K] Gamma
    source_dynamic_mass: torch.Tensor    # [S]
    target_dynamic_mass: torch.Tensor    # [S, K]
    unmatched_source_mass: torch.Tensor  # [S]
    unmatched_target_mass: torch.Tensor  # scalar tensor
    stats: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def solve_partial_pixel_transport(
    source_marginals,
    target_marginals,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    calibration,
    *,
    camera_id: int = 0,
    max_source_pixels: int = 2048,
    max_target_pixels: int = 2048,
    candidates: int = 16,
    sinkhorn_iterations: int = 60,
    generator: torch.Generator | None = None,
) -> PixelTransport:
    """Calibrated partial OT on the dynamic pixel domain of one camera."""
    from utils.dynamic_pixel_marginals import build_dynamic_pixel_domain

    source_index, source_mass = build_dynamic_pixel_domain(
        source_marginals, max_pixels=int(max_source_pixels), generator=generator
    )
    target_index, target_mass = build_dynamic_pixel_domain(
        target_marginals, max_pixels=int(max_target_pixels), generator=generator
    )
    height, width = int(source_marginals.total_mass.shape[0]), int(source_marginals.total_mass.shape[1])
    device = source_marginals.total_mass.device

    if int(source_index.numel()) == 0 or int(target_index.numel()) == 0:
        empty = torch.zeros((0, 1), device=device)
        return PixelTransport(
            camera_id=int(camera_id),
            source_pixel_indices=source_index, target_pixel_indices=empty.long(),
            coupling_values=empty, source_dynamic_mass=source_mass,
            target_dynamic_mass=empty, unmatched_source_mass=source_mass,
            unmatched_target_mass=torch.zeros((), device=device),
            stats={"reason": "no dynamic pixel mass in this view"},
        )

    source_xy = torch.stack((source_index // width, source_index % width), dim=1).float()
    target_xy = torch.stack((target_index // width, target_index % width), dim=1).float()
    source_rgb = source_image.reshape(3, -1)[:, source_index].T
    target_rgb = target_image.reshape(3, -1)[:, target_index].T

    # Candidate targets by image distance -- an ordering device only, so the
    # transport does not have to consider every pixel pair.
    k = min(int(candidates), int(target_index.numel()))
    distance = torch.cdist(source_xy, target_xy)
    near = torch.topk(distance, k=k, dim=1, largest=False)
    candidate_index = near.indices
    candidate_distance = near.values

    appearance = torch.linalg.norm(
        target_rgb[candidate_index] - source_rgb[:, None, :], dim=-1
    )
    # Calibrated log-likelihood: each term already in units of its own
    # measured noise, so the sum of logs is the joint density (§11).
    log_likelihood = (
        calibration.displacement_loglik(candidate_distance)
        + calibration.appearance_loglik(appearance)
    )
    cost = -log_likelihood

    # Partial transport: rows may keep mass unmatched, columns are capped by
    # their own dynamic mass so a popular target cannot absorb everything.
    unmatched_cost = float(torch.quantile(cost.flatten().float(), 0.90))
    logits = torch.cat((-cost, torch.full((int(cost.shape[0]), 1), -unmatched_cost, device=device)), dim=1)
    plan = torch.softmax(logits, dim=1)
    coupling = plan[:, :-1] * source_mass[:, None]
    unmatched_source = plan[:, -1] * source_mass

    column_capacity = target_mass[candidate_index]
    for _ in range(max(1, int(sinkhorn_iterations) // 12)):
        claimed = torch.zeros(int(target_index.numel()), device=device)
        claimed.scatter_add_(0, candidate_index.reshape(-1), coupling.reshape(-1))
        overload = (claimed / target_mass.clamp_min(EPS)).clamp_min(1.0)
        scale = 1.0 / overload[candidate_index]
        reduced = coupling * scale
        unmatched_source = unmatched_source + (coupling - reduced).sum(dim=1)
        coupling = reduced

    moved = torch.zeros(int(target_index.numel()), device=device)
    moved.scatter_add_(0, candidate_index.reshape(-1), coupling.reshape(-1))
    return PixelTransport(
        camera_id=int(camera_id),
        source_pixel_indices=source_index,
        target_pixel_indices=target_index[candidate_index],
        coupling_values=coupling,
        source_dynamic_mass=source_mass,
        target_dynamic_mass=column_capacity,
        unmatched_source_mass=unmatched_source,
        unmatched_target_mass=(target_mass - moved).clamp_min(0.0).sum(),
        stats={
            "source_pixels": int(source_index.numel()),
            "target_pixels": int(target_index.numel()),
            "transported_mass": float(coupling.sum()),
            "unmatched_source_fraction": float(unmatched_source.sum() / source_mass.sum().clamp_min(EPS)),
            "unmatched_target_fraction": float((target_mass - moved).clamp_min(0.0).sum() / target_mass.sum().clamp_min(EPS)),
            "median_displacement_px": float((coupling * candidate_distance).sum(1).div(
                coupling.sum(1).clamp_min(EPS)).median()),
        },
    )


@torch.no_grad()
def summarise_pixel_transport(transports: list[PixelTransport]) -> dict[str, Any]:
    """Aggregate over cameras, reporting unmatched mass honestly."""
    if not transports:
        return {}
    return {
        "cameras": len(transports),
        "total_transported_mass": float(sum(t.stats.get("transported_mass", 0.0) for t in transports)),
        "mean_unmatched_source_fraction": float(
            sum(t.stats.get("unmatched_source_fraction", 0.0) for t in transports) / len(transports)
        ),
        "mean_unmatched_target_fraction": float(
            sum(t.stats.get("unmatched_target_fraction", 0.0) for t in transports) / len(transports)
        ),
        "median_displacement_px": float(
            sorted(t.stats.get("median_displacement_px", 0.0) for t in transports)[len(transports) // 2]
        ),
    }
