"""Initialise K finite rigid hypotheses from immutable endpoint FM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from utils.progressive_assignment import (
    AssignmentThresholds,
    compute_fm_rigid_residuals,
    initialize_from_fm_probability,
    update_responsibilities_from_residuals,
)
from utils.progressive_rigid_state import ProgressiveRigidState, PROVENANCE_RIGID_INIT, status_for_part
from utils.rigid_endpoint_fit import (
    build_fm_target_candidates,
    build_source_neighbor_graph,
    fm_guided_correspondence_em,
    ransac_candidate_se3,
    se3_log,
    target_median_nn_scale,
    transform_summary,
    weighted_kabsch,
)


@dataclass
class RigidHypothesisInitConfig:
    num_moving_parts: int = 1
    fm_core_dynamic_threshold: float = 0.65
    fm_core_static_threshold: float = 0.25
    fm_core_confidence_threshold: float = 0.25
    rigid_init_ransac_iters: int = 256
    rigid_init_inlier_threshold: float = 0.04
    ransac_threshold_mode: str = "derived"
    candidate_target_count: int = 32
    candidate_endpoint_knn: int = 16
    candidate_chunk_size: int = 1024
    correspondence_em_iters: int = 8
    rigid_irls_iters: int = 5
    rigid_huber_delta: float = 0.02
    assignment_temperature: float = 0.05
    responsibility_gamma: float = 2.0
    seed: int = 20260912
    correspondence_isometry_k: int = 8
    correspondence_isometry_weight: float = 1.0


def initialize_rigid_hypotheses(
    state: ProgressiveRigidState,
    cfg: RigidHypothesisInitConfig,
    *,
    endpoint_target_xyz: torch.Tensor,
) -> tuple[ProgressiveRigidState, dict[str, Any]]:
    thresholds = AssignmentThresholds(
        static_probability=float(cfg.fm_core_static_threshold),
        dynamic_probability=float(cfg.fm_core_dynamic_threshold),
        confidence=float(cfg.fm_core_confidence_threshold),
        moving_residual=float(cfg.rigid_init_inlier_threshold),
        static_residual=float(cfg.rigid_init_inlier_threshold),
        temperature=float(cfg.assignment_temperature),
    )
    initialize_from_fm_probability(state, thresholds)
    target = endpoint_target_xyz.to(device=state.device, dtype=state.canonical_fg.dtype)
    spcd = target_median_nn_scale(target, chunk_size=int(cfg.candidate_chunk_size))
    derived_threshold = 3.0 * spcd
    threshold = derived_threshold if str(cfg.ransac_threshold_mode) == "derived" else float(cfg.rigid_init_inlier_threshold)
    print(
        "[RIGID_SCALE]\n"
        f"target median NN distance = {spcd:.6g}\n"
        f"RANSAC threshold          = {threshold:.6g}\n"
        f"old configured threshold  = {float(cfg.rigid_init_inlier_threshold):.6g}\n"
        f"ratio old/new             = {float(cfg.rigid_init_inlier_threshold) / max(threshold, 1.0e-12):.6g}"
    )
    if state.candidate_target_xyz is None:
        candidates = build_fm_target_candidates(
            fm_endpoint_xyz=state.fm_endpoint_xyz,
            target_xyz=target,
            legacy_indices=state.fm_candidate_matches,
            legacy_costs=state.fm_candidate_costs,
            endpoint_knn=int(cfg.candidate_endpoint_knn),
            max_candidates=int(cfg.candidate_target_count),
            spcd=spcd,
            chunk_size=int(cfg.candidate_chunk_size),
            source_xyz=state.canonical_fg,
        )
        state.candidate_target_xyz = candidates["candidate_target_xyz"].to(state.canonical_fg)
        state.candidate_target_idx = candidates["candidate_target_idx"].to(state.device)
        state.candidate_fm_prior = candidates["candidate_fm_prior"].to(state.canonical_fg)
        state.candidate_valid = candidates["candidate_valid"].to(state.device)
        state.candidate_source_id = candidates["candidate_source_id"].to(state.device)
    diagnostics: list[dict[str, Any]] = []
    seed_mask = state.fm_valid & (state.fm_probability >= float(cfg.fm_core_dynamic_threshold))
    trusted_static = state.trusted_static_mask()
    if bool((seed_mask & trusted_static).any()):
        raise RuntimeError("trusted-static points entered moving rigid seed mask")
    if int(seed_mask.sum()) < 3:
        raise RuntimeError("Rigid initialization has fewer than three high-confidence dynamic seeds")
    if state.initial_fm_dynamic_core is not None:
        state.initial_fm_dynamic_core.zero_()
    if state.outer_accepted_moving is not None:
        state.outer_accepted_moving.zero_()
    source = state.canonical_fg
    if (
        state.neighbor_idx is None
        and int(cfg.correspondence_isometry_k) > 0
        and float(cfg.correspondence_isometry_weight) > 0.0
    ):
        graph = build_source_neighbor_graph(source, int(cfg.correspondence_isometry_k))
        state.neighbor_idx = graph["neighbor_idx"].to(state.device)
        state.neighbor_dist0 = graph["neighbor_dist0"].to(device=state.device, dtype=state.canonical_fg.dtype)
    neighbor_idx = state.neighbor_idx
    neighbor_dist0 = state.neighbor_dist0
    fm_seed = None
    try:
        fm_seed = weighted_kabsch(
            source[seed_mask],
            state.fm_endpoint_xyz[seed_mask],
            state.fm_dynamic_confidence[seed_mask] if state.fm_dynamic_confidence is not None else state.fm_confidence_current[seed_mask],
        )["T"].to(state.canonical_fg)
    except Exception:
        fm_seed = torch.eye(4, device=state.device, dtype=state.canonical_fg.dtype)
    for part in range(int(cfg.num_moving_parts)):
        ransac = ransac_candidate_se3(
            source,
            state.candidate_target_xyz,
            state.candidate_target_idx,
            state.candidate_fm_prior,
            state.candidate_valid,
            state.fm_endpoint_xyz,
            seed_mask,
            state.fm_dynamic_confidence if state.fm_dynamic_confidence is not None else state.fm_confidence_current,
            threshold=threshold,
            spcd=spcd,
            iters=int(cfg.rigid_init_ransac_iters),
            seed=int(cfg.seed) + part,
        )
        em = fm_guided_correspondence_em(
            source,
            state.candidate_target_xyz,
            state.candidate_target_idx,
            state.candidate_fm_prior,
            state.candidate_valid,
            state.fm_endpoint_xyz,
            seed_mask,
            state.fm_dynamic_confidence if state.fm_dynamic_confidence is not None else state.fm_confidence_current,
            init_T=ransac["T"],
            spcd=spcd,
            irls_iters=int(cfg.rigid_irls_iters),
            huber_delta=float(cfg.rigid_huber_delta),
            em_iters=int(cfg.correspondence_em_iters),
            neighbor_idx=neighbor_idx,
            neighbor_dist0=neighbor_dist0,
            isometry_weight=float(cfg.correspondence_isometry_weight),
        )
        T = em["T"].to(device=state.device, dtype=state.canonical_fg.dtype)
        residual = em["residual_per_point"].to(device=state.device, dtype=state.canonical_fg.dtype)
        weight = em["support_weight"].to(state.canonical_fg)
        state.part_transforms_T[part] = T
        state.part_twists_xi[part] = se3_log(T)
        state.part_fit_residual[part] = residual[weight > 0].mean() if bool((weight > 0).any()) else residual.mean()
        state.part_support_weight[part] = weight
        if state.initial_fm_dynamic_core is not None:
            state.initial_fm_dynamic_core[part] = seed_mask
        if state.correspondence_confidence is not None:
            state.correspondence_confidence[part] = em["corr_confidence"].to(state.correspondence_confidence)
        if state.correspondence_selected is not None:
            state.correspondence_selected[part] = em["selected"].to(state.correspondence_selected)
        if state.correspondence_target_xyz is not None:
            state.correspondence_target_xyz[part] = em["selected_target_xyz"].to(state.correspondence_target_xyz)
        state.status[seed_mask] = status_for_part(part + 1)
        state.hard_labels[seed_mask] = part + 1
        state.provenance[seed_mask] = PROVENANCE_RIGID_INIT
        diagnostics.append({
            "part": int(part + 1),
            "ransac": ransac["summary"],
            "correspondence_em": em["history"],
            "collision": em["collision"],
            "seed_points": int(seed_mask.sum()),
            "support_weight_sum": float(weight.sum().detach().cpu()),
            "weighted_rms": float(state.part_fit_residual[part].detach().cpu()),
            "twist": [float(v) for v in state.part_twists_xi[part].detach().cpu()],
            "T_refined_matrix": T.detach().cpu().tolist(),
            "T_refined": transform_summary(T),
        })

    compute_fm_rigid_residuals(state)
    update_responsibilities_from_residuals(
        state,
        temperature=float(cfg.assignment_temperature),
        move_prior_bias=0.0,
    )
    state.enforce_committed_labels()
    summary = {
        "num_moving_parts": int(cfg.num_moving_parts),
        "label_counts": state.label_counts(),
        "target_scale": {
            "target_median_nn_distance": spcd,
            "ransac_threshold": threshold,
            "old_configured_threshold": float(cfg.rigid_init_inlier_threshold),
            "ratio_old_new": float(cfg.rigid_init_inlier_threshold) / max(threshold, 1.0e-12),
            "mode": str(cfg.ransac_threshold_mode),
        },
        "T_seed_from_fm": transform_summary(fm_seed),
        "T_seed_from_fm_matrix": fm_seed.detach().cpu().tolist(),
        "parts": diagnostics,
        "ground_truth_used_as_algorithm_input": False,
    }
    return state.validate(), summary
