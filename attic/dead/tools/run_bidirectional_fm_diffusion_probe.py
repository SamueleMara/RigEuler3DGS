#!/usr/bin/env python3
"""Standalone bidirectional FM + diffusion correspondence probe.

This intentionally does not call the old endpoint-transport matcher or the
progressive rigid loop.  It ends after a high-precision reciprocal match set
and one robust SE(3) fit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from plyfile import PlyData

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.bidirectional_fm import (  # noqa: E402
    FlowTrainConfig,
    local_spacing,
    save_flow_result,
    target_median_spacing,
    train_flow_direction,
)
from utils.bidirectional_match_candidates import (  # noqa: E402
    CandidateSet,
    build_motion_candidates,
    save_candidate_set,
)
from utils.diffusion_displacement_matcher import (  # noqa: E402
    DisplacementDiffusionConfig,
    save_diffusion_model,
    score_displacement_candidates,
    train_displacement_denoiser,
)
from utils.fm_cycle_consistency import cycle_consistency, save_cycle_result  # noqa: E402
from utils.matching_debug import binary_metrics, tensor_summary, write_json, write_matching_html  # noqa: E402
from utils.reciprocal_matching import ReciprocalMatches, save_reciprocal_matches, strict_reciprocal_matches  # noqa: E402
from utils.rigid_match_refinement import (  # noqa: E402
    RigidFitConfig,
    robust_rigid_fit,
    se3_delta_norm,
    transform_points,
    transform_summary,
)


def _resolve_ply(root: str | Path, iteration: int) -> Path:
    path = Path(root).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".ply":
        return path
    if int(iteration) > 0:
        ply = path / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
        if ply.is_file():
            return ply
    found: list[tuple[int, Path]] = []
    for child in (path / "point_cloud").glob("iteration_*"):
        try:
            it = int(child.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        ply = child / "point_cloud.ply"
        if ply.is_file():
            found.append((it, ply))
    if not found:
        raise FileNotFoundError(f"no point_cloud.ply found under {path}")
    return max(found, key=lambda item: item[0])[1]


def _load_cloud(path: Path) -> dict[str, torch.Tensor | list[str]]:
    vertex = PlyData.read(str(path))["vertex"].data
    names = list(vertex.dtype.names or [])
    xyz = torch.from_numpy(np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1).astype("float32"))
    out: dict[str, torch.Tensor | list[str]] = {"xyz": xyz, "fields": names}
    if "semantic_prob" in names:
        out["semantic_probability"] = torch.from_numpy(np.asarray(vertex["semantic_prob"]).astype("float32"))
    elif "semantic_logit" in names:
        logit = torch.from_numpy(np.asarray(vertex["semantic_logit"]).astype("float32"))
        out["semantic_probability"] = torch.sigmoid(logit)
    elif "semantic_logits" in names:
        logit = torch.from_numpy(np.asarray(vertex["semantic_logits"]).astype("float32"))
        out["semantic_probability"] = torch.sigmoid(logit)
    if "gt_dynamic" in names:
        out["gt_dynamic"] = torch.from_numpy(np.asarray(vertex["gt_dynamic"]).astype("float32")) > 0.5
    elif "gt_part" in names:
        out["gt_dynamic"] = torch.from_numpy(np.asarray(vertex["gt_part"]).astype("int64")) > 0
    return out


def _foreground_mask(cloud: dict[str, torch.Tensor | list[str]], threshold: float) -> torch.Tensor:
    xyz = torch.as_tensor(cloud["xyz"])
    probability = cloud.get("semantic_probability")
    if torch.is_tensor(probability) and probability.numel() == xyz.shape[0]:
        mask = probability.float() >= float(threshold)
        if bool(mask.any()):
            return mask
    return torch.ones(xyz.shape[0], dtype=torch.bool)


def _load_gt_axis(root_scene: str | Path) -> torch.Tensor | None:
    root = Path(root_scene).expanduser().resolve()
    for rel in ("pointcloud/joint_metadata.json", "metadata/colmap_export.json", "colmap/export_summary.json"):
        path = root / rel
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        axis = data.get("axis_vector")
        if axis is None and isinstance(data.get("joint"), dict):
            axis = data["joint"].get("axis_vector")
        if axis is not None and len(axis) == 3:
            value = torch.tensor(axis, dtype=torch.float32)
            if float(value.norm()) > 1.0e-8:
                return value / value.norm()
    return None


def _dynamic_mask(result, cycle_conf: torch.Tensor, args) -> torch.Tensor:
    return (
        result.valid.bool()
        & (result.probability.float() >= float(args.dynamic_probability_threshold))
        & (result.confidence.float() >= float(args.dynamic_confidence_threshold))
        & (cycle_conf.float() >= float(args.cycle_confidence_threshold))
    )


def _fit_coarse(
    x0: torch.Tensor,
    yT: torch.Tensor,
    forward,
    backward,
    d0: torch.Tensor,
    dT: torch.Tensor,
    wF: torch.Tensor,
    wB: torch.Tensor,
    *,
    scale: float,
    fit_cfg: RigidFitConfig,
) -> dict[str, Any]:
    pair_source = torch.cat((x0[d0], backward.endpoint[dT]), dim=0)
    pair_target = torch.cat((forward.endpoint[d0], yT[dT]), dim=0)
    pair_weight = torch.cat((wF[d0], wB[dT]), dim=0)
    return robust_rigid_fit(pair_source, pair_target, pair_weight, scale=scale, cfg=fit_cfg)


def _forward_best(candidates: CandidateSet, energy: torch.Tensor, source_weight: torch.Tensor, target_weight: torch.Tensor) -> ReciprocalMatches:
    # For ablations A/B only; production matching below is strict reciprocal.
    best: dict[int, tuple[float, int]] = {}
    s = candidates.source_ids.detach().cpu().tolist()
    e = energy.detach().cpu().tolist()
    for idx, (src, val) in enumerate(zip(s, e)):
        old = best.get(int(src))
        if old is None or float(val) < old[0]:
            best[int(src)] = (float(val), idx)
    ids = torch.tensor(sorted(idx for _, idx in best.values()), dtype=torch.long)
    if ids.numel() == 0:
        return ReciprocalMatches(ids, ids, ids, torch.empty(0), {"forward_best_matches": 0, "reciprocal_matches": 0})
    src = candidates.source_ids.detach().cpu()[ids]
    tgt = candidates.target_ids.detach().cpu()[ids]
    conf = torch.exp(-0.5 * energy.detach().cpu()[ids].clamp_min(0.0)).clamp(0.0, 1.0)
    weights = torch.sqrt(source_weight.detach().cpu()[src].clamp_min(0.0) * target_weight.detach().cpu()[tgt].clamp_min(0.0)) * conf
    unique, counts = torch.unique(tgt, return_counts=True)
    stats = {
        "forward_best_matches": int(ids.numel()),
        "reciprocal_matches": 0,
        "unique_target_ratio": float(unique.numel() / max(1, int(ids.numel()))),
        "maximum_target_multiplicity": int(counts.max()) if counts.numel() else 0,
        "p90_target_multiplicity": float(torch.quantile(counts.float(), 0.9)) if counts.numel() > 1 else 0.0,
    }
    return ReciprocalMatches(src, tgt, ids, weights, stats)


def _fit_matches(
    name: str,
    matches: ReciprocalMatches,
    x0: torch.Tensor,
    yT: torch.Tensor,
    *,
    scale: float,
    fit_cfg: RigidFitConfig,
    init_T: torch.Tensor | None = None,
) -> dict[str, Any]:
    if int(matches.source_ids.numel()) < 3:
        return {"name": name, "ok": False, "reason": "fewer_than_three_matches", "matching": matches.stats}
    try:
        fit = robust_rigid_fit(
            x0[matches.source_ids.long()],
            yT[matches.target_ids.long()],
            matches.weights.float().clamp_min(1.0e-8),
            scale=scale,
            cfg=fit_cfg,
            init_T=init_T,
        )
    except Exception as exc:
        return {"name": name, "ok": False, "reason": str(exc), "matching": matches.stats}
    return {
        "name": name,
        "ok": True,
        "matching": matches.stats,
        "rigid": {
            **fit["summary"],
            "weighted_rms": fit["weighted_rms"],
            "robust_inliers": fit["robust_inliers"],
        },
    }


def _reciprocal(
    candidates: CandidateSet,
    energy: torch.Tensor,
    source_weight: torch.Tensor,
    target_weight: torch.Tensor,
) -> ReciprocalMatches:
    return strict_reciprocal_matches(
        source_ids=candidates.source_ids,
        target_ids=candidates.target_ids,
        energy=energy,
        source_weight=source_weight,
        target_weight=target_weight,
        num_sources=candidates.num_sources,
        num_targets=candidates.num_targets,
    )


def _score_diffusion_for_candidates(
    candidates: CandidateSet,
    x0: torch.Tensor,
    yT: torch.Tensor,
    forward,
    backward,
    cycle: dict[str, torch.Tensor],
    source_weight: torch.Tensor,
    target_weight: torch.Tensor,
    cfg: DisplacementDiffusionConfig,
    args,
    training_matches: ReciprocalMatches | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    sid = candidates.source_ids.long().cpu()
    tid = candidates.target_ids.long().cpu()
    train_sid, train_tid = sid, tid
    if training_matches is not None and int(training_matches.source_ids.numel()) > 0:
        train_sid = training_matches.source_ids.long().cpu()
        train_tid = training_matches.target_ids.long().cpu()
        base_weight = training_matches.weights.detach().cpu().float().clamp_min(1.0e-6)
    else:
        base_weight = torch.sqrt(source_weight.cpu()[train_sid].clamp_min(0.0) * target_weight.cpu()[train_tid].clamp_min(0.0))
        base_weight = base_weight / base_weight.max().clamp_min(1.0e-6)
    train_clean_f = yT[train_tid] - x0[train_sid]
    train_clean_b = x0[train_sid] - yT[train_tid]
    model_f, log_f = train_displacement_denoiser(
        clean_displacement=train_clean_f,
        xyz=x0[train_sid],
        local_scale=forward.local_scale[train_sid],
        p_dynamic=forward.probability[train_sid],
        fm_confidence=forward.confidence[train_sid],
        cycle_confidence=cycle["forward_cycle_confidence"][train_sid],
        fm_displacement=forward.displacement[train_sid],
        sample_weight=base_weight,
        cfg=cfg,
        direction_name="forward",
        device=args.device,
    )
    model_b, log_b = train_displacement_denoiser(
        clean_displacement=train_clean_b,
        xyz=yT[train_tid],
        local_scale=backward.local_scale[train_tid],
        p_dynamic=backward.probability[train_tid],
        fm_confidence=backward.confidence[train_tid],
        cycle_confidence=cycle["backward_cycle_confidence"][train_tid],
        fm_displacement=backward.displacement[train_tid],
        sample_weight=base_weight,
        cfg=cfg,
        direction_name="backward",
        device=args.device,
    )
    clean_f = yT[tid] - x0[sid]
    clean_b = x0[sid] - yT[tid]
    df, corr_f = score_displacement_candidates(
        model_f,
        clean_displacement=clean_f,
        xyz=x0[sid],
        local_scale=forward.local_scale[sid],
        p_dynamic=forward.probability[sid],
        fm_confidence=forward.confidence[sid],
        cycle_confidence=cycle["forward_cycle_confidence"][sid],
        fm_displacement=forward.displacement[sid],
        cfg=cfg,
        device=args.device,
    )
    db, corr_b = score_displacement_candidates(
        model_b,
        clean_displacement=clean_b,
        xyz=yT[tid],
        local_scale=backward.local_scale[tid],
        p_dynamic=backward.probability[tid],
        fm_confidence=backward.confidence[tid],
        cycle_confidence=cycle["backward_cycle_confidence"][tid],
        fm_displacement=backward.displacement[tid],
        cfg=cfg,
        device=args.device,
    )
    correction = 0.5 * (corr_f - corr_b)
    return df, db, correction, {"model": model_f, "log": log_f}, {"model": model_b, "log": log_b}


def _rescore_diffusion_candidates(
    candidates: CandidateSet,
    x0: torch.Tensor,
    yT: torch.Tensor,
    forward,
    backward,
    cycle: dict[str, torch.Tensor],
    cfg: DisplacementDiffusionConfig,
    args,
    f_model,
    b_model,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sid = candidates.source_ids.long().cpu()
    tid = candidates.target_ids.long().cpu()
    clean_f = yT[tid] - x0[sid]
    clean_b = x0[sid] - yT[tid]
    df, corr_f = score_displacement_candidates(
        f_model,
        clean_displacement=clean_f,
        xyz=x0[sid],
        local_scale=forward.local_scale[sid],
        p_dynamic=forward.probability[sid],
        fm_confidence=forward.confidence[sid],
        cycle_confidence=cycle["forward_cycle_confidence"][sid],
        fm_displacement=forward.displacement[sid],
        cfg=cfg,
        device=args.device,
    )
    db, corr_b = score_displacement_candidates(
        b_model,
        clean_displacement=clean_b,
        xyz=yT[tid],
        local_scale=backward.local_scale[tid],
        p_dynamic=backward.probability[tid],
        fm_confidence=backward.confidence[tid],
        cycle_confidence=cycle["backward_cycle_confidence"][tid],
        fm_displacement=backward.displacement[tid],
        cfg=cfg,
        device=args.device,
    )
    return df, db, 0.5 * (corr_f - corr_b)


def _best_diffusion_query(candidates: CandidateSet, diffusion_energy: torch.Tensor, correction: torch.Tensor, yT: torch.Tensor, num_sources: int) -> torch.Tensor:
    query = torch.zeros((num_sources, 3), dtype=yT.dtype)
    best: dict[int, tuple[float, int]] = {}
    for idx, (src, val) in enumerate(zip(candidates.source_ids.cpu().tolist(), diffusion_energy.cpu().tolist())):
        old = best.get(int(src))
        if old is None or float(val) < old[0]:
            best[int(src)] = (float(val), idx)
    for src, (_, idx) in best.items():
        target_id = int(candidates.target_ids[idx])
        query[src] = yT[target_id] + correction[idx].to(yT)
    missing = torch.linalg.norm(query, dim=-1) == 0
    if bool(missing.any()):
        # Missing sources are inactive; value is ignored by the active-source mask.
        query[missing] = yT.mean(0)
    return query


def _save_ablation(path: Path, row: dict[str, Any]) -> None:
    write_json(path, row)


def _cycle_pair_energy(candidates: CandidateSet, cycle: dict[str, torch.Tensor]) -> torch.Tensor:
    sid = candidates.source_ids.long().cpu()
    tid = candidates.target_ids.long().cpu()
    fc = cycle["forward_cycle_confidence"][sid].float().clamp_min(1.0e-8)
    bc = cycle["backward_cycle_confidence"][tid].float().clamp_min(1.0e-8)
    return -(torch.log(fc) + torch.log(bc))


def _print_report(
    *,
    iteration: int,
    x0: torch.Tensor,
    yT: torch.Tensor,
    forward,
    backward,
    cycle: dict[str, torch.Tensor],
    d0: torch.Tensor,
    dT: torch.Tensor,
    candidates: CandidateSet,
    df: torch.Tensor,
    db: torch.Tensor,
    correction: torch.Tensor,
    matches: ReciprocalMatches,
    fit: dict[str, Any],
    delta_norm: float,
) -> None:
    source_energy = df + db
    print(
        "\n========================================================\n"
        f"[BIDIRECTIONAL FM MATCHING iter={iteration:02d}]\n\n"
        "FORWARD FM\n"
        f"    total source                 = {int(x0.shape[0])}\n"
        f"    valid                        = {int(forward.valid.sum())}\n"
        f"    confident dynamic            = {int(d0.sum())}\n"
        f"    p_dynamic mean / median      = {float(forward.probability.mean()):.6g} / {float(forward.probability.median()):.6g}\n"
        f"    FM confidence mean / median  = {float(forward.confidence.mean()):.6g} / {float(forward.confidence.median()):.6g}\n"
        f"    cycle error median / p90     = {tensor_summary(cycle['forward_cycle_error'])['median']:.6g} / {tensor_summary(cycle['forward_cycle_error'])['p90']:.6g}\n"
        f"    cycle conf mean / median     = {float(cycle['forward_cycle_confidence'].mean()):.6g} / {float(cycle['forward_cycle_confidence'].median()):.6g}\n\n"
        "BACKWARD FM\n"
        f"    total target                 = {int(yT.shape[0])}\n"
        f"    valid                        = {int(backward.valid.sum())}\n"
        f"    confident dynamic            = {int(dT.sum())}\n"
        f"    p_dynamic mean / median      = {float(backward.probability.mean()):.6g} / {float(backward.probability.median()):.6g}\n"
        f"    FM confidence mean / median  = {float(backward.confidence.mean()):.6g} / {float(backward.confidence.median()):.6g}\n"
        f"    cycle error median / p90     = {tensor_summary(cycle['backward_cycle_error'])['median']:.6g} / {tensor_summary(cycle['backward_cycle_error'])['p90']:.6g}\n"
        f"    cycle conf mean / median     = {float(cycle['backward_cycle_confidence'].mean()):.6g} / {float(cycle['backward_cycle_confidence'].median()):.6g}\n\n"
        "CANDIDATES\n"
        f"    active sources               = {candidates.duplicate_stats.get('active_sources', 0)}\n"
        f"    candidates/source            = {candidates.duplicate_stats.get('candidates_per_source_mean', 0.0):.6g}\n"
        f"    duplicates before dedup      = {candidates.duplicate_stats.get('duplicates_before_dedup', 0)}\n"
        f"    duplicates after dedup       = {candidates.duplicate_stats.get('duplicates_after_dedup', 0)}\n"
        f"    candidate unique targets     = {candidates.duplicate_stats.get('candidate_unique_targets', 0)}\n\n"
        "DIFFUSION\n"
        f"    F energy median / p90        = {tensor_summary(df)['median']:.6g} / {tensor_summary(df)['p90']:.6g}\n"
        f"    B energy median / p90        = {tensor_summary(db)['median']:.6g} / {tensor_summary(db)['p90']:.6g}\n"
        f"    correction median / p90      = {tensor_summary(torch.linalg.norm(correction, dim=-1))['median']:.6g} / {tensor_summary(torch.linalg.norm(correction, dim=-1))['p90']:.6g}\n\n"
        "RECIPROCAL\n"
        f"    forward-best matches         = {matches.stats.get('forward_best_matches', 0)}\n"
        f"    reciprocal matches           = {matches.stats.get('reciprocal_matches', 0)}\n"
        f"    reciprocal fraction          = {matches.stats.get('reciprocal_fraction', 0.0):.6g}\n"
        f"    unique target ratio          = {matches.stats.get('unique_target_ratio', 0.0):.6g}\n"
        f"    maximum target multiplicity  = {matches.stats.get('maximum_target_multiplicity', 0)}\n\n"
        "SE3\n"
        f"    matches used                 = {int(matches.source_ids.numel())}\n"
        f"    robust inliers               = {fit.get('robust_inliers', 0)}\n"
        f"    residual RMS                 = {fit.get('weighted_rms', float('inf')):.6g}\n"
        f"    rotation angle               = {fit['summary']['rotation_angle_deg']:.6g}\n"
        f"    rotation axis                = {fit['summary']['rotation_axis']}\n"
        f"    translation                  = {fit['summary']['translation']}\n"
        f"    delta from previous T        = {delta_norm:.6g}\n"
        f"    diffusion pair energy med    = {tensor_summary(source_energy)['median']:.6g}\n"
        "========================================================"
    )


def run(args) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve() / "bidirectional_matching"
    output.mkdir(parents=True, exist_ok=True)
    source_ply = _resolve_ply(args.static_model_dir, int(args.static_iteration))
    target_ply = _resolve_ply(args.endpoint_static_model_dir, int(args.endpoint_static_iteration))
    source_cloud = _load_cloud(source_ply)
    target_cloud = _load_cloud(target_ply)
    source_mask = _foreground_mask(source_cloud, float(args.semantic_threshold))
    target_mask = _foreground_mask(target_cloud, float(args.semantic_threshold))
    x0 = torch.as_tensor(source_cloud["xyz"]).float()[source_mask]
    yT = torch.as_tensor(target_cloud["xyz"]).float()[target_mask]
    source_gt = source_cloud.get("gt_dynamic")
    target_gt = target_cloud.get("gt_dynamic")
    if torch.is_tensor(source_gt):
        source_gt = source_gt[source_mask]
    if torch.is_tensor(target_gt):
        target_gt = target_gt[target_mask]
    if int(x0.shape[0]) < 3 or int(yT.shape[0]) < 3:
        raise RuntimeError("source and endpoint foreground clouds need at least three points")

    write_json(output / "input_summary.json", {
        "source_ply": str(source_ply),
        "target_ply": str(target_ply),
        "source_points": int(x0.shape[0]),
        "target_points": int(yT.shape[0]),
        "source_foreground_from_semantic": bool(source_mask.sum() != source_mask.numel()),
        "target_foreground_from_semantic": bool(target_mask.sum() != target_mask.numel()),
        "ground_truth_used_as_algorithm_input": False,
    })

    flow_cfg = FlowTrainConfig(
        knn=int(args.fm_knn),
        temperature=float(args.fm_temperature),
        robust_delta=float(args.fm_robust_delta),
        hidden_dim=int(args.fm_hidden_dim),
        layers=int(args.fm_layers),
        lr=float(args.fm_lr),
        iters=int(args.fm_iters),
        batch_size=int(args.fm_batch_size),
        identity_weight=float(args.fm_identity_weight),
        smooth_weight=float(args.fm_smooth_weight),
        integration_steps=int(args.fm_integration_steps),
        integrator=str(args.fm_integrator),
        prior_prob_bias=float(args.fm_prior_prob_bias),
        prior_prob_temperature=float(args.fm_prior_prob_temperature),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed),
    )
    backward_cfg = FlowTrainConfig(**{**asdict(flow_cfg), "seed": int(args.seed) + 1})
    forward = train_flow_direction(x0, yT, flow_cfg, direction_name="forward", device=args.device)
    backward = train_flow_direction(yT, x0, backward_cfg, direction_name="backward", device=args.device)
    cycle = cycle_consistency(
        source_xyz=x0,
        target_xyz=yT,
        forward_endpoint=forward.endpoint,
        backward_endpoint=backward.endpoint,
        forward_model=forward.model,
        backward_model=backward.model,
        forward_cfg=flow_cfg,
        backward_cfg=backward_cfg,
        source_scale=forward.local_scale,
        target_scale=backward.local_scale,
        device=args.device,
    )
    save_cycle_result(cycle, output / "cycle")
    save_flow_result(forward, output / "forward_fm", flow_cfg)
    save_flow_result(backward, output / "backward_fm", backward_cfg)

    d0 = _dynamic_mask(forward, cycle["forward_cycle_confidence"], args)
    dT = _dynamic_mask(backward, cycle["backward_cycle_confidence"], args)
    if int(d0.sum()) < 3 or int(dT.sum()) < 3:
        raise RuntimeError(f"not enough confident dynamic points: D0={int(d0.sum())}, DT={int(dT.sum())}")
    source_weight = (forward.probability * forward.confidence).clamp(0.0, 1.0)
    target_weight = (backward.probability * backward.confidence).clamp(0.0, 1.0)
    source_weight_no_cycle = (forward.probability * forward.confidence).clamp(0.0, 1.0)
    target_weight_no_cycle = (backward.probability * backward.confidence).clamp(0.0, 1.0)

    scale = max(target_median_spacing(yT, chunk_size=int(args.chunk_size)), float(forward.local_scale.median()))
    fit_cfg = RigidFitConfig(irls_iters=int(args.rigid_irls_iters), huber_delta=float(args.rigid_huber_delta))
    coarse = _fit_coarse(x0, yT, forward, backward, d0, dT, source_weight, target_weight, scale=scale, fit_cfg=fit_cfg)
    T = coarse["T"].detach().cpu()
    (output / "rigid").mkdir(exist_ok=True)
    torch.save(T, output / "rigid" / "T_initial.pt")
    write_json(output / "rigid" / "coarse_stats.json", {
        "scale": scale,
        "weighted_rms": coarse["weighted_rms"],
        "robust_inliers": coarse["robust_inliers"],
        "transform": coarse["summary"],
    })

    candidates = build_motion_candidates(
        source_xyz=x0,
        target_xyz=yT,
        forward_endpoint=forward.endpoint,
        backward_endpoint=backward.endpoint,
        forward_displacement=forward.displacement,
        backward_displacement=backward.displacement,
        source_scale=forward.local_scale,
        target_scale=backward.local_scale,
        active_source_mask=d0,
        active_target_mask=dT,
        T=T,
        k=int(args.candidate_knn),
        chunk_size=int(args.chunk_size),
        include_rigid_branch=True,
    )
    if candidates.duplicate_stats.get("duplicates_after_dedup", 1) != 0:
        raise RuntimeError("candidate deduplication failed")

    df = torch.zeros(candidates.source_ids.numel())
    db = torch.zeros(candidates.source_ids.numel())
    correction = torch.zeros((candidates.source_ids.numel(), 3))
    diffusion_info: dict[str, Any] = {"enabled": not bool(args.no_diffusion)}
    f_diff = b_diff = None
    diff_cfg = None
    if not bool(args.no_diffusion):
        diff_cfg = DisplacementDiffusionConfig(
            hidden_dim=int(args.diffusion_hidden_dim),
            layers=int(args.diffusion_layers),
            iters=int(args.diffusion_iters),
            batch_size=int(args.diffusion_batch_size),
            lr=float(args.diffusion_lr),
            trials=int(args.diffusion_trials),
            schedule=str(args.diffusion_schedule),
            seed=int(args.seed) + 10,
            chunk_size=int(args.diffusion_chunk_size),
        )
        seed = _reciprocal(candidates, candidates.base_energy, source_weight, target_weight)
        if int(seed.source_ids.numel()) < 3:
            raise RuntimeError("diffusion training requires at least three clean reciprocal seed pairs")
        df, db, correction, f_diff, b_diff = _score_diffusion_for_candidates(
            candidates, x0, yT, forward, backward, cycle, source_weight, target_weight, diff_cfg, args, training_matches=seed
        )
        save_diffusion_model(f_diff["model"], f_diff["log"], output / "diffusion" / "forward", diff_cfg)
        save_diffusion_model(b_diff["model"], b_diff["log"], output / "diffusion" / "backward", diff_cfg)
        diffusion_info.update({"config": asdict(diff_cfg), "forward_log": f_diff["log"], "backward_log": b_diff["log"]})

    ablations = output / "ablations"
    ablations.mkdir(exist_ok=True)
    cand_a = build_motion_candidates(
        source_xyz=x0, target_xyz=yT, forward_endpoint=forward.endpoint, backward_endpoint=backward.endpoint,
        forward_displacement=forward.displacement, backward_displacement=backward.displacement,
        source_scale=forward.local_scale, target_scale=backward.local_scale,
        active_source_mask=d0, active_target_mask=torch.ones_like(dT), T=torch.eye(4),
        k=int(args.candidate_knn), chunk_size=int(args.chunk_size), include_rigid_branch=False,
    )
    mA = _forward_best(cand_a, cand_a.forward_energy, source_weight_no_cycle, torch.ones_like(target_weight_no_cycle))
    _save_ablation(ablations / "A_forward_fm_endpoint_knn_only.json", _fit_matches("A", mA, x0, yT, scale=scale, fit_cfg=fit_cfg))
    mB = _forward_best(candidates, candidates.forward_energy + candidates.backward_energy + candidates.target_occupancy_energy, source_weight_no_cycle, target_weight_no_cycle)
    _save_ablation(ablations / "B_bidirectional_fm.json", _fit_matches("B", mB, x0, yT, scale=scale, fit_cfg=fit_cfg, init_T=T))
    mC = _reciprocal(candidates, candidates.forward_energy + candidates.backward_energy + candidates.target_occupancy_energy, source_weight_no_cycle, target_weight_no_cycle)
    _save_ablation(ablations / "C_bidirectional_reciprocal.json", _fit_matches("C", mC, x0, yT, scale=scale, fit_cfg=fit_cfg, init_T=T))
    mD = _reciprocal(candidates, candidates.base_energy + _cycle_pair_energy(candidates, cycle), source_weight, target_weight)
    _save_ablation(ablations / "D_cycle_confidence.json", _fit_matches("D", mD, x0, yT, scale=scale, fit_cfg=fit_cfg, init_T=T))
    mE = _reciprocal(candidates, candidates.forward_energy + candidates.backward_energy + candidates.target_occupancy_energy + df + db, source_weight, target_weight)
    _save_ablation(ablations / "E_diffusion_compatibility.json", _fit_matches("E", mE, x0, yT, scale=scale, fit_cfg=fit_cfg, init_T=T))
    mF = _reciprocal(candidates, candidates.base_energy + df + db, source_weight, target_weight)
    _save_ablation(ablations / "F_coarse_se3_compatibility.json", _fit_matches("F", mF, x0, yT, scale=scale, fit_cfg=fit_cfg, init_T=T))
    _save_ablation(ablations / "H_residual_diffusion_disabled.json", {"name": "H", "ok": False, "reason": "residual diffusion intentionally disabled until base matcher passes"})

    history = []
    prev_pairs: set[tuple[int, int]] | None = None
    final_matches = mF
    final_fit = robust_rigid_fit(x0[mF.source_ids.long()], yT[mF.target_ids.long()], mF.weights.float().clamp_min(1.0e-8), scale=scale, cfg=fit_cfg, init_T=T)
    T_history = [T.clone()]
    for iteration in range(max(1, int(args.match_iters))):
        diffusion_query = None
        if not bool(args.no_diffusion) and int(args.diffusion_correction_knn) > 0:
            diffusion_query = _best_diffusion_query(candidates, df + db, correction, yT, int(x0.shape[0]))
        candidates = build_motion_candidates(
            source_xyz=x0,
            target_xyz=yT,
            forward_endpoint=forward.endpoint,
            backward_endpoint=backward.endpoint,
            forward_displacement=forward.displacement,
            backward_displacement=backward.displacement,
            source_scale=forward.local_scale,
            target_scale=backward.local_scale,
            active_source_mask=d0,
            active_target_mask=dT,
            T=T,
            k=int(args.candidate_knn),
            chunk_size=int(args.chunk_size),
            include_rigid_branch=True,
            diffusion_query_xyz=diffusion_query,
            diffusion_knn=int(args.diffusion_correction_knn),
        )
        if not bool(args.no_diffusion):
            assert diff_cfg is not None and f_diff is not None and b_diff is not None
            df, db, correction = _rescore_diffusion_candidates(
                candidates, x0, yT, forward, backward, cycle, diff_cfg, args, f_diff["model"], b_diff["model"]
            )
        energy = candidates.base_energy + df + db
        matches = _reciprocal(candidates, energy, source_weight, target_weight)
        if int(matches.source_ids.numel()) < 3:
            raise RuntimeError("strict reciprocal matching produced fewer than three pairs")
        fit = robust_rigid_fit(
            x0[matches.source_ids.long()],
            yT[matches.target_ids.long()],
            matches.weights.float().clamp_min(1.0e-8),
            scale=scale,
            cfg=fit_cfg,
            init_T=T,
        )
        new_T = fit["T"].detach().cpu()
        delta = se3_delta_norm(T, new_T)
        pairs = set(zip(matches.source_ids.cpu().tolist(), matches.target_ids.cpu().tolist()))
        stable = prev_pairs == pairs
        row = {
            "iteration": int(iteration),
            "delta_norm": delta,
            "pairs_stable": bool(stable),
            "matching": matches.stats,
            "rigid": {**fit["summary"], "weighted_rms": fit["weighted_rms"], "robust_inliers": fit["robust_inliers"]},
            "candidate_stats": candidates.duplicate_stats,
        }
        history.append(row)
        _print_report(
            iteration=iteration,
            x0=x0,
            yT=yT,
            forward=forward,
            backward=backward,
            cycle=cycle,
            d0=d0,
            dT=dT,
            candidates=candidates,
            df=df,
            db=db,
            correction=correction,
            matches=matches,
            fit=fit,
            delta_norm=delta,
        )
        T = new_T
        T_history.append(T.clone())
        final_matches = matches
        final_fit = fit
        prev_pairs = pairs
        if stable and delta < float(args.transform_stop_epsilon):
            break

    save_candidate_set(candidates, output / "candidates", diffusion_forward=df, diffusion_backward=db, diffusion_correction=correction)
    save_reciprocal_matches(final_matches, output / "reciprocal")
    write_json(output / "reciprocal" / "matching_stats.json", final_matches.stats)
    rigid_dir = output / "rigid"
    rigid_dir.mkdir(exist_ok=True)
    torch.save(torch.stack(T_history), rigid_dir / "T_history.pt")
    torch.save(T, rigid_dir / "T_final.pt")
    rigid_stats = {
        "scale": scale,
        "history": history,
        "final": {**final_fit["summary"], "weighted_rms": final_fit["weighted_rms"], "robust_inliers": final_fit["robust_inliers"]},
    }
    write_json(rigid_dir / "rigid_stats.json", rigid_stats)
    write_json(output / "diffusion" / "summary.json", diffusion_info)
    _save_ablation(ablations / "G_iterative_rematch_refit.json", {"name": "G", "ok": True, "history": history, "final": rigid_stats["final"], "matching": final_matches.stats})

    final_energy = candidates.base_energy + df + db
    transformed = transform_points(T, x0)
    if bool(args.save_debug_html):
        write_matching_html(
            output / "debug" / "bidirectional_matching.html",
            source_xyz=x0,
            target_xyz=yT,
            transformed_source_xyz=transformed,
            forward_endpoint=forward.endpoint,
            backward_endpoint=backward.endpoint,
            active_source_mask=d0,
            active_target_mask=dT,
            match_source_ids=final_matches.source_ids,
            match_target_ids=final_matches.target_ids,
            match_energy=final_energy[final_matches.candidate_indices] if final_matches.candidate_indices.numel() else torch.empty(0),
            max_points=int(args.max_html_points),
            seed=int(args.seed),
        )

    gt_debug: dict[str, Any] = {"available": False, "ground_truth_used_as_algorithm_input": False}
    if torch.is_tensor(source_gt) or torch.is_tensor(target_gt):
        gt_debug = {"available": True, "ground_truth_used_as_algorithm_input": False}
        print("\n[GT DEBUG ONLY — NOT USED BY MATCHER]")
        if torch.is_tensor(source_gt):
            f_metrics = binary_metrics(forward.probability, source_gt, threshold=0.5)
            gt_debug["forward_dynamic"] = f_metrics
            print(f"GT DEBUG ONLY — NOT USED BY MATCHER forward dynamic P/R/F1 = {f_metrics['precision']:.4f}/{f_metrics['recall']:.4f}/{f_metrics['f1']:.4f}")
            gt_debug["reciprocal_source_gt_dynamic_fraction"] = float(source_gt[final_matches.source_ids.long()].float().mean()) if final_matches.source_ids.numel() else 0.0
            print(f"GT DEBUG ONLY — NOT USED BY MATCHER reciprocal source GT-dynamic fraction = {gt_debug['reciprocal_source_gt_dynamic_fraction']:.4f}")
        if torch.is_tensor(target_gt):
            b_metrics = binary_metrics(backward.probability, target_gt, threshold=0.5)
            gt_debug["backward_dynamic"] = b_metrics
            print(f"GT DEBUG ONLY — NOT USED BY MATCHER backward dynamic P/R/F1 = {b_metrics['precision']:.4f}/{b_metrics['recall']:.4f}/{b_metrics['f1']:.4f}")
            gt_debug["reciprocal_target_gt_dynamic_fraction"] = float(target_gt[final_matches.target_ids.long()].float().mean()) if final_matches.target_ids.numel() else 0.0
            print(f"GT DEBUG ONLY — NOT USED BY MATCHER reciprocal target GT-dynamic fraction = {gt_debug['reciprocal_target_gt_dynamic_fraction']:.4f}")
        axis = _load_gt_axis(args.root_scene or args.scene or ".")
        if axis is not None:
            est = torch.tensor(rigid_stats["final"]["rotation_axis"])
            if float(est.norm()) > 1.0e-8:
                angle = math.degrees(math.acos(float(torch.abs((est / est.norm()) @ axis).clamp(-1.0, 1.0))))
                gt_debug["axis_error_degrees_sign_invariant"] = angle
                print(f"GT DEBUG ONLY — NOT USED BY MATCHER axis error = {angle:.4f} deg")
    write_json(output / "debug" / "gt_debug.json", gt_debug)

    summary = {
        "schema_version": 1,
        "pipeline": "bidirectional_fm_diffusion_matching",
        "source_points": int(x0.shape[0]),
        "target_points": int(yT.shape[0]),
        "D0_confident_dynamic": int(d0.sum()),
        "DT_confident_dynamic": int(dT.sum()),
        "final_matches": int(final_matches.source_ids.numel()),
        "final_unique_target_ratio": final_matches.stats.get("unique_target_ratio", 0.0),
        "final_max_target_multiplicity": final_matches.stats.get("maximum_target_multiplicity", 0),
        "final_transform": rigid_stats["final"],
        "candidate_stats": candidates.duplicate_stats,
        "gt_debug": gt_debug,
        "ground_truth_used_as_algorithm_input": False,
    }
    write_json(output / "summary.json", summary)
    print(f"\n[complete] {output}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="", help="Alias for --root-scene")
    p.add_argument("--root-scene", default="", help="Dataset root; only used for GT debug metadata lookup")
    p.add_argument("--static-model-dir", required=True)
    p.add_argument("--endpoint-static-model-dir", required=True)
    p.add_argument("--static-iteration", type=int, default=30000)
    p.add_argument("--endpoint-static-iteration", type=int, default=30000)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--semantic-threshold", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--chunk-size", type=int, default=1024)

    p.add_argument("--fm-iters", type=int, default=1000)
    p.add_argument("--fm-knn", type=int, default=16)
    p.add_argument("--fm-temperature", type=float, default=0.02)
    p.add_argument("--fm-robust-delta", type=float, default=0.02)
    p.add_argument("--fm-hidden-dim", type=int, default=128)
    p.add_argument("--fm-layers", type=int, default=4)
    p.add_argument("--fm-lr", type=float, default=1.0e-3)
    p.add_argument("--fm-batch-size", type=int, default=2048)
    p.add_argument("--fm-identity-weight", type=float, default=0.01)
    p.add_argument("--fm-smooth-weight", type=float, default=0.0)
    p.add_argument("--fm-integration-steps", type=int, default=16)
    p.add_argument("--fm-integrator", choices=["euler", "rk4"], default="euler")
    p.add_argument("--fm-prior-prob-bias", type=float, default=1.0)
    p.add_argument("--fm-prior-prob-temperature", type=float, default=0.35)

    p.add_argument("--dynamic-probability-threshold", type=float, default=0.65)
    p.add_argument("--dynamic-confidence-threshold", type=float, default=0.25)
    p.add_argument("--cycle-confidence-threshold", type=float, default=0.0)
    p.add_argument("--candidate-knn", type=int, default=16)
    p.add_argument("--match-iters", type=int, default=4)
    p.add_argument("--transform-stop-epsilon", type=float, default=1.0e-4)
    p.add_argument("--rigid-irls-iters", type=int, default=8)
    p.add_argument("--rigid-huber-delta", type=float, default=0.02)

    p.add_argument("--no-diffusion", action="store_true")
    p.add_argument("--diffusion-iters", type=int, default=500)
    p.add_argument("--diffusion-hidden-dim", type=int, default=128)
    p.add_argument("--diffusion-layers", type=int, default=4)
    p.add_argument("--diffusion-batch-size", type=int, default=4096)
    p.add_argument("--diffusion-lr", type=float, default=1.0e-3)
    p.add_argument("--diffusion-trials", type=int, default=4)
    p.add_argument("--diffusion-schedule", choices=["cosine"], default="cosine")
    p.add_argument("--diffusion-chunk-size", type=int, default=65536)
    p.add_argument("--diffusion-correction-knn", type=int, default=4)

    p.add_argument("--save-debug-html", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-html-points", type=int, default=3000)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.root_scene:
        args.root_scene = args.scene
    torch.manual_seed(int(args.seed))
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
