from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from utils.temporal_visibility_pretraining import _atomic_torch_save
from utils.visualize_transported_gaussians_html import write_transported_gaussians_html

from .embeddings import PartEmbeddingModel
from .endpoint import load_ply_xyz
from .inner_acceptance import endpoint_usefulness
from utils.rigid_transform_utils import apply_rotation_to_gaussian_quaternions
from .handoff_diagnostics import polarization_row
from .shared_killing_solver import solve_shared_killing_sequence
from .killing_inner_solver import (
    NestedKillingMotionState,
    save_multiview_debug_images,
    solve_nested_killing_sequence,
)
from .outer_render import (
    candidate_state_at_frame,
    compose_straight_through_scene,
    compute_detached_3d_gradient_cap,
    kinetic_label_loss,
    motion_aware_label_graph_loss,
    static_kinetic_energy,
    outer_render_membership_loss,
    straight_through_membership,
)
from .outer_embedding_solver import (
    build_outer_targets,
    candidate_nn_energies,
    candidates_from_local_endpoints,
    geometry_guard_value,
    outer_embedding_losses,
)
from .endpoint_contrastive_init import run_endpoint_contrastive_initialization
from .neighborhood import build_neighbor_graph
from .static_anchor_init import extract_static_anchors_from_ply
from .pixel_matching import (
    accumulate_pair_affinity,
    build_target_cache,
    current_pixel_attraction_loss,
    normalized_match_confidence,
    render_contributors,
    target_coverage_loss,
    candidate_pixel_evidence,
    covariance_from_scale_rotation,
    pixel_candidate_loss,
    pixel_margin_loss,
)
from .render_contrastive import (
    build_render_contrast_graph,
    build_render_response_fingerprints,
    contrast_gt_diagnostics,
    refine_soft_labels_with_render_contrast,
)
from .render_graph_outer import render_class_preference, run_render_graph_outer
from .motion_boundary_split import (
    apply_motion_boundary_split,
    find_motion_boundary_split_candidates,
    invalidate_per_gaussian_caches_after_split,
    rebuild_state_after_boundary_split,
    save_boundary_split_artifacts,
)
from .soft_labels import (
    TemporalOrderGuard,
    render_evidence_against_gt,
    soft_oscillation,
    soft_outer_update,
)
from .static_init import diffuse_static_anchors, initial_part_logits
from .temporal_credit import (
    TemporalCreditCycle,
    combine_global_score,
    combine_local_score,
    establish_normalization,
    evaluate_match_score,
    evaluate_render_score,
    unsupported_penalty_from,
)


def _jsonable(value):
    if torch.is_tensor(value):
        v = value.detach().cpu()
        return _jsonable(float(v)) if v.numel() == 1 else _jsonable(v.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Some diagnostics are legitimately undefined -- the soft-oscillation
        # reversal fraction needs three states and reports NaN until it has them
        # (soft-labels §42). Emit JSON null rather than relaxing allow_nan=False,
        # which is the guard that catches genuinely corrupt numbers.
        return None
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_final_ply(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_file():
        return p
    candidates: list[tuple[int, Path]] = []
    for ply in p.glob("point_cloud/iteration_*/point_cloud.ply"):
        try:
            candidates.append((int(ply.parent.name.rsplit("_", 1)[-1]), ply))
        except ValueError:
            pass
    if not candidates:
        raise RuntimeError(f"--final_3dgs_path does not resolve to a point_cloud.ply: {path}")
    return max(candidates, key=lambda item: item[0])[1]


def _camera_frame_id(camera, fallback: int) -> int:
    value = getattr(camera, "dynamic_frame_id", None)
    if value is None:
        value = getattr(camera, "frame_id", None)
    if value is None:
        return int(fallback)
    return int(value)


def _has_real_timestamps(args) -> bool:
    """True only when the sequence carries measured acquisition timestamps.

    Without a timestamp table the dataset reader synthesises ``frame_id / fps``
    from ``--fps`` (default 30). For pose sequences sampled along an articulation
    trajectory that is a fabricated physical timescale, not a measurement.
    """
    source = str(getattr(args, "source_path", "") or "")
    if str(getattr(args, "dynamic_timestamps_file", "") or "").strip():
        return True
    root = Path(source)
    return (root / "timestamps.json").is_file() or (root / "metadata" / "timestamps.json").is_file()


def _camera_groups(scene, args=None) -> tuple[list[int], torch.Tensor, list[list[object]]]:
    by_frame: dict[int, list[object]] = {}
    by_time: dict[int, float] = {}
    for idx, camera in enumerate(scene.getTrainCameras()):
        fid = _camera_frame_id(camera, idx)
        by_frame.setdefault(fid, []).append(camera)
        by_time[fid] = float(getattr(camera, "timestamp", fid))
    frame_ids = sorted(by_frame)
    if not frame_ids:
        raise RuntimeError("nested Killing discovery needs training cameras")
    times = torch.tensor([by_time[fid] for fid in frame_ids], dtype=torch.float32)

    mode = str(getattr(args, "nested_time_unit", "auto") or "auto").lower()
    use_frame_time = mode == "frame"
    if mode == "auto":
        # Only trust a physical timescale when it was actually measured.
        use_frame_time = args is not None and not _has_real_timestamps(args)
    if use_frame_time:
        if times.numel() > 1:
            dt = float((times[1:] - times[:-1]).median())
            print(
                f"[Nested Killing] no measured timestamps; using frame-index time "
                f"(dt=1 per interval) instead of the synthesised dt={dt:.5f}s. "
                f"dt scales Exp(xi dt), so a fabricated dt rescales every twist and "
                f"attenuates the render gradient by the same factor."
            )
        times = torch.arange(len(frame_ids), dtype=torch.float32)
    elif times.numel() > 1 and float((times[1:] - times[:-1]).abs().max()) <= 1.0e-12:
        times = torch.arange(len(frame_ids), dtype=torch.float32)
    return frame_ids, times, [by_frame[fid] for fid in frame_ids]


def _temperature(start: float, end: float, outer: int, total: int) -> float:
    """Legacy iteration-count anneal. FIRM v3 §23 forbids using this to drive
    tau to 0.15 before the motion is correct; kept only for reference/tests."""
    if int(total) <= 1:
        return float(end)
    alpha = float(outer) / float(int(total) - 1)
    return float(start) + alpha * (float(end) - float(start))


def adaptive_temperature(
    current: float,
    *,
    motion_accepted: bool,
    label_churn_fraction: float,
    candidate_separation: float,
    geometry_stable: bool,
    transitions: int | None = None,
    previous_transitions: int | None = None,
    soft_floor: float = 0.7,
    hard_floor: float = 0.4,
    decay: float = 0.85,
    churn_tolerance: float = 0.01,
    separation_tolerance: float = 0.0,
) -> tuple[float, list[str]]:
    """FIRM v3 §23: tau stays soft until the motion is actually correct.

    tau may only decrease when the inner motion was accepted, the hard labels
    have stopped moving, the candidate endpoint energies are separated, and the
    outer geometry is stable.  While any of those fails, tau >= soft_floor.
    It never drops below ``hard_floor``.
    """
    blockers: list[str] = []
    if not bool(motion_accepted):
        blockers.append("inner motion not accepted")
    if float(label_churn_fraction) > float(churn_tolerance):
        blockers.append(f"labels still moving ({float(label_churn_fraction) * 100:.2f}%)")
    # §26: tau may only fall once render-driven label transitions DECREASE.
    # Net churn can sit under tolerance while the transition count still rises,
    # which is how tau reached 0.44 with transitions climbing 117 -> 213.
    if transitions is not None and previous_transitions is not None:
        if int(transitions) > int(previous_transitions):
            blockers.append(f"label transitions still rising ({previous_transitions} -> {transitions})")
    elif transitions is not None and int(transitions) > 0:
        blockers.append(f"first outer with {int(transitions)} transitions; no trend yet")
    if float(candidate_separation) <= float(separation_tolerance):
        blockers.append("candidate endpoint energies weakly separated")
    if not bool(geometry_stable):
        blockers.append("outer geometry unstable")

    tau = float(current)
    if blockers:
        return max(tau, float(soft_floor)), blockers
    return max(float(hard_floor), tau * float(decay)), blockers


def _weighted_sum(losses: dict[str, torch.Tensor], weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    weighted = {key: float(weights.get(key, 0.0)) * value for key, value in losses.items()}
    total = sum(weighted.values()) if weighted else torch.tensor(0.0)
    denom = sum(abs(float(v.detach().cpu())) for v in weighted.values()) + 1.0e-12
    fractions = {f"frac_{key}": abs(float(value.detach().cpu())) / denom for key, value in weighted.items()}
    return total, weighted, fractions


def _track_payload(
    xyz0: torch.Tensor,
    rot0: torch.Tensor,
    motion: NestedKillingMotionState,
    prob: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    positions = motion.positions.detach().cpu()
    times = motion.times.detach().cpu()
    if positions.shape[0] <= 1:
        velocities = torch.zeros_like(positions)
    else:
        dt = torch.diff(times).reshape(-1, 1, 1).clamp_min(1.0e-8)
        interval_velocity = torch.diff(positions, dim=0) / dt
        velocities = torch.cat((interval_velocity, interval_velocity[-1:]), dim=0)
    return {
        "schema_version": 1,
        "source_of_truth": "nested_part_embedding_differential_killing_field",
        "times": times,
        "positions": positions,
        "velocities": velocities,
        "rotations": motion.rotations.detach().cpu(),
        "canonical_xyz": xyz0.detach().cpu(),
        "canonical_rotation": rot0.detach().cpu(),
        "part_probabilities": prob.detach().cpu(),
        "motion_membership": prob[:, 1:].sum(dim=-1).detach().cpu(),
        "part_labels": labels.detach().cpu(),
        "part_rotation": motion.part_rotation.detach().cpu(),
        "part_translation": motion.part_translation.detach().cpu(),
        # Same [F, K+1, ...] layout as part_rotation/part_translation so the part
        # index means the same thing in all three (0 = static identity).
        "part_rotation_quaternions": motion.part_quaternion.detach().cpu(),
        "omega": motion.omega.detach().cpu(),
        "b": motion.b.detach().cpu(),
        "center_velocity": motion.center_velocity.detach().cpu(),
        "motion_separation": motion.motion_separation.detach().cpu(),
        "se3_integration_convention": "spatial/world-frame Exp(SE3), T_next = Exp(dt xi_hat) @ T",
        "kabsch_used": False,
        "per_gaussian_twist_optimized": False,
    }


def run_nested_part_embedding_killing_discovery(
    scene,
    gaussians,
    pipe,
    background: torch.Tensor,
    dataset,
    args,
    output: Path,
    *,
    log_fn=None,
) -> dict:
    """Spec path: embeddings outside, differential Killing fields inside."""
    if str(getattr(args, "dynamic_prior_path", "") or "").strip():
        raise RuntimeError("--enable_nested_killing_part_discovery forbids --dynamic_prior_path")
    if bool(getattr(args, "kpart_rigid_discovery", False)):
        raise RuntimeError("--enable_nested_killing_part_discovery forbids legacy --kpart_rigid_discovery")
    result_dir = output / "nested_part_embedding_killing"
    result_dir.mkdir(parents=True, exist_ok=True)
    final_ply = _resolve_final_ply(str(getattr(args, "final_3dgs_path", "")))

    device = gaussians.get_canonical_xyz.device
    xyz0 = gaussians.get_canonical_xyz.detach().float().to(device)
    rot0 = gaussians.get_rotation.detach().float().to(device)
    final_xyz = torch.as_tensor(load_ply_xyz(final_ply), dtype=xyz0.dtype, device=device)
    anchors = extract_static_anchors_from_ply(
        xyz0.detach().cpu(),
        final_ply,
        distance_ratio=float(getattr(args, "static_anchor_distance_ratio", 0.002)),
        mutual_nn=bool(getattr(args, "static_anchor_use_mutual_nn", True)),
        use_neighborhood_check=bool(getattr(args, "static_anchor_use_neighborhood_check", False)),
        neighborhood_k=int(getattr(args, "static_anchor_neighborhood_k", 8)),
        chunk_size=int(getattr(args, "rigid_part_endpoint_chunk", 2048)),
    )
    anchor_mask = anchors.mask.to(device)
    frame_ids, frame_times, camera_groups = _camera_groups(scene, args)
    # FINAL spec §7-§9: diffuse the trusted anchors over the canonical kNN graph
    # so the initial belief decays spatially away from them, then bias ONLY the
    # static logit and leave every other logit random. Far from anchors means
    # "less evidence for static", not "automatically moving" (§8).
    neighbor_graph = build_neighbor_graph(
        xyz0,
        k=int(getattr(args, "killing_neighbor_k", 16)),
        chunk_size=int(getattr(args, "killing_neighbor_chunk", 4096)),
    )
    num_parts = int(getattr(args, "num_moving_parts", 1)) + 1
    init_mode = str(getattr(args, "init_mode", "legacy")).strip().lower()
    endpoint_init = None
    # CODEX_soft_labels §1/§17: the persistent label state is a SOFT distribution.
    # ``soft_part_probs`` is that state; hard labels are a temporary projection
    # created only when entering the inner solver (§18, §34).
    soft_labels_mode = bool(getattr(args, "outer_soft_labels", True))
    soft_part_probs: torch.Tensor | None = None
    soft_init = None
    if soft_labels_mode and init_mode == "high_precision_bootstrap":
        # CODEX_high_precision_bootstrap_storage: SEED, do not segment. Only the
        # two ultra-confident cores are decided here; everything else starts at
        # exactly [0.5, 0.5] and the soft outer loop discovers the rest (§1/§14).
        from .bootstrap_init import run_high_precision_bootstrap_initialization

        soft_init = run_high_precision_bootstrap_initialization(
            xyz0, final_xyz, int(getattr(args, "num_moving_parts", 1)),
            trusted_static_mask=anchor_mask, args=args, result_dir=result_dir,
            gt_part_labels=getattr(gaussians, "_gt_part_labels", None),
        )
        soft_part_probs = soft_init.part_probs.to(device)
        static_affinity = soft_part_probs[:, 0]
        init_logits = float(getattr(args, "part_temperature", 1.0)) * torch.log(
            soft_part_probs.clamp_min(1.0e-8))
    elif soft_labels_mode:
        from .endpoint_contrastive_init import run_soft_endpoint_contrastive_initialization

        # §3/§44: initialization receives no temporal memory and cannot.
        soft_init = run_soft_endpoint_contrastive_initialization(
            xyz0, final_xyz, int(getattr(args, "num_moving_parts", 1)),
            trusted_static_mask=anchor_mask, args=args, result_dir=result_dir,
            gt_part_labels=getattr(gaussians, "_gt_part_labels", None),
        )
        soft_part_probs = soft_init.part_probs.to(device)
        static_affinity = soft_part_probs[:, 0]
        init_logits = float(getattr(args, "part_temperature", 1.0)) * torch.log(
            soft_part_probs.clamp_min(1.0e-8))
    elif init_mode == "endpoint_contrastive":
        # CODEX_endpoint_contrastive_initialization: learn only static vs dynamic
        # from endpoint geometry, then split the predicted dynamic support into K.
        # The legacy path below biased only the static logit and left every moving
        # logit random, so the first Killing solve received a random support.
        endpoint_init = run_endpoint_contrastive_initialization(
            xyz0,
            final_xyz,
            int(getattr(args, "num_moving_parts", 1)),
            trusted_static_mask=anchor_mask,
            args=args,
            result_dir=result_dir,
            gt_part_labels=getattr(gaussians, "_gt_part_labels", None),
        )
        static_affinity = 1.0 - endpoint_init.p_dynamic
        init_logits = endpoint_init.part_logits(
            float(getattr(args, "part_temperature", 1.0))).to(device)
    else:
        static_affinity = diffuse_static_anchors(
            neighbor_graph,
            anchor_mask,
            alpha=float(getattr(args, "static_affinity_alpha", 0.85)),
            iterations=int(getattr(args, "static_affinity_iterations", 32)),
        )
        init_logits = initial_part_logits(
            static_affinity,
            anchor_mask,
            num_parts,
            static_gain=float(getattr(args, "static_affinity_gain", 4.0)),
            init_std=float(getattr(args, "part_logit_init_std", 1.0)),
            seed=int(getattr(args, "seed", 20260919)),
        )
    model = PartEmbeddingModel(
        int(xyz0.shape[0]),
        num_parts,
        int(getattr(args, "part_embedding_dim", 8)),
        temperature=float(getattr(args, "part_temperature", 1.0)),
        static_anchor_mask=anchor_mask.detach().cpu(),
        init_std=float(getattr(args, "part_embedding_init_std", 0.05)),
        seed=int(getattr(args, "seed", 20260919)),
        init_logits=init_logits,
    ).to(device)
    with torch.no_grad():
        _p0 = model.probabilities(float(getattr(args, "part_temperature", 1.0)))
        _z0 = _p0.argmax(dim=-1)
    init_diag = {
        "init_mode": init_mode,
        "num_static_anchors": int(anchor_mask.sum().cpu()),
        "static_affinity_mean": float(static_affinity.mean().cpu()),
        "static_affinity_hist": torch.histc(static_affinity.float().cpu(), bins=10, min=0.0, max=1.0).tolist(),
        "initial_hard_counts": [int((_z0 == c).sum().cpu()) for c in range(num_parts)],
        "initial_mean_static_probability": float(_p0[:, 0].mean().cpu()),
        "initial_mean_max_probability": float(_p0.max(dim=-1).values.mean().cpu()),
    }
    if endpoint_init is not None:
        init_diag["endpoint_contrastive"] = _jsonable(endpoint_init.summary)
        init_diag["endpoint_contrastive_gt_evaluation"] = _jsonable(endpoint_init.gt_evaluation)
        # The model must reproduce the membership section 21 handed it, or the
        # first Killing solve is not solving on the support that was initialised.
        realised = float((_z0 != endpoint_init.class_label.to(_z0.device)).float().mean())
        init_diag["endpoint_contrastive_label_mismatch"] = realised
        if realised > 1.0e-6:
            raise RuntimeError(
                "endpoint_contrastive initialisation did not survive the "
                f"PartEmbeddingModel softmax: {realised:.6f} of labels differ")
    print(f"[Nested Killing] init ({init_mode}): {init_diag['num_static_anchors']} anchors, "
          f"hard counts {init_diag['initial_hard_counts']}, "
          f"mean p_static {init_diag['initial_mean_static_probability']:.4f}")

    if bool(getattr(args, "init_only", False)):
        # Spec §28: stop before the first inner Killing solve so the legacy and
        # endpoint_contrastive initialisations can be compared on equal terms.
        _atomic_torch_save({"schema_version": 1, "mask": anchor_mask.detach().cpu(),
                            "target_index": anchors.target_index, "distance": anchors.distance,
                            "summary": anchors.summary}, result_dir / "static_anchors.pt")
        _gt = getattr(gaussians, "_gt_part_labels", None)
        _atomic_torch_save({
            "schema_version": 1,
            "init_mode": init_mode,
            "canonical_xyz": xyz0.detach().cpu(),
            "initial_probabilities": _p0.detach().cpu(),
            "initial_labels": _z0.detach().cpu(),
            "static_anchor_mask": anchor_mask.detach().cpu(),
            # Evaluation only. --init_only skips the ply export that normally
            # carries gt_part, so the ablation would otherwise have nothing to
            # score the LEGACY mode against. It never touches any loss.
            "gt_part_labels": (_gt.detach().cpu()
                               if torch.is_tensor(_gt) and _gt.numel() else None),
            "diagnostics": init_diag,
        }, result_dir / "initial_labels.pt")
        (result_dir / "initialization_summary.json").write_text(
            json.dumps(_jsonable(init_diag), indent=2), encoding="utf-8")
        print(f"[Nested Killing] --init_only: stopped after initialization; wrote "
              f"{result_dir / 'initial_labels.pt'}")
        return {
            "schema_version": 1,
            "source_of_truth": "endpoint_contrastive_initialization_only",
            "init_only": True,
            "init_mode": init_mode,
            "canonical_xyz": xyz0.detach().cpu(),
            "part_probabilities": _p0.detach().cpu(),
            "part_labels": _z0.detach().cpu(),
            "diagnostics": init_diag,
        }

    optimizer = torch.optim.Adam(model.parameters(), lr=float(getattr(args, "part_embedding_lr", 1.0e-2)))
    outer_rows: list[dict[str, object]] = []
    video_rows: list[dict[str, object]] = []
    top_gradient_records: list[dict] = []
    last_cap = (torch.as_tensor(1.0), 0.0, 0.0)
    previous_outer_transitions: int | None = None
    render_graph_rows: list[dict[str, object]] = []
    render_graph_mode = str(getattr(args, "outer_label_mode", "learned")).lower() == "render_graph"
    if render_graph_mode and not bool(getattr(args, "enable_pixel_matching", False)):
        raise RuntimeError("--outer_label_mode render_graph requires --enable_pixel_matching")
    render_graph_labels: torch.Tensor | None = None
    warm_start_field: list[torch.Tensor] | None = None
    inner_rows: list[dict[str, object]] = []
    latest_motion: NestedKillingMotionState | None = None
    latest_prob = None
    previous_hard_for_metrics = None

    outer_iters = max(1, int(getattr(args, "part_outer_iters", 3)))
    embedding_steps = max(1, int(getattr(args, "part_outer_embedding_steps", 10)))
    chunk = int(getattr(args, "rigid_part_endpoint_chunk", 2048))
    bbox_diag = anchors.bbox_diag.to(device)
    anchor_warmup = int(getattr(args, "anchor_hard_freeze_outer_iters", 1))
    base_embedding_lr = float(getattr(args, "part_embedding_lr", 1.0e-2))
    guard_eps = float(getattr(args, "outer_geometry_guard_eps", 1.0e-6))
    guard_backtracks = int(getattr(args, "outer_geometry_guard_backtracks", 4))

    weights = {
        "candidate_nn": float(getattr(args, "lambda_outer_candidate_nn", 1.0)),
        "final_nn": float(getattr(args, "lambda_outer_final_nn", 1.0)),
        "global_com": float(getattr(args, "lambda_outer_global_com", 1.0)),
        "global_inertia": float(getattr(args, "lambda_outer_global_inertia", 1.0)),
        "part_mass": float(getattr(args, "lambda_outer_part_mass", 2.0)),
        "part_com": float(getattr(args, "lambda_outer_part_com", 2.0)),
        "part_inertia": float(getattr(args, "lambda_outer_part_inertia", 2.0)),
        "static_anchor": float(getattr(args, "lambda_static_anchor", 10.0)),
        "embedding_motion": float(getattr(args, "lambda_outer_embedding_motion", 0.0)),
        "candidate_margin": float(getattr(args, "lambda_outer_candidate_margin", 0.1)),
        "entropy": float(getattr(args, "lambda_outer_entropy", 0.001)),
        "occupancy": float(getattr(args, "lambda_outer_occupancy", 0.01)),
        "trust_region": float(getattr(args, "lambda_outer_trust_region", 1.0)),
    }

    # ---- Pixel-mediated matching: cache the FIXED final 3DGS contributor maps
    # once (section 7). The final model and its cameras never change.
    pixel_enabled = bool(getattr(args, "enable_pixel_matching", False))
    pixel_target_maps, final_pixel_cams, final_model = None, [], None
    if pixel_enabled:
        from scene.gaussian_model import GaussianModel

        final_model = GaussianModel(int(getattr(args, "sh_degree", 3)))
        final_model.load_ply(str(final_ply))
        final_pixel_cams = camera_groups[-1]
        pixel_target_maps = build_target_cache(
            final_pixel_cams, final_model,
            result_dir / "pixel_matching" / "target_cache",
            top_l=int(getattr(args, "raster_top_contributors", 4)),
            contributor_min_weight=float(getattr(args, "raster_contributor_min_weight", 1.0e-4)),
            final_model_id=str(final_ply),
        )
        print(f"[Pixel matching] cached {len(pixel_target_maps)} final contributor maps "
              f"({final_model.get_xyz.shape[0]} final gaussians, L="
              f"{int(getattr(args, 'raster_top_contributors', 4))})")

    # ---- Temporal credit outer cycle (CODEX_temporal_credit_outer_cycle) ----
    # §2: the inner solver is untouched; this state lives entirely around
    # complete inner solves. §16: credit uses its OWN spatial graph so the
    # render-preference weights never feed back into the attribution.
    temporal_enabled = bool(getattr(args, "enable_temporal_credit", False))
    temporal_cycle = None
    temporal_graph = neighbor_graph
    temporal_rows: list[dict[str, object]] = []
    # CODEX_soft_labels §42: the last three persistent states, for O_i.
    soft_prob_history: list[torch.Tensor] = []
    if soft_part_probs is not None:
        soft_prob_history.append(soft_part_probs.detach().clone())
    # §44/§45: the causal order is enforced, not merely documented.
    order_guard = TemporalOrderGuard()
    # Motion-boundary split: §4 mandates exactly ONE event for this first pass.
    boundary_split_enabled = bool(getattr(args, "enable_motion_boundary_split", False))
    boundary_split_done = False
    contrast_enabled = bool(getattr(args, "enable_render_response_contrast", False))
    if temporal_enabled:
        if not render_graph_mode:
            raise RuntimeError("--enable_temporal_credit requires --outer_label_mode render_graph")
        credit_k = int(getattr(args, "temporal_credit_graph_k", 16))
        if credit_k != int(getattr(args, "killing_neighbor_k", 16)):
            temporal_graph = build_neighbor_graph(
                xyz0, k=credit_k,
                chunk_size=int(getattr(args, "killing_neighbor_chunk", 4096)))
        temporal_cycle = TemporalCreditCycle(
            int(xyz0.shape[0]), num_parts, device=device,
            beta_global=float(getattr(args, "temporal_global_reward_weight", 0.25)),
            beta_component=float(getattr(args, "temporal_component_reward_weight", 0.0)),
            reward_clip=float(getattr(args, "temporal_reward_clip", 1.0)),
            memory_decay=float(getattr(args, "temporal_memory_decay", 0.8)),
            memory_global_decay=float(getattr(args, "temporal_memory_global_decay", 1.0)),
            lambda_memory=float(getattr(args, "lambda_temporal_memory", 0.25)),
            memory_margin_cap=float(getattr(args, "temporal_memory_margin_cap", 0.5)),
            lambda_render=float(getattr(args, "temporal_score_render_weight", 1.0)),
            lambda_match=float(getattr(args, "temporal_score_match_weight", 0.05)),
        )
        print(f"[Temporal credit] enabled: credit graph k={temporal_graph.k}, "
              f"beta_global={temporal_cycle.beta_global}, gamma={temporal_cycle.memory_decay}, "
              f"lambda_mem={temporal_cycle.lambda_memory}, cap={temporal_cycle.memory_margin_cap}; "
              f"memory starts at exactly zero (§19/§40)")

    temperature = float(getattr(args, "part_temperature", 1.0))
    previous_strain: float | None = None
    polarization_rows: list[dict] = []                       # CODEX_DEBUG §11/§12
    previous_soft_probs = (None if soft_part_probs is None
                           else soft_part_probs.detach().clone())
    outer_executed = 0
    outer_progress = tqdm(range(outer_iters), desc="Nested part embeddings", unit="outer")
    for outer in outer_progress:
        # §22: hard pinning is allowed only as a finite warmup.
        apply_anchor = outer < anchor_warmup
        with torch.no_grad():
            frozen_prob = model.probabilities(temperature, apply_anchor=apply_anchor).detach()
            # §2: z_i = argmax_k p_ik, frozen for the entire inner solve.
            # §13/§16: in render_graph mode the previous propagation's labels are
            # the partition; there is no learned outer optimizer.
            if soft_labels_mode:
                # CODEX_soft_labels §18/§34: the ONLY place a hard label exists.
                # It is frozen for this inner solve and discarded afterwards;
                # ``soft_part_probs`` is never overwritten by its own one-hot (§43).
                frozen_hard = soft_part_probs.argmax(dim=-1).detach()
                frozen_prob = torch.nn.functional.one_hot(
                    frozen_hard, int(soft_part_probs.shape[1])).to(soft_part_probs)
            else:
                frozen_hard = (
                    render_graph_labels.detach()
                    if (render_graph_mode and render_graph_labels is not None)
                    else frozen_prob.argmax(dim=-1).detach()
                )
                if render_graph_mode and render_graph_labels is not None:
                    frozen_prob = torch.nn.functional.one_hot(
                        frozen_hard, int(frozen_prob.shape[1])).to(frozen_prob)

        # CODEX_robust_inner_killing: one shared spatial field + scalar speed,
        # selected by --inner_solver. The legacy per-frame solver is untouched.
        _shared = str(getattr(args, "inner_solver", "per_frame")) == "shared_scalar"
        _inner_solver = solve_shared_killing_sequence if _shared else solve_nested_killing_sequence
        # CODEX_MERGED §41/§52 need the final static geometry, and §64's support
        # metrics need GT. GT is evaluation-only and never enters construction.
        _extra: dict = {}
        if _shared:
            _extra["final_xyz"] = final_xyz
            # CODEX_DEBUG §3/§8/§9: hand the support module the PERSISTENT SOFT
            # P, not ``frozen_prob``. ``frozen_prob`` is one_hot(argmax P) by
            # construction a few lines above -- correct as the temporary hard
            # gate (CODEX_soft_labels §18/§34), but passing it as
            # ``soft_part_probs`` destroys exactly the confidence spread the
            # merged contract's core/attachment/fit-weight gates run on, which
            # is why they reported core == clean with min P_dyn = 1.000.
            # ``hard_label=frozen_hard`` still carries the hard gate (§2).
            if soft_labels_mode and soft_part_probs is not None:
                _extra["soft_part_probs"] = soft_part_probs.detach()
                _extra["soft_part_probs_name"] = "soft_part_probs (persistent outer state)"
            else:
                # render_graph / legacy-hard mode genuinely has no soft state;
                # the handoff diagnostics will report that honestly.
                _extra["soft_part_probs"] = frozen_prob
                _extra["soft_part_probs_name"] = (
                    "frozen_prob (one-hot; render_graph/legacy-hard mode has no soft P)")
            _gt = getattr(gaussians, "_gt_part_labels", None)
            if _gt is not None and int(torch.as_tensor(_gt).numel()) == int(xyz0.shape[0]):
                _extra["gt_moving"] = (torch.as_tensor(_gt, device=device).reshape(-1) > 0)
        motion = _inner_solver(
            canonical_xyz=xyz0,
            canonical_rotation=rot0,
            part_prob=frozen_prob,
            frame_times=frame_times.to(device),
            cameras_by_frame=camera_groups,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            dataset=dataset,
            args=args,
            output_dir=result_dir / f"outer_{outer:04d}",
            log_fn=log_fn,
            outer_iter=outer,
            neighbor_graph=neighbor_graph,
            hard_label=frozen_hard,
            warm_start_field=warm_start_field,
            **_extra,
        )
        order_guard.record_solve()           # CODEX_soft_labels §44/§45
        # §47: reuse this solve's field as the next inner solve's initialisation.
        if bool(getattr(args, "killing_warm_start", True)):
            warm_start_field = [
                entry["control_velocity"].detach().clone() for entry in motion.field_debug
            ]
        inner_rows.extend(motion.history)
        latest_motion = motion

        # §25: candidates are the per-Gaussian integrated endpoints, with the
        # static identity always present as candidate 0.
        candidates = candidates_from_local_endpoints(xyz0, motion.moving_endpoints.to(device))
        cand_nn = candidate_nn_energies(candidates, final_xyz, chunk)

        # §19: decide whether this motion may touch the memberships at all.
        # FINAL spec §31/§32: there is NO global render-vs-static gate and NO
        # global endpoint gate. The first partition is intentionally imperfect;
        # a bad articulated render means the partition still needs repair, not
        # that the E-step should be frozen. Only numerical invalidity (§33) may
        # stop anything, and that is handled inside the inner solver.
        delta_e = endpoint_usefulness(cand_nn, frozen_prob)
        inner_metrics = dict(motion.acceptance)
        for k, value in enumerate(delta_e.tolist()):
            inner_metrics[f"delta_endpoint_part_{k + 1}"] = float(value)
        previous_strain = float(motion.acceptance.get("mean_strain_norm", 0.0))
        separation = float((cand_nn[:, 1:] - cand_nn[:, 0:1]).abs().median())
        outer_executed += 1

        # §28: build the final-cloud targets ONCE and hold them fixed for every
        # embedding step, so the target cannot chase a shrinking part.
        targets = build_outer_targets(
            final_xyz=final_xyz,
            candidates=candidates,
            temperature_final_assignment=float(getattr(args, "part_final_assignment_temperature", 0.01)),
            chunk_size=chunk,
        )
        q_detached = targets["q"]
        with torch.no_grad():
            p_old = model.probabilities(temperature, apply_anchor=apply_anchor).detach()
            hard_before_outer = p_old.argmax(dim=-1)

        # ======================================================================
        # D. OUTER E-STEP: RENDER DRIVES LABELS (§1, §22, §23, §27)
        # Motion is frozen and detached; only membership parameters train.
        # ======================================================================
        frozen_R = motion.part_rotation.detach().to(device)
        frozen_t = motion.part_translation.detach().to(device)
        frozen_q = motion.part_quaternion.detach().to(device)
        kinetic_energy = motion.kinetic_energy.detach().to(device)
        # §17: dE_i = E_i1 - E_i0 is the motion-compatibility evidence.
        delta_energy = (cand_nn[:, 1:].min(dim=1).values - cand_nn[:, 0]).detach()

        # section 38: T_k(T) changed, so recompute E_pixel ONCE here and keep it
        # fixed for every label epoch of this E-step.
        pixel_evidence = None
        if pixel_enabled:
            with torch.no_grad():
                cand_q = torch.stack([
                    apply_rotation_to_gaussian_quaternions(rot0, frozen_q[-1][k])
                    for k in range(int(frozen_R.shape[1]))
                ], dim=1)
                n_g = int(xyz0.shape[0])
                k1 = int(frozen_R.shape[1])
                scaling = gaussians.get_scaling.detach()
                cov = covariance_from_scale_rotation(
                    scaling[:, None, :].expand(n_g, k1, 3).reshape(-1, 3),
                    cand_q.reshape(-1, 4),
                ).reshape(n_g, k1, 3, 3)
            pixel_evidence, pixel_sparse = candidate_pixel_evidence(
                candidate_xyz=candidates.detach(), candidate_cov3d=cov,
                opacity=gaussians.get_opacity.detach(),
                cameras=final_pixel_cams, target_maps=pixel_target_maps,
                final_xyz=final_xyz, bbox_diag=float(bbox_diag),
                window=int(getattr(args, "pixel_footprint_window", 3)),
                sigma_z_ratio=float(getattr(args, "pixel_sigma_z_ratio", 0.02)),
                lambda_overlap=float(getattr(args, "lambda_pixel_overlap", 0.5)),
                lambda_xyz=float(getattr(args, "lambda_pixel_xyz", 1.0)),
                lambda_depth=float(getattr(args, "lambda_pixel_depth", 0.5)),
                collect_sparse=True,
                # §4: the contrastive fingerprint needs E_ik per observation.
                collect_per_view=bool(getattr(args, "enable_render_response_contrast", False)),
            )
            # §8: render the CURRENT hard transported scene into the same final
            # cameras and build the current source<->final sparse matches.
            with torch.no_grad():
                hard_now = frozen_prob.argmax(dim=-1)
                cur_xyz = torch.stack([xyz0] + [candidates[:, k] for k in range(1, int(candidates.shape[1]))], dim=1)
                sel = torch.zeros((int(xyz0.shape[0]), int(candidates.shape[1]), 1), device=device)
                sel.scatter_(1, hard_now[:, None, None], 1.0)
                transported = (sel * cur_xyz).sum(dim=1)
                source_maps = [
                    render_contributors(
                        cam, gaussians,
                        top_l=int(getattr(args, "raster_top_contributors", 4)),
                        contributor_min_weight=float(getattr(args, "raster_contributor_min_weight", 1.0e-4)),
                        override_xyz=transported,
                    )
                    for cam in final_pixel_cams
                ]
                m_final = int(final_xyz.shape[0])
                bb = float(bbox_diag)
                si, tj, aw = accumulate_pair_affinity(
                    source_maps, pixel_target_maps, num_target=m_final,
                    sigma_z=float(getattr(args, "pixel_sigma_z_ratio", 0.02)) * bb,
                    delta_z=float(getattr(args, "pixel_delta_z_ratio", 0.15)) * bb,
                )
                conf_stats = normalized_match_confidence(
                    si, tj, aw, num_source=int(xyz0.shape[0]), num_target=m_final)
                attract, match_mass = current_pixel_attraction_loss(
                    transported, final_xyz, si, tj, aw, bbox_diag=bb)
                pixel_target_support = conf_stats["target_mass"]
                _atomic_torch_save({
                    "schema_version": 1,
                    "source_ids": si.cpu(), "target_ids": tj.cpu(), "weights": aw.cpu(),
                    "p_j_given_i": conf_stats["p_j_given_i"].cpu(),
                    "p_i_given_j": conf_stats["p_i_given_j"].cpu(),
                    "symmetric": conf_stats["symmetric"].cpu(),
                    "source_mass": conf_stats["source_mass"].cpu(),
                    "target_mass": conf_stats["target_mass"].cpu(),
                }, result_dir / "pixel_matching" / f"outer_{outer:04d}_current_sparse_matches.pt")
                pixel_summary = {
                    "outer": int(outer),
                    "num_sparse_pairs": int(si.numel()),
                    "num_source_matched": int((conf_stats["source_mass"] > 0).sum()),
                    "num_target_matched": int((conf_stats["target_mass"] > 0).sum()),
                    "current_pixel_attraction": float(attract),
                    "mean_match_mass": float(match_mass.mean()),
                    "mean_pixel_confidence": float(pixel_evidence.confidence.mean()),
                    "coverage_fraction": float((pixel_evidence.num_views.max(dim=1).values > 0).float().mean()),
                    "fraction_pixel_prefers": [
                        float((pixel_evidence.energy.argmin(dim=1) == c).float().mean())
                        for c in range(int(pixel_evidence.energy.shape[1]))
                    ],
                }
                (result_dir / "pixel_matching" / f"outer_{outer:04d}_summary.json").write_text(
                    json.dumps(pixel_summary, indent=2), encoding="utf-8")
                print(f"[Pixel matching] outer {outer}: {si.numel()} sparse pairs, "
                      f"attraction {float(attract):.5f}, conf {float(pixel_evidence.confidence.mean()):.5f}")
            _atomic_torch_save(
                {k: v.cpu() for k, v in pixel_evidence.as_dict().items()},
                result_dir / "pixel_matching" / f"outer_{outer:04d}_candidate_evidence.pt",
            )

        # ======================================================================
        # D'. SIMPLE RENDER-MATCHING OUTER LOOP (render_graph mode)
        # Bypasses the whole learned objective: no video ST loss, no NN/mass/
        # COM/inertia/kinetic/entropy terms, no temperature schedule, no
        # gradient caps, no embedding optimizer (§4, §16, §17, §18).
        # ======================================================================
        # ======================================================================
        # D''. TEMPORAL CREDIT (§27, §28 steps C and D)
        # The solve above produced T_k^(m) from z^(m). Score that SOLVED state,
        # then credit the transition z^(m-1) -> z^(m) that a previous proposal
        # caused. On outer 0 this only establishes the baseline S^(0) and the
        # fixed normalisation -- no reward exists yet (§7).
        # ======================================================================
        temporal_energy = pixel_evidence.energy if pixel_enabled else None
        if temporal_cycle is not None:
            with torch.no_grad():
                render_raw, s_render, render_support, render_detail = evaluate_render_score(
                    cameras_by_frame=camera_groups,
                    gaussians=gaussians, pipe=pipe, background=background,
                    dataset=dataset, args=args,
                    positions=motion.positions.detach().to(device),
                    rotations=motion.rotations.detach().to(device),
                    num_gaussians=int(xyz0.shape[0]),
                    attribute=bool(getattr(args, "temporal_score_attribute", True)),
                    top_l=int(getattr(args, "raster_top_contributors", 4)),
                    contributor_min_weight=float(
                        getattr(args, "raster_contributor_min_weight", 1.0e-4)),
                    frame_stride=int(getattr(args, "temporal_score_frame_stride", 1)),
                )
                match_kwargs = dict(
                    pixel_energy=pixel_evidence.energy,
                    pixel_num_views=pixel_evidence.num_views,
                    final_nn_energy=cand_nn,
                    labels=frozen_hard,
                    lambda_final_nn=float(getattr(args, "lambda_score_final_nn", 0.1)),
                    min_support_views=int(getattr(args, "min_support_views", 2)),
                )
                # §6/§15: the constants are taken ONCE at the baseline state and
                # then never recomputed, or S^(m) - S^(m-1) is not comparable.
                if temporal_cycle.normalization is None:
                    # First pass only fixes the §30 penalty; the baseline match
                    # score is then RE-computed with it so the baseline and every
                    # later iteration share one convention and one population.
                    _, _, _, probe_sel, probe_valid = evaluate_match_score(**match_kwargs)
                    penalty = unsupported_penalty_from(probe_sel, probe_valid)
                    match_raw, pixel_raw, nn_raw, s_match, match_valid = evaluate_match_score(
                        **match_kwargs, unsupported_penalty=penalty)
                    temporal_cycle.normalization = establish_normalization(
                        render_raw, match_raw, s_render, s_match, match_valid,
                        outer=outer, unsupported_penalty=penalty)
                    print(f"[Temporal credit] fixed normalisation from outer {outer}: "
                          f"{temporal_cycle.normalization.as_dict()}")
                    print(f"[Temporal credit] §30 population fixed at N="
                          f"{int(probe_valid.numel())}; {int((~probe_valid).sum())} unsupported "
                          f"selected candidates carry the fixed penalty {penalty:.5f}")
                else:
                    match_raw, pixel_raw, nn_raw, s_match, match_valid = evaluate_match_score(
                        **match_kwargs,
                        unsupported_penalty=temporal_cycle.normalization.unsupported_penalty)
                quality = combine_global_score(
                    render_raw, match_raw, pixel_raw, nn_raw, temporal_cycle.normalization,
                    lambda_render=temporal_cycle.lambda_render,
                    lambda_match=temporal_cycle.lambda_match, detail=render_detail,
                )
                local = combine_local_score(
                    s_render, s_match, match_valid, render_support,
                    temporal_cycle.normalization, temporal_graph.to(device=device),
                    lambda_render=temporal_cycle.lambda_render,
                    lambda_match=temporal_cycle.lambda_match,
                )
                record = temporal_cycle.observe(
                    outer=outer, labels=frozen_hard, quality=quality, local=local,
                    graph=temporal_graph.to(device=device), probabilities=latest_prob,
                )
                if not record["is_baseline"]:
                    # CODEX_soft_labels §45: a memory update is only legal once two
                    # solved states exist; the guard raises otherwise.
                    order_guard.record_memory_update()
            record.update(order_guard.as_dict())
            temporal_rows.append(record)                                       # §35
            if record["is_baseline"]:
                print(f"[Temporal credit] outer {outer}: BASELINE S^(0)="
                      f"{quality.total:.6f} (render {quality.normalized_render:.6f}, "
                      f"match {quality.normalized_match:.6f}); memory = 0, no reward (§7)")
            else:
                print(f"[Temporal credit] outer {outer}: S={quality.total:.6f} "
                      f"R_rel={record['relative_global_reward']:+.5f}  "
                      f"changed {record['num_changed']} in "
                      f"{record['num_changed_components']} components  "
                      f"reward mean {record['mean_local_reward_changed']:+.4f} "
                      f"(+{record['num_positive_reward_changes']}/"
                      f"-{record['num_negative_reward_changes']})  "
                      f"|M| mean {record['mean_abs_memory']:.5f} "
                      f"max {record['max_abs_memory']:.5f}  "
                      f"oscillations {record['num_label_oscillations']}")
                for entry in (temporal_cycle._artifact.get("transition_report") or [])[:6]:
                    print(f"     {entry['transition']:>8s}  n={entry['count']:6d}  "
                          f"reward {entry['mean_reward']:+.4f} "
                          f"(pos {entry['fraction_positive_reward']*100:5.1f}%)  "
                          f"render {entry['mean_render_improvement']:+.6f}  "
                          f"match {entry['mean_match_improvement']:+.6f}")

            # §23/§24: bias the candidate score for the NEXT proposal. §25: only
            # the score changes; the render-graph mechanism stays as it is.
            raw_pref = render_class_preference(
                pixel_evidence.energy, pixel_evidence.num_views,
                min_support_views=int(getattr(args, "min_support_views", 2)),
                tau_render=float(getattr(args, "tau_render_preference", 1.0)),
            )
            temporal_energy, bias_stats = temporal_cycle.bias_energy(
                pixel_evidence.energy, raw_margin=raw_pref["raw_margin"])
            temporal_rows[-1].update({f"bias_{k}": v for k, v in bias_stats.items()})
            payload = temporal_cycle.artifact(pixel_evidence.energy, temporal_energy)
            if payload:
                _atomic_torch_save(payload,                                     # §36
                                   result_dir / f"temporal_credit_outer_{outer:03d}.pt")

        # ======================================================================
        # D'''. SOFT OUTER UPDATE (CODEX_soft_labels §21-§24, §29-§34, §39, §41)
        # P^(m) -> P^(m+1), entirely in probability space. No hard seeds, no
        # clamping except the trusted static anchors (§22/§31).
        # ======================================================================
        if soft_labels_mode:
            pref = render_class_preference(
                pixel_evidence.energy, pixel_evidence.num_views,
                min_support_views=int(getattr(args, "min_support_views", 2)),
                tau_render=float(getattr(args, "tau_render_preference", 1.0)),
            )
            memory_now = temporal_cycle.memory.to(device) if temporal_cycle is not None else None
            if temporal_cycle is not None:
                # §44: before two solved states exist the memory must be exactly 0.
                order_guard.check_memory_is_zero(memory_now)
            update = soft_outer_update(
                soft_probs=soft_part_probs,
                energy=pixel_evidence.energy,
                normalized_margin=pref["normalized_margin"],
                comparable=pref["comparable"],
                graph=neighbor_graph.to(device=device),
                trusted_mask=anchor_mask,
                memory=memory_now,
                valid=pixel_evidence.num_views >= int(getattr(args, "min_support_views", 2)),
                tau_render=float(getattr(args, "outer_render_temperature", 1.0)),
                lambda_memory=float(getattr(args, "lambda_temporal_memory", 0.25)),
                smoothing=float(getattr(args, "outer_probability_smoothing", 0.0)),
                # Corrections §15/§17: validity-aware alpha is the new default.
                confidence_mode=str(getattr(args, "outer_confidence_mode",
                                            "candidate_validity_entropy")),
            )
            # Corrections §20: evaluation only, never touches the update above.
            _gt_labels = getattr(gaussians, "_gt_part_labels", None)
            if _gt_labels is not None and int(torch.as_tensor(_gt_labels).numel()) and \
                    update.num_valid is not None:
                ev = render_evidence_against_gt(update.render_probs, update.num_valid, _gt_labels)
                if ev.get("available"):
                    for group in ("valid0", "valid1", "valid2plus", "all"):
                        stats = ev.get(group, {})
                        if stats.get("n"):
                            print(f"     render evidence [{group:>10s}] n={stats['n']:6d} "
                                  f"acc {stats['accuracy']:6.2f}  prec {stats['precision']:6.2f}  "
                                  f"rec {stats['recall']:6.2f}  F1 {stats['f1']:6.2f}")
                    update.diagnostics["render_evidence_gt"] = ev
            # ==================================================================
            # NEW: RENDER-RESPONSE CONTRASTIVE REFINEMENT (§2, §33)
            # P_base (render + KNN + existing memory) -> P_new. Memory is NOT
            # touched here and is not moved after this stage (§2, §22).
            # ==================================================================
            contrast_diag = None
            if contrast_enabled and pixel_evidence.per_view_energy is not None:
                fingerprints = build_render_response_fingerprints(
                    pixel_evidence.per_view_energy, pixel_evidence.per_view_valid,
                    num_classes=int(soft_part_probs.shape[1]))
                cgraph = build_render_contrast_graph(
                    fingerprints, neighbor_graph.to(device=device),
                    positive_threshold=float(getattr(args, "render_contrast_positive_threshold", 0.70)),
                    negative_threshold=float(getattr(args, "render_contrast_negative_threshold", 0.70)),
                    min_shared_obs=int(getattr(args, "render_contrast_min_shared_obs", 2)))
                refined = refine_soft_labels_with_render_contrast(
                    old_probs=update.soft_before, base_probs=update.soft_after,
                    contrast_graph=cgraph, trusted_static_mask=anchor_mask,
                    num_steps=int(getattr(args, "render_contrast_steps", 10)),
                    lr=float(getattr(args, "render_contrast_lr", 1.0e-2)),
                    lambda_preserve=float(getattr(args, "lambda_render_contrast_preserve", 1.0)),
                    lambda_agree=float(getattr(args, "lambda_render_contrast_agree", 1.0)),
                    lambda_disagree=float(getattr(args, "lambda_render_contrast_disagree", 0.25)),
                    sign_temperature=float(getattr(args, "render_contrast_sign_temperature", 0.10)))
                contrast_diag = dict(refined.diagnostics)
                _gt = getattr(gaussians, "_gt_part_labels", None)
                if _gt is not None and int(torch.as_tensor(_gt).numel()):
                    contrast_diag.update(contrast_gt_diagnostics(   # §37, evaluation only
                        cgraph, update.soft_after, refined.probs, _gt))
                d = contrast_diag
                print(f"[RenderContrast] outer {outer}: +{d['num_positive']} / "
                      f"-{d['num_negative']} edges  conf mean "
                      f"{d.get('mean_response_confidence', float('nan')):.4f} "
                      f"(>0.5 {d.get('fraction_high_response_confidence', float('nan'))*100:.1f}%)  "
                      f"loss P/A/D {d['loss_preserve']:.5f}/{d['loss_agree']:.5f}/"
                      f"{d['loss_disagree']:.5f}  |dL| {d.get('mean_update_norm_before_contrast', float('nan')):.4f}"
                      f"->{d.get('mean_update_norm_after_contrast', float('nan')):.4f}  "
                      f"argmax changes {d['num_argmax_changes_due_to_contrast']}")
                if "contrast_changes_improving_gt" in d:
                    print(f"     GT (eval only): +edges same-class "
                          f"{d.get('positive_edges_same_gt_class', float('nan')):.1f}%  "
                          f"changes {d['contrast_changes_total']} "
                          f"(better {d['contrast_changes_improving_gt']}, "
                          f"worse {d['contrast_changes_worsening_gt']})")
                # §33: P_new replaces P_base as the persistent state.
                update.soft_after = refined.probs
                update.hard_after = refined.probs.argmax(dim=1)

            osc = soft_oscillation(update.soft_after, soft_prob_history[-1] if soft_prob_history else None,
                                   soft_prob_history[-2] if len(soft_prob_history) >= 2 else None)
            soft_prob_history.append(update.soft_after.detach().clone())
            if len(soft_prob_history) > 3:
                soft_prob_history.pop(0)

            payload = {k: (v.cpu() if torch.is_tensor(v) else v)
                       for k, v in update.as_dict().items()}                    # §39
            if contrast_diag is not None:                                        # §35
                payload["soft_part_probs_base"] = refined.base_probs.detach().cpu()
                payload["soft_part_probs_after_contrast"] = refined.probs.detach().cpu()
                payload["render_response_confidence"] = cgraph.response_confidence.detach().cpu()
                payload["render_contrast_diagnostics"] = contrast_diag
            payload["memory_before"] = (torch.zeros_like(update.soft_after).cpu()
                                        if memory_now is None else memory_now.detach().cpu())
            payload["memory_after"] = (torch.zeros_like(update.soft_after).cpu()
                                       if temporal_cycle is None
                                       else temporal_cycle.memory.detach().cpu())
            payload["schema_version"] = 1
            _atomic_torch_save(payload, result_dir / f"soft_outer_{outer:03d}.pt")

            # CODEX_DEBUG §11/§12: measure polarization on BOTH sides of the
            # outer update, so saturation arriving from the outer loop can be
            # told apart from saturation manufactured at the inner boundary.
            # Measurement only -- the update itself is untouched (§13).
            polarization_rows.append({
                "stage": "before_outer_update",
                **polarization_row(outer, update.soft_before
                                   if hasattr(update, "soft_before") else soft_part_probs,
                                   previous_soft_probs)})
            soft_part_probs = update.soft_after                                  # §34: persistent
            polarization_rows.append({
                "stage": "after_outer_update",
                **polarization_row(outer, soft_part_probs, previous_soft_probs)})
            previous_soft_probs = soft_part_probs.detach().clone()
            latest_prob = soft_part_probs

            # ==================================================================
            # MOTION-BOUNDARY 3DGS SPLIT (CODEX_motion_boundary_3dgs_split §4)
            # Exactly ONE event, here: after OUTER 0 produced P^(1) and before
            # INNER 1. Temporal memory is still zero at this point (§22), so no
            # credit is disturbed. Fixes the REPRESENTATION where one primitive
            # spans a rigid boundary; the label architecture is untouched.
            # ==================================================================
            if boundary_split_enabled and not boundary_split_done:
                boundary_split_done = True
                candidates, split_diag = find_motion_boundary_split_candidates(
                    xyz=gaussians.get_canonical_xyz.detach().float(),
                    scaling=gaussians.get_scaling.detach(),
                    rotation=gaussians._rotation.detach(),
                    soft_probs=soft_part_probs,
                    graph=neighbor_graph.to(device=device),
                    trusted_static_mask=anchor_mask,
                    local_class_min=float(getattr(args, "boundary_split_local_class_min", 0.15)),
                    extent_sigma=float(getattr(args, "boundary_split_extent_sigma", 2.0)),
                    size_threshold=float(getattr(args, "boundary_split_size_threshold", 0.0)),
                    percent_dense=float(getattr(args, "percent_dense", 0.01)),
                    scene_extent=float(getattr(scene, "cameras_extent", 1.0) or 1.0),
                    gradient_ridge=float(getattr(args, "boundary_split_gradient_ridge", 1.0e-6)),
                    confident_class=float(getattr(args, "boundary_split_confident_class", 0.6)),
                    max_fraction=float(getattr(args, "boundary_split_max_fraction", 0.01)),
                    max_count=int(getattr(args, "boundary_split_max_count", 0)),
                )
                print(f"[Boundary split] candidates: {split_diag}")
                if candidates:
                    xyz_before = gaussians.get_canonical_xyz.detach().cpu().clone()
                    extra = {"anchor_mask": anchor_mask.float()}
                    if temporal_cycle is not None:
                        extra["memory"] = temporal_cycle.memory        # §22: zeroed for children
                    split = apply_motion_boundary_split(
                        gaussians, candidates, soft_part_probs,
                        offset_sigma=float(getattr(args, "boundary_split_offset_sigma", 0.5)),
                        max_offset_sigma=float(getattr(args, "boundary_split_max_offset_sigma", 1.0)),
                        label_mix=float(getattr(args, "boundary_split_label_mix", 0.05)),
                        opacity_mode=str(getattr(args, "boundary_split_opacity_mode", "weighted_mass")),
                        extra_tensors=extra,
                    )
                    soft_part_probs = split.soft_part_probs.to(device)
                    latest_prob = soft_part_probs
                    rebuilt = split.diagnostics["extra"]
                    anchor_mask = rebuilt["anchor_mask"].to(device) > 0.5
                    if temporal_cycle is not None:
                        temporal_cycle.memory = rebuilt["memory"].to(device)
                        # §22: children carry no inherited credit.
                        order_guard.check_memory_is_zero(temporal_cycle.memory)
                    # §26/§27: rebuild the graph, then cut sibling smoothing edges.
                    neighbor_graph = rebuild_state_after_boundary_split(
                        gaussians, split,
                        k=int(getattr(args, "killing_neighbor_k", 16)),
                        chunk_size=int(getattr(args, "killing_neighbor_chunk", 4096)))
                    temporal_graph = neighbor_graph
                    # §29: every Gaussian-indexed cache is now invalid.
                    invalidate_per_gaussian_caches_after_split(warm_start_field)
                    warm_start_field = None
                    xyz0 = gaussians.get_canonical_xyz.detach().float().to(device)
                    rot0 = gaussians.get_rotation.detach().float().to(device)
                    soft_prob_history.clear()
                    soft_prob_history.append(soft_part_probs.detach().clone())
                    save_boundary_split_artifacts(
                        result_dir / "motion_boundary_split", split,
                        xyz_before, gaussians.get_canonical_xyz.detach().cpu())
                    print(f"[Boundary split] {split.num_parents} parents -> "
                          f"{split.num_children} children; N {split.diagnostics['n_old']} -> "
                          f"{split.diagnostics['n_new']}; graph rebuilt "
                          f"({neighbor_graph.num_edges} edges), caches invalidated")
                else:
                    print("[Boundary split] no candidate passed the reach test; nothing split")
            k1 = int(soft_part_probs.shape[1])
            row = {                                                              # §41, §53
                "outer": outer,
                **{f"num_part_{c}": int((update.hard_after == c).sum()) for c in range(k1)},
                **update.diagnostics,
                **osc,
                "num_changed": int(update.diagnostics["num_argmax_changes"]),
                "mean_render_alpha": float(update.alpha.mean()),
                "num_valid_gaussians": int(pref["comparable"].sum()),
            }
            render_graph_rows.append(row)
            outer_rows.append({**row, "step": -1, "temperature": 1.0,
                               "mean_static_probability": float(soft_part_probs[:, 0].mean()),
                               "mean_label_confidence": float(soft_part_probs.max(dim=1).values.mean())})
            d = update.diagnostics
            print(f"[Soft outer] outer {outer}: H {d['mean_entropy_before']:.4f}->"
                  f"{d['mean_entropy_after']:.4f}  mean|dP|_1 {d['mean_l1_probability_change']:.5f}  "
                  f"argmax churn {d['num_argmax_changes']} "
                  f"({d['fraction_argmax_changes']*100:.2f}%)  "
                  f"max(P)>0.90 {d['fraction_max_prob_gt_090']*100:.1f}%  "
                  f"classes {[int((update.hard_after == c).sum()) for c in range(k1)]}  "
                  f"soft reversal {osc['fraction_soft_update_reversal']*100 if osc['fraction_soft_update_reversal'] == osc['fraction_soft_update_reversal'] else float('nan'):.1f}%")
            continue

        if render_graph_mode:
            rg = run_render_graph_outer(
                energy=temporal_energy if temporal_energy is not None else pixel_evidence.energy,
                num_views=pixel_evidence.num_views,
                graph=neighbor_graph.to(device=device),
                anchor_mask=anchor_mask,
                seed_confidence=float(getattr(args, "render_seed_confidence", 0.75)),
                min_support_views=int(getattr(args, "min_support_views", 2)),
                tau_render=float(getattr(args, "tau_render_preference", 1.0)),
                iterations=int(getattr(args, "graph_iterations", 1000)),
                tol=float(getattr(args, "graph_tolerance", 1.0e-4)),
                boundary_aware_graph=bool(getattr(args, "boundary_aware_graph", False)),
                per_class_seeds=bool(getattr(args, "per_class_seeds", True)),
                min_seeds_per_class=int(getattr(args, "min_seeds_per_class", 32)),
                absolute_margin_floor=float(getattr(args, "absolute_margin_floor", 0.05)),
            )
            previous_labels = render_graph_labels
            render_graph_labels = rg.final_label.to(device)
            latest_prob = rg.propagated_probabilities.to(device)

            saved = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in rg.as_dict().items()}
            if temporal_cycle is not None and pixel_enabled:
                # In the temporal arm ``candidate_energy`` is the memory-BIASED
                # score the graph actually decided on. Keep the unbiased render
                # energy beside it, or a later "raw render matcher" analysis would
                # silently be reading the biased one and the A/B arms would not be
                # comparable on that quantity.
                saved["candidate_energy_biased"] = saved["candidate_energy"]
                saved["candidate_energy"] = pixel_evidence.energy.detach().cpu()
                saved["temporal_memory"] = temporal_cycle.memory.detach().cpu()
            _atomic_torch_save(saved, result_dir / f"render_graph_outer_{outer:03d}.pt")   # §21

            k1 = int(rg.propagated_probabilities.shape[1])
            changed = 0 if previous_labels is None else int((previous_labels != render_graph_labels).sum())
            seeds = rg.seed_mask
            row = {                                                                  # §22
                "outer": outer,
                **{f"num_part_{c}": int((render_graph_labels == c).sum()) for c in range(k1)},
                "num_changed": changed,
                "num_render_seeds": int(seeds.sum()),
                **{f"num_part{c}_seeds": int((seeds & (rg.seed_label == c)).sum()) for c in range(k1)},
                "mean_raw_margin": float(rg.raw_margin.mean()),
                "mean_normalized_margin": float(rg.normalized_margin.mean()),
                # The energy the graph decided on (biased in the temporal arm)...
                **{f"mean_energy_part{c}": float(rg.candidate_energy[:, c].mean()) for c in range(k1)},
                # ...and the unbiased render energy, so the arms stay comparable.
                **({f"mean_render_energy_part{c}": float(pixel_evidence.energy[:, c].mean())
                    for c in range(k1)} if pixel_enabled else {}),
                "mean_seed_margin": float(rg.normalized_margin[seeds].mean()) if bool(seeds.any()) else 0.0,
                "mean_nonseed_margin": float(rg.normalized_margin[~seeds].mean()) if bool((~seeds).any()) else 0.0,
                "graph_iterations": rg.graph_iterations,
                "graph_final_delta": rg.graph_final_delta,
                "graph_converged": bool(rg.graph_converged),
                "num_valid_gaussians": int((rg.candidate_valid.sum(dim=1) >= 2).sum()),
            }
            render_graph_rows.append(row)
            outer_rows.append({**row, "step": -1, "temperature": 1.0,
                               "mean_static_probability": float(latest_prob[:, 0].mean()),
                               "mean_label_confidence": float(latest_prob.max(dim=1).values.mean())})
            print(f"[Render graph] outer {outer}: seeds {int(seeds.sum())} "
                  f"({', '.join(str(int((seeds & (rg.seed_label == c)).sum())) for c in range(k1))}), "
                  f"labels {[int((render_graph_labels == c).sum()) for c in range(k1)]}, "
                  f"changed {changed}, iters {rg.graph_iterations}")
            # §20: no global acceptance gate; always continue.
            continue

        geometry_stable = True
        video_frames = list(range(len(camera_groups)))
        for step in tqdm(range(embedding_steps), desc=f"Outer E-step {outer:03d}", unit="epoch", leave=False):
            # §8: rotate frame order between epochs so no subset dominates.
            if bool(getattr(args, "outer_video_shuffle_frames", True)):
                offset = step % max(1, len(video_frames))
                epoch_frames = video_frames[offset:] + video_frames[:offset]
            else:
                epoch_frames = list(video_frames)
            block = int(getattr(args, "outer_video_frame_block", 0))
            if block > 0:
                epoch_frames = epoch_frames[:block]

            optimizer.zero_grad(set_to_none=True)
            p_soft = model.probabilities(temperature, apply_anchor=apply_anchor)
            p_st = straight_through_membership(p_soft)
            label_param = model.logits if model.direct_logits else model.embedding

            # ---- VIDEO: the PRIMARY label loss (§6) --------------------------
            # §8: accumulate each camera's share immediately so the whole
            # F x C video never has to be alive at once.
            camera_rows: list[dict] = []
            video = outer_render_membership_loss(
                xyz0=xyz0, rot0=rot0,
                part_rotation=frozen_R, part_translation=frozen_t, part_quaternion=frozen_q,
                p_st=p_st, cameras_by_frame=camera_groups,
                gaussians=gaussians, pipe=pipe, background=background,
                dataset=dataset, args=args, frames=epoch_frames,
                collect_rows=camera_rows,
                accumulate=lambda share: share.backward(retain_graph=True),
            )
            # The video gradient is now exactly what has accumulated (§16).
            grad_video = (
                label_param.grad.detach().clone()
                if label_param.grad is not None
                else torch.zeros_like(label_param)
            )
            g_video = float(grad_video.norm())
            video_loss = None

            # ---- PIXEL-MEDIATED MATCHING: primary alongside video (§22-§24) ---
            if pixel_evidence is not None:
                pix_cand = pixel_candidate_loss(p_soft, pixel_evidence)
                pix_marg = pixel_margin_loss(
                    p_soft, pixel_evidence,
                    margin=float(getattr(args, "pixel_candidate_margin", 0.05)),
                )
                primary_extra = (
                    float(getattr(args, "lambda_pixel_candidate", 1.0)) * pix_cand
                    + float(getattr(args, "lambda_pixel_margin", 0.1)) * pix_marg
                )
                # §25: the pixel signal is PRIMARY, so it joins the video in the
                # reference gradient rather than being capped with the 3D block.
                grads = torch.autograd.grad(primary_extra, [label_param], retain_graph=True, allow_unused=True)
                g_pixel = 0.0 if grads[0] is None else float(grads[0].norm())
                g_primary = (g_video ** 2 + g_pixel ** 2) ** 0.5
                # §28: weak, normalised coverage so many sources do not collapse
                # onto one final Gaussian. Deliberately NOT one-to-one.
                if pixel_sparse is not None and pixel_target_support is not None:
                    pix_cover = target_coverage_loss(
                        p_soft, pixel_sparse["source"], pixel_sparse["target"],
                        pixel_sparse["weight"], pixel_target_support,
                        num_target=int(final_xyz.shape[0]),
                        q_class=pixel_sparse["klass"],
                    )
                    primary_extra = primary_extra + float(
                        getattr(args, "lambda_pixel_coverage", 0.02)) * pix_cover
                else:
                    pix_cover = p_soft.sum() * 0.0
            else:
                pix_cand = p_soft.sum() * 0.0
                pix_marg = p_soft.sum() * 0.0
                pix_cover = p_soft.sum() * 0.0
                primary_extra = p_soft.sum() * 0.0
                g_pixel = 0.0
                g_primary = g_video

            # ---- local label regularizers (§17, §18) -------------------------
            graph_loss = motion_aware_label_graph_loss(
                p_soft, delta_energy, neighbor_graph.to(device=device),
                sigma_e=float(getattr(args, "outer_label_graph_sigma", 1.0)),
            )
            kin_loss = kinetic_label_loss(p_soft, kinetic_energy)
            # §19: optional static-incompatibility term, inert at alpha_0 = 0 so
            # static-vs-moving stays render-driven unless explicitly enabled.
            alpha_0 = float(getattr(args, "kinetic_alpha_static", 0.0))
            if alpha_0 > 0.0:
                e_static = static_kinetic_energy(
                    motion.inferred_speed.to(device), alpha_0=alpha_0
                )
                kin_loss = kin_loss + (p_soft[:, 0] * e_static).mean()

            # ---- 3D endpoint regularizers, now SUBORDINATE (§13) -------------
            losses, diagnostics = outer_embedding_losses(
                canonical_xyz=xyz0, final_xyz=final_xyz, part_prob=p_soft,
                static_anchor_mask=anchor_mask, prototypes=model.prototype,
                candidates=candidates, targets=targets, previous_prob=p_old,
                motion_separation=motion.motion_separation.to(device),
                bbox_diag=bbox_diag, chunk_size=chunk,
                margin=float(getattr(args, "outer_candidate_margin", 0.001)),
            )
            reg_3d = sum(
                float(weights.get(k, 0.0)) * losses[k]
                for k in ("candidate_nn", "final_nn", "part_mass", "global_com",
                          "global_inertia", "part_com", "part_inertia")
            )

            # §16: keep the 3D gradient subordinate to the video gradient. The
            # video half is already accumulated, so only the (cheap, render-free)
            # 3D half needs an extra autograd pass.
            if step % max(1, int(getattr(args, "outer_3d_cap_interval", 1))) == 0:
                g_3d = 0.0
                if torch.is_tensor(reg_3d) and reg_3d.requires_grad:
                    grads = torch.autograd.grad(reg_3d, [label_param], retain_graph=True, allow_unused=True)
                    g_3d = 0.0 if grads[0] is None else float(grads[0].norm())
                rho = float(getattr(args, "rho_3d_gradient_cap", 0.30))
                scale_value = 1.0 if g_3d <= 1.0e-12 else min(1.0, rho * g_primary / (g_3d + 1.0e-12))
                scale_3d = torch.as_tensor(scale_value, device=device).detach()
                last_cap = (scale_3d, g_video, g_3d)
            else:
                scale_3d, _, g_3d = last_cap

            # §15/§43: the constants are not the rule -- "the video gradient must
            # be the principal label gradient" is. The local regularizers get the
            # same detached subordination as the 3D block, so no hand-tuned
            # per-scene constant can quietly outrank the video.
            local_reg = (
                float(getattr(args, "lambda_outer_label_graph", 0.05)) * graph_loss
                + float(getattr(args, "lambda_outer_kinetic", 0.05)) * kin_loss
            )
            g_local = 0.0
            if torch.is_tensor(local_reg) and local_reg.requires_grad:
                grads = torch.autograd.grad(local_reg, [label_param], retain_graph=True, allow_unused=True)
                g_local = 0.0 if grads[0] is None else float(grads[0].norm())
            rho_local = float(getattr(args, "rho_local_gradient_cap", 0.50))
            scale_local = torch.as_tensor(
                1.0 if g_local <= 1.0e-12 else min(1.0, rho_local * g_primary / (g_local + 1.0e-12)),
                device=device,
            ).detach()

            anchor_loss = losses["static_anchor"]
            trust = losses["trust_region"]
            entropy = losses["entropy"]

            # The video term is already in label_param.grad; everything else is
            # backwarded now and simply accumulates on top of it.
            total = (
                primary_extra
                + scale_local * local_reg
                + scale_3d * reg_3d
                + float(getattr(args, "lambda_static_anchor", 10.0)) * anchor_loss
                + float(getattr(args, "lambda_outer_trust_region", 1.0)) * trust
                + float(getattr(args, "lambda_outer_entropy", 0.001)) * entropy
            )

            state_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            total.backward()
            # §22/§39: only membership parameters may carry a gradient here.
            logit_grad = label_param.grad
            optimizer.step()

            # §24/§42: no acceptance gate. Backtrack only on numerical failure.
            if not (torch.isfinite(total.detach()) and torch.isfinite(torch.as_tensor(video["value"]))):
                model.load_state_dict(state_backup)
                geometry_stable = False
                for group in optimizer.param_groups:
                    group["lr"] = max(group["lr"] * 0.5, 1.0e-6)

            with torch.no_grad():
                current = model.probabilities(temperature, apply_anchor=apply_anchor)
                hard = current.argmax(dim=-1)
                if previous_hard_for_metrics is None:
                    num_new_static = static_to_moving = moving_to_static = moving_to_moving = 0
                else:
                    moving_to_static = int(((previous_hard_for_metrics != 0) & (hard == 0)).sum().cpu())
                    static_to_moving = int(((previous_hard_for_metrics == 0) & (hard != 0)).sum().cpu())
                    num_new_static = moving_to_static
                    moving_to_moving = int((
                        (previous_hard_for_metrics != 0) & (hard != 0)
                        & (previous_hard_for_metrics != hard)
                    ).sum().cpu())
                changed = (
                    torch.zeros_like(hard, dtype=torch.bool)
                    if previous_hard_for_metrics is None
                    else (previous_hard_for_metrics != hard)
                )
                weighted = {
                    "video": float(getattr(args, "lambda_outer_video", 1.0)) * float(video["value"]),
                    "pixel": float(getattr(args, "lambda_pixel_candidate", 1.0)) * float(pix_cand)
                             + float(getattr(args, "lambda_pixel_margin", 0.1)) * float(pix_marg),
                    "graph": float(scale_local) * float(getattr(args, "lambda_outer_label_graph", 0.05)) * float(graph_loss),
                    "kinetic": float(scale_local) * float(getattr(args, "lambda_outer_kinetic", 0.05)) * float(kin_loss),
                    "reg_3d": float(scale_3d) * float(reg_3d),
                    "anchor": float(getattr(args, "lambda_static_anchor", 10.0)) * float(anchor_loss),
                    "trust": float(getattr(args, "lambda_outer_trust_region", 1.0)) * float(trust),
                }
                denom = sum(abs(v) for v in weighted.values()) + 1.0e-12
                grad_norm = 0.0 if logit_grad is None else float(logit_grad.norm())
                row = {
                    "outer": outer, "step": step, "label_epoch": step, "temperature": temperature,
                    "loss_total": float(total.detach().cpu()) + float(getattr(args, "lambda_outer_video", 1.0)) * float(video["value"]),
                    # §32 outer video diagnostics
                    "outer_video_rgb": video["rgb"], "outer_video_ssim": video["ssim"],
                    "outer_video_depth": video["depth"], "outer_video_total": float(video["value"]),
                    "outer_video_num_frames": float(video["num_frames"]),
                    "outer_video_num_views": float(video["num_views"]),
                    "outer_video_grad_norm": g_video, "outer_3d_grad_norm": g_3d,
                    "outer_3d_grad_scale": float(scale_3d),
                    "outer_local_grad_norm": g_local, "outer_local_grad_scale": float(scale_local),
                    # sections 43/44 pixel diagnostics
                    "pixel_candidate_loss": float(pix_cand),
                    "pixel_margin_loss": float(pix_marg),
                    "pixel_coverage_loss": float(pix_cover),
                    "outer_pixel_grad_norm": g_pixel,
                    "outer_primary_grad_norm": g_primary,
                    **({} if pixel_evidence is None else {
                        "mean_pixel_confidence": float(pixel_evidence.confidence.mean()),
                        "mean_pixel_energy_margin": float(
                            (pixel_evidence.energy.topk(2, dim=1, largest=False).values[:, 1]
                             - pixel_evidence.energy.min(dim=1).values).mean()),
                        "num_gaussians_with_multiview_pixel_support": float(
                            (pixel_evidence.num_views.max(dim=1).values > 1).sum()),
                        **{f"mean_pixel_energy_class_{c}": float(pixel_evidence.energy[:, c].mean())
                           for c in range(int(pixel_evidence.energy.shape[1]))},
                        **{f"fraction_pixel_prefers_{c}": float(
                            (pixel_evidence.energy.argmin(dim=1) == c).float().mean())
                           for c in range(int(pixel_evidence.energy.shape[1]))},
                    }),
                    "outer_graph_loss": float(graph_loss), "outer_kinetic_loss": float(kin_loss),
                    "outer_3d_total": float(reg_3d), "outer_anchor_loss": float(anchor_loss),
                    "outer_trust_loss": float(trust),
                    **{f"outer_fraction_{k}": abs(v) / denom for k, v in weighted.items()},
                    **{f"loss_{k}": float(v.detach().cpu()) for k, v in losses.items()},
                    # §34 label transition diagnostics
                    "num_static_anchors": int(anchor_mask.sum().cpu()),
                    "mean_static_probability": float(current[:, 0].mean().cpu()),
                    "mean_label_entropy": float((-(current.clamp_min(1e-8) * current.clamp_min(1e-8).log()).sum(dim=1)).mean().cpu()),
                    "mean_label_confidence": float(current.max(dim=-1).values.mean().cpu()),
                    "num_new_static": num_new_static, "static_to_moving": static_to_moving,
                    "moving_to_static": moving_to_static, "moving_to_moving": moving_to_moving,
                    "num_changed_points": int(changed.sum().cpu()),
                    "mean_probability_changed_points": float(current[changed].max(dim=-1).values.mean().cpu()) if bool(changed.any()) else 0.0,
                    "mean_video_gradient_changed_points": float(logit_grad[changed].norm(dim=-1).mean().cpu()) if (logit_grad is not None and bool(changed.any())) else 0.0,
                    "label_logit_grad_norm": grad_norm,
                    "candidate_nn": float(losses["candidate_nn"].detach().cpu()),
                    "final_symmetric_nn": float(losses["final_nn"].detach().cpu()),
                    "global_com_error": float(diagnostics["global_com_error"].cpu()),
                    "global_inertia_error": float(diagnostics["global_inertia_error"].cpu()),
                    "candidate_separation": separation,
                    "outer_step_accepted": True,
                    **{f"inner_{k}": v for k, v in inner_metrics.items()},
                }
                moment_fraction = row["outer_fraction_reg_3d"]
                row["moment_weight_fraction"] = float(moment_fraction)
                row["video_is_primary"] = bool(row["outer_fraction_video"] >= max(
                    row["outer_fraction_reg_3d"], row["outer_fraction_graph"], row["outer_fraction_kinetic"]))
                for part, value in enumerate(diagnostics["part_fraction"].tolist()):
                    row[f"part_fraction_{part}"] = float(value)
                for part, value in enumerate(diagnostics["part_com_error"].tolist()):
                    row[f"part_{part}_com_error"] = float(value)
                for part, value in enumerate(diagnostics["part_inertia_error"].tolist()):
                    row[f"part_{part}_inertia_error"] = float(value)
                for part, value in enumerate(diagnostics["part_mass_error"].tolist()):
                    row[f"part_{part}_mass_error"] = float(value)
                for part in range(current.shape[1]):
                    row[f"num_part_{part}"] = int((hard == part).sum().cpu())
                outer_rows.append(row)
                for cam_row in camera_rows:
                    video_rows.append({
                        "outer": outer, "label_epoch": step, **cam_row,
                        **{f"num_part_{c}": int((hard == c).sum().cpu()) for c in range(current.shape[1])},
                    })
                if log_fn is not None:
                    log_fn("nested_outer", outer * embedding_steps + step,
                           {f"nested_outer/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
                # §34: which Gaussians are actually being driven, and why.
                if logit_grad is not None and step == embedding_steps - 1:
                    top_n = int(getattr(args, "outer_top_gradient_points", 256))
                    order = logit_grad.norm(dim=-1).argsort(descending=True)[:top_n]
                    top_gradient_records.append({
                        "outer": outer,
                        "gaussian_id": order.detach().cpu(),
                        "canonical_xyz": xyz0[order].detach().cpu(),
                        "old_probability": p_old[order].detach().cpu(),
                        "new_probability": current[order].detach().cpu(),
                        "render_gradient_norm": logit_grad[order].norm(dim=-1).detach().cpu(),
                        "kinetic_energy": kinetic_energy[order].detach().cpu(),
                        "candidate_nn": cand_nn[order].detach().cpu(),
                    })
                previous_hard_for_metrics = hard.detach().clone()

        # §50: per-camera visual debug for selected frames of this outer.
        if int(getattr(args, "outer_video_debug_every", 0)) > 0:
            every = int(getattr(args, "outer_video_debug_every", 0))
            for frame in range(0, len(camera_groups), every):
                if not camera_groups[frame]:
                    continue
                with torch.no_grad():
                    cand_xyz, cand_quat = candidate_state_at_frame(
                        xyz0, rot0, frozen_R, frozen_t, frozen_q, frame
                    )
                    p_now = model.probabilities(temperature, apply_anchor=apply_anchor)
                    xyz_dbg, quat_dbg = compose_straight_through_scene(
                        cand_xyz, cand_quat, straight_through_membership(p_now)
                    )
                save_multiview_debug_images(
                    cameras=camera_groups[frame], gaussians=gaussians, pipe=pipe,
                    background=background, dataset=dataset, args=args,
                    positions=xyz_dbg.detach(), rotations=quat_dbg.detach(),
                    out_dir=result_dir / "outer_video_debug" / f"outer_{outer:03d}"
                            / f"label_epoch_{embedding_steps - 1:03d}" / f"frame_{frame:03d}",
                )


        latest_prob = model.probabilities(temperature, apply_anchor=apply_anchor).detach()
        with torch.no_grad():
            churn = float((latest_prob.argmax(dim=-1) != hard_before_outer).float().mean())
        # §23: only now may tau move, and only if every condition holds.
        outer_transitions = sum(
            int(r["static_to_moving"]) + int(r["moving_to_static"])
            for r in outer_rows if r.get("outer") == outer
        )
        temperature, blockers = adaptive_temperature(
            temperature,
            motion_accepted=True,  # no global gate (§31/§32)
            transitions=outer_transitions,
            previous_transitions=previous_outer_transitions,
            label_churn_fraction=churn,
            candidate_separation=separation,
            geometry_stable=geometry_stable,
            soft_floor=float(getattr(args, "part_temperature_soft_floor", 0.7)),
            hard_floor=float(getattr(args, "part_temperature_hard_floor", 0.4)),
            decay=float(getattr(args, "part_temperature_decay", 0.85)),
            churn_tolerance=float(getattr(args, "part_temperature_churn_tolerance", 0.01)),
        )
        previous_outer_transitions = outer_transitions
        if outer_rows:
            outer_rows[-1]["outer_transitions"] = outer_transitions
            outer_rows[-1]["temperature_blockers"] = ";".join(blockers)
            outer_rows[-1]["label_churn_fraction"] = churn
        outer_progress.set_postfix(
            executed=f"{outer_executed}/{outer + 1}",
            tau=f"{temperature:.2f}",
            static=f"{float(latest_prob[:, 0].mean().cpu()):.3f}",
            churn=f"{churn:.3f}",
        )

    assert latest_motion is not None and latest_prob is not None
    labels = latest_prob.argmax(dim=-1)
    tracks = _track_payload(xyz0, rot0, latest_motion, latest_prob, labels)
    summary = {
        "schema_version": 4,
        "algorithm": "nested_part_embedding_local_velocity_field_analytic_partwise_twist",
        "forbidden_legacy_kpart_used": False,
        "dynamic_prior_used_for_initialization": False,
        "kabsch_used_in_training": False,
        "per_gaussian_twist_optimized": False,
        "num_static_anchors": int(anchor_mask.sum().detach().cpu()),
        "static_anchor_distance_threshold": float(anchors.threshold.cpu()),
        "static_anchor_distance_ratio": float(getattr(args, "static_anchor_distance_ratio", 0.002)),
        "outer_iters": outer_iters,
        "outer_embedding_steps": embedding_steps,
        "inner_steps_per_frame": int(getattr(args, "killing_inner_steps_per_frame", 10)),
        "num_parts": int(latest_prob.shape[1]),
        "class_counts": [int((labels == part).sum().detach().cpu()) for part in range(int(latest_prob.shape[1]))],
        "mean_label_confidence": float(latest_prob.max(-1).values.mean().detach().cpu()),
        "mean_static_probability": float(latest_prob[:, 0].mean().detach().cpu()),
        "frame_ids": frame_ids,
        "frame_times": frame_times.tolist(),
        "final_3dgs_path": str(final_ply),
        "ground_truth_used_as_algorithm_input": False,
        "inner_field_type": "local_gaussian_velocity_field_with_analytic_partwise_twist",
        "embedding_motion_loss_is_placeholder": False,
        "motion_separation": latest_motion.motion_separation.detach().cpu(),
        "final_temperature": float(temperature),
        "outer_iterations_executed": int(outer_executed),
        "outer_iterations_total": int(outer_iters),
        "inner_acceptance": latest_motion.acceptance,
        "global_hard_twist_imposed": False,
        "per_gaussian_se3_transport": False,
        "partwise_twist_fitted_analytically": True,
        "anchor_hard_freeze_outer_iters": int(anchor_warmup),
        "initialization": init_diag,
        "hard_partition_inner_scene": True,
        "global_render_gate": False,
        "global_endpoint_gate": False,
    }
    loss_contract = {
        "schema_version": 4,
        "architecture": "nested_part_embedding_outer_loop__local_velocity_field_analytic_partwise_twist_inner_loop",
        "initialization": {
            "static_anchors": "initial_final_pcd_reciprocal_nn_only",
            "non_anchor_embeddings": "random",
            "dynamic_prior_used": False,
            "legacy_kpart_used": False,
            "kabsch_used": False,
            "per_gaussian_twist_optimized": False,
        },
        "inner_solver": {
            "field_type": "local_gaussian_velocity_field_with_analytic_partwise_twist",
            "trainable_quantity": "per_gaussian_velocity_field",
            "twist_estimator": "analytic_weighted_least_squares_H_omega_eq_g",
            "per_gaussian_se3_integration": False,
            "partwise_se3_integration": True,
            "transform_increments_per_frame": "one_per_moving_part",
            "kabsch_used": False,
            "partwise_twist_reg": float(getattr(args, "partwise_twist_reg", 1.0e-8)),
            "global_hard_twist_imposed": False,
            "jacobian_mode": str(getattr(args, "killing_jacobian_mode", "analytic")),
            "neighbor_k": int(getattr(args, "killing_neighbor_k", 16)),
            "observed_velocity_gate": {
                "c_min": float(getattr(args, "observed_velocity_c_min", 0.60)),
                "gamma": float(getattr(args, "observed_velocity_gamma", 2.0)),
                "rho_obs_max": float(getattr(args, "observed_velocity_rho_max", 0.20)),
                "min_high_confidence": int(getattr(args, "observed_velocity_min_high_conf", 16)),
            },
            "control_points_per_part": int(getattr(args, "nested_killing_control_points", 64)),
            "kernel_scale": float(getattr(args, "nested_killing_kernel_scale", 0.0)),
            "exact_se3_integration": True,
            "gaussian_quaternion_transport": True,
            "loss_weights": {
                "rgb": float(getattr(args, "lambda_inner_rgb", 1.0)),
                "ssim": float(getattr(args, "lambda_inner_ssim", 0.0)),
                "depth": float(getattr(args, "lambda_inner_depth", 0.1)),
                "observed_velocity": float(getattr(args, "lambda_inner_observed_velocity", 0.0)),
                "strain": float(getattr(args, "lambda_inner_strain", 10.0)),
                "rigid_speed": float(getattr(args, "lambda_inner_rigid_speed", 1.0)),
                "omega_local_coherence": float(getattr(args, "lambda_inner_omega_consistency", 1.0)),
                "b_local_coherence": float(getattr(args, "lambda_inner_b_consistency", 1.0)),
                "twist_temporal": float(getattr(args, "lambda_inner_twist_temporal", 0.01)),
                "twist_acceleration": float(getattr(args, "lambda_inner_twist_acceleration", 0.0)),
                "field_separation": float(getattr(args, "lambda_inner_field_separation", 0.0)),
            },
        },
        "outer_solver": {
            "loss_weights": {
                "candidate_nn": float(getattr(args, "lambda_outer_candidate_nn", 1.0)),
                "final_nn": float(getattr(args, "lambda_outer_final_nn", 1.0)),
                "global_com": float(getattr(args, "lambda_outer_global_com", 1.0)),
                "global_inertia": float(getattr(args, "lambda_outer_global_inertia", 1.0)),
                "part_mass": float(getattr(args, "lambda_outer_part_mass", 2.0)),
                "part_com": float(getattr(args, "lambda_outer_part_com", 2.0)),
                "part_inertia": float(getattr(args, "lambda_outer_part_inertia", 2.0)),
                "trust_region": float(getattr(args, "lambda_outer_trust_region", 1.0)),
                "static_anchor": float(getattr(args, "lambda_static_anchor", 10.0)),
                "embedding_motion": float(getattr(args, "lambda_outer_embedding_motion", 0.0)),
                "candidate_margin": float(getattr(args, "lambda_outer_candidate_margin", 0.1)),
                "entropy": float(getattr(args, "lambda_outer_entropy", 0.001)),
                "occupancy": float(getattr(args, "lambda_outer_occupancy", 0.01)),
            },
            "fixed_target_per_outer_iteration": True,
            "trust_region_kl": True,
            "geometry_guard": {
                "terms": ["candidate_nn", "final_nn", "global_com", "part_com", "part_mass"],
                "epsilon": float(getattr(args, "outer_geometry_guard_eps", 1.0e-6)),
                "max_backtracks": int(getattr(args, "outer_geometry_guard_backtracks", 4)),
            },
        },
        "temperature": {
            "start": float(getattr(args, "part_temperature", 1.0)),
            "soft_floor": float(getattr(args, "part_temperature_soft_floor", 0.7)),
            "hard_floor": float(getattr(args, "part_temperature_hard_floor", 0.4)),
            "iteration_count_anneal": False,
        },
        "debug_outputs": [
            "outer_XXXX/inner_history.csv",
            "outer_XXXX/inner_history.json",
            "outer_XXXX/local_killing_field_debug.pt",
            "outer_XXXX/motion_separation.json",
            "outer_history.csv",
            "outer_history.json",
            "inner_history.csv",
            "inner_history.json",
            "nested_killing_tracks.pt",
            "nested_killing_motion.html",
        ],
    }
    _atomic_torch_save({"schema_version": 1, "mask": anchor_mask.detach().cpu(), "target_index": anchors.target_index, "distance": anchors.distance, "summary": summary}, result_dir / "static_anchors.pt")
    _atomic_torch_save({"summary": summary, "outer_history": outer_rows, "inner_history": inner_rows}, result_dir / "nested_state.pt")
    _write_csv(result_dir / "outer_video_history.csv", video_rows)          # §33
    _write_csv(result_dir / "render_graph_history.csv", render_graph_rows)  # §22
    if temporal_cycle is not None:
        # Temporal credit §35: one row per SOLVED outer state. Non-scalar fields
        # (the part permutation) go to the json/pt artifacts, not the csv.
        _write_csv(result_dir / "temporal_outer_history.csv",
                   [{k: v for k, v in row.items() if not isinstance(v, (list, tuple, dict))}
                    for row in temporal_rows])
        _atomic_torch_save({
            "schema_version": 1,
            "memory": temporal_cycle.memory.detach().cpu(),
            "normalization": (temporal_cycle.normalization.as_dict()
                              if temporal_cycle.normalization else None),
            "history": temporal_rows,
            "hyperparameters": {
                "lambda_score_render": float(getattr(args, "lambda_score_render", 1.0)),
                "lambda_score_match": float(getattr(args, "lambda_score_match", 1.0)),
                "lambda_score_final_nn": float(getattr(args, "lambda_score_final_nn", 0.1)),
                "temporal_credit_graph_k": int(temporal_graph.k),
                "temporal_global_reward_weight": temporal_cycle.beta_global,
                "temporal_component_reward_weight": temporal_cycle.beta_component,
                "temporal_reward_clip": temporal_cycle.reward_clip,
                "temporal_memory_decay": temporal_cycle.memory_decay,
                "temporal_memory_global_decay": temporal_cycle.memory_global_decay,
                "lambda_temporal_memory": temporal_cycle.lambda_memory,
                "temporal_memory_margin_cap": temporal_cycle.memory_margin_cap,
                "temporal_score_frame_stride": int(getattr(args, "temporal_score_frame_stride", 1)),
            },
        }, result_dir / "temporal_credit_memory.pt")
    if top_gradient_records:                                                # §34
        _atomic_torch_save({"schema_version": 1, "records": top_gradient_records},
                           result_dir / "top_N_gaussians_by_label_gradient.pt")
    _atomic_torch_save(tracks, result_dir / "nested_killing_tracks.pt")
    _atomic_torch_save({"probabilities": latest_prob.detach().cpu(), "labels": labels.detach().cpu(), "confidence": latest_prob.max(-1).values.detach().cpu()}, result_dir / "part_embeddings_final.pt")
    _write_csv(result_dir / "outer_history.csv", outer_rows)
    _write_csv(result_dir / "inner_history.csv", inner_rows)
    (result_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, allow_nan=False), encoding="utf-8")
    # CODEX_DEBUG §12: how polarization evolved across the whole outer loop.
    (result_dir / "outer_polarization.json").write_text(
        json.dumps(_jsonable({"rows": polarization_rows}), indent=2), encoding="utf-8")
    (result_dir / "implementation_contract.json").write_text(json.dumps(_jsonable(loss_contract), indent=2, allow_nan=False), encoding="utf-8")
    (result_dir / "outer_history.json").write_text(json.dumps(_jsonable(outer_rows), indent=2, allow_nan=False), encoding="utf-8")
    (result_dir / "inner_history.json").write_text(json.dumps(_jsonable(inner_rows), indent=2, allow_nan=False), encoding="utf-8")
    try:
        write_transported_gaussians_html(
            tracks,
            result_dir / "nested_killing_motion.html",
            title="Nested part-embedding differential Killing motion",
            max_points=int(getattr(args, "transported_html_max_points", 5000)),
            max_vectors=min(250, int(getattr(args, "transported_html_max_points", 5000))),
        )
    except Exception as exc:
        print(f"[Nested Killing] HTML debug skipped: {exc}")

    if getattr(gaussians, "material_assignment", None) is not None:
        with torch.no_grad():
            gaussians.material_assignment.logits.copy_(torch.log(latest_prob.to(gaussians.material_assignment.logits).clamp_min(1.0e-8)))
    return {
        "summary": summary,
        "responsibilities": latest_prob.detach().cpu(),
        "labels": labels.detach().cpu(),
        "coarse_tracks": tracks,
        "rigid_tracks": tracks,
        "raw_tracks": tracks,
        "frame_times": frame_times,
        "frame_ids": frame_ids,
        "stage2_legacy_modes_are_stale": True,
    }
