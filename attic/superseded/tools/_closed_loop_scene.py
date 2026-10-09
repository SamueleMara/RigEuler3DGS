"""Bridge between the closed loop and the existing 3DGS scene machinery.

Everything the loop needs from the repository comes through here: the canonical
model, the dynamic cameras, the foreground domain, the rasterized multiview
velocity observations, and the renderer callbacks the residual critic is
supervised by.  Keeping the bridge in one file means the loop itself never
touches a ``Scene`` or a rasterizer, so it stays testable on synthetic inputs.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.closed_loop_motion_state import extract_foreground  # noqa: E402


def _find_model_ply(root: str | Path, preferred_iteration: int | None) -> Path | None:
    """Resolve a 3DGS model directory or direct PLY path to one PLY."""
    path = Path(root).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".ply":
        return path
    if preferred_iteration:
        candidate = path / "point_cloud" / f"iteration_{int(preferred_iteration)}" / "point_cloud.ply"
        if candidate.is_file():
            return candidate
    point_root = path / "point_cloud"
    if not point_root.is_dir():
        return None
    found: list[tuple[int, Path]] = []
    for child in point_root.glob("iteration_*"):
        try:
            iteration = int(child.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        ply = child / "point_cloud.ply"
        if ply.is_file():
            found.append((iteration, ply))
    return max(found, key=lambda item: item[0])[1] if found else None


def _endpoint_candidates(args) -> list[Path]:
    """Common final-frame model locations produced by train_all_static_endpoints.py."""
    explicit = str(getattr(args, "endpoint_static_model_dir", "") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        candidates = [path]
        manifest = path.parent / "static_endpoints_manifest.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if data.get("frame_t_static_dir"):
                    candidates.append(Path(data["frame_t_static_dir"]).expanduser().resolve())
            except Exception:
                pass
        candidates.extend([path.parent / "frame_t_static", path.parent / "endpoint_static"])
        seen = set()
        unique = []
        for item in candidates:
            key = str(item)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique
    frame0 = Path(args.static_model_dir).expanduser().resolve()
    parent = frame0.parent
    candidates = [
        parent / "frame_t_static",
        parent / "endpoint_static",
    ]
    if parent.name != "shared":
        candidates += [
            parent / "shared" / "frame_t_static",
            parent / "shared" / "endpoint_static",
        ]
    return candidates


def _endpoint_transport_init(args, gaussians, foreground, frame_times) -> dict | None:
    """Initialise FM from frame-0 -> frame-T static 3DGS transport."""
    if bool(getattr(args, "no_endpoint_fm", False)):
        return None

    from scene.gaussian_model import GaussianModel
    from utils.endpoint_transport_prior import EndpointTransportConfig, EndpointTransportPrior
    from utils.endpoint_flow_matching import (
        FlowMatchingConfig,
        build_dynamic_prior_from_flow,
        flow_checkpoint_payload,
        integrate_flow,
        train_endpoint_flow_matching,
    )

    fm_init = str(getattr(args, "fm_init", "legacy")).lower()
    if fm_init not in {"legacy", "current_knn"}:
        raise ValueError("--fm-init must be 'legacy' or 'current_knn'")

    iteration = int(getattr(args, "endpoint_static_iteration", 0) or getattr(args, "static_iteration", 0))
    endpoint_ply = None
    endpoint_root = None
    for candidate in _endpoint_candidates(args):
        ply = _find_model_ply(candidate, iteration)
        if ply is not None:
            endpoint_root = candidate
            endpoint_ply = ply
            break
    if endpoint_ply is None:
        if fm_init == "legacy":
            raise RuntimeError(
                "--fm-init legacy requires a final-frame static 3DGS; pass "
                "--endpoint-static-model-dir or use --fm-init current_knn explicitly"
            )
        print("[closed loop] endpoint FM disabled: no final-frame static 3DGS found")
        return None

    endpoint = GaussianModel(int(gaussians.max_sh_degree), optimizer_type="default")
    endpoint.load_ply(str(endpoint_ply))
    target_xyz = endpoint.get_canonical_xyz.detach()
    # `get_semantic_probability` is a foreground/background (object-vs-empty-space)
    # classifier -- see GaussianModel.build_dynamic_foreground_mask, "the strict
    # semantic-foreground contract" -- not a static-vs-moved discriminator. On an
    # isolated-object capture it is ~1.0 for essentially every reconstructed
    # Gaussian (measured: min 0.97, 100% > 0.5 on USB_100109), so thresholding it
    # here used to silently keep the entire static+dynamic-mixed point cloud
    # while claiming to have restricted it to "the dynamic part". Matching every
    # source point -- static and dynamic alike -- against that unrestricted cloud
    # is exactly why nearest-neighbour correspondence collapsed onto nearby
    # static structure (see the collision stats this produced: unique-target
    # ratios of 0.08-0.18 with individual targets absorbing 60-270 matches).
    # There is no reliable frame-T-side dynamic label available at this point in
    # the pipeline, so we no longer pretend to filter on one; protection against
    # matching static structure now comes from `fm_guided_correspondence_em`'s
    # isometry-consistency term (t0-vs-T neighbour distance preservation).
    print(
        "[closed loop] endpoint target cloud: "
        f"matching against all {int(target_xyz.shape[0])} endpoint Gaussians "
        "(no static/dynamic pre-filter; see correspondence isometry weighting)"
    )

    source_idx = foreground["foreground_idx"].to(device=gaussians.get_canonical_xyz.device)
    source_xyz = foreground["canonical_fg"]
    source_scale = None
    try:
        source_scale = gaussians.get_scaling.detach().amax(dim=1).index_select(0, source_idx).detach().cpu()
    except Exception:
        source_scale = None

    cfg = EndpointTransportConfig(
        knn=int(getattr(args, "endpoint_transport_knn", 16)),
        temperature=float(getattr(args, "endpoint_transport_temperature", 0.02)),
        robust_delta=float(getattr(args, "endpoint_transport_robust_delta", 0.02)),
        motion_scale_mode=str(getattr(args, "endpoint_motion_scale", "local_knn")),
        confidence_distance_scale=float(getattr(args, "endpoint_transport_confidence_distance_scale", 0.0)),
        confidence_entropy_scale=float(getattr(args, "endpoint_transport_confidence_entropy_scale", 1.0)),
        chunk_size=int(getattr(args, "endpoint_transport_chunk_size", 1024)),
        prior_prob_bias=float(getattr(args, "fm_prior_prob_bias", 1.0)),
        prior_prob_temperature=float(getattr(args, "fm_prior_prob_temperature", 0.35)),
    )
    result = EndpointTransportPrior(
        target_xyz,
        {"source_scale": source_scale} if source_scale is not None else {},
        cfg,
    ).compute(source_xyz, foreground["foreground_idx"])

    elapsed = max(float(frame_times[-1]) - float(frame_times[0]), 1.0e-8)
    if fm_init == "legacy":
        fm_cfg = FlowMatchingConfig(
            hidden_dim=int(getattr(args, "legacy_fm_hidden_dim", getattr(args, "fm_hidden_dim", 128))),
            layers=int(getattr(args, "legacy_fm_layers", getattr(args, "fm_layers", 4))),
            lr=float(getattr(args, "legacy_fm_lr", getattr(args, "fm_learning_rate", 1.0e-3))),
            iters=int(getattr(args, "legacy_fm_iters", 1000)),
            batch_size=int(getattr(args, "legacy_fm_batch_size", 2048)),
            identity_weight=float(getattr(args, "legacy_fm_identity_weight", 0.01)),
            smooth_weight=float(getattr(args, "legacy_fm_smooth_weight", 0.0)),
            integration_steps=int(getattr(args, "legacy_fm_integration_steps", 16)),
            integrator=str(getattr(args, "legacy_fm_integrator", "euler")),
            prior_prob_bias=float(getattr(args, "fm_prior_prob_bias", 1.0)),
            prior_prob_temperature=float(getattr(args, "fm_prior_prob_temperature", 0.35)),
            seed=int(getattr(args, "seed", 20260904)),
        )
        cache_dir = None
        if getattr(args, "output_dir", ""):
            cache_dir = Path(args.output_dir).expanduser().resolve() / "fm_legacy_init"
            cache_dir.mkdir(parents=True, exist_ok=True)
        cfg_dict = {
            **dataclasses.asdict(fm_cfg),
            "endpoint_ply": str(endpoint_ply),
            "source_count": int(source_xyz.shape[0]),
            "target_count": int(target_xyz.shape[0]),
            "transport_knn": int(cfg.knn),
            "transport_temperature": float(cfg.temperature),
            "transport_robust_delta": float(cfg.robust_delta),
        }
        integrated = None
        if cache_dir is not None:
            cfg_path = cache_dir / "fm_config.json"
            integrated_path = cache_dir / "integrated_endpoint.pt"
            if (
                not bool(getattr(args, "force_legacy_fm", False))
                and cfg_path.is_file()
                and integrated_path.is_file()
                and json.loads(cfg_path.read_text()) == cfg_dict
            ):
                integrated = torch.load(integrated_path, map_location="cpu", weights_only=False)
                print("[closed loop] legacy endpoint FlowMLP cached — skipping FM training")
        if integrated is None:
            model, logs = train_endpoint_flow_matching(
                source_xyz.detach().cpu(),
                target_xyz.detach().cpu(),
                result,
                fm_cfg,
                device=source_xyz.device,
            )
            integrated = integrate_flow(model, source_xyz.detach().cpu(), fm_cfg)
            if cache_dir is not None:
                torch.save(integrated, cache_dir / "integrated_endpoint.pt")
                torch.save(integrated - source_xyz.detach().cpu(), cache_dir / "displacement.pt")
                torch.save(flow_checkpoint_payload(model.cpu(), fm_cfg, logs), cache_dir / "fm_checkpoint.pt")
                (cache_dir / "training_log.json").write_text(
                    json.dumps(logs, indent=2, allow_nan=False), encoding="utf-8"
                )
                (cache_dir / "fm_config.json").write_text(
                    json.dumps(cfg_dict, indent=2, allow_nan=False), encoding="utf-8"
                )
        prior = build_dynamic_prior_from_flow(
            source_xyz.detach().cpu(),
            integrated,
            foreground["foreground_idx"].detach().cpu(),
            fm_cfg,
            confidence=result.prior.confidence,
            valid=result.prior.valid,
        )
        prior.method = "legacy_flow_matching"
        displacement = (
            torch.as_tensor(integrated).to(device=source_xyz.device, dtype=source_xyz.dtype)
            - source_xyz
        )
        probability = prior.probability.to(device=source_xyz.device, dtype=source_xyz.dtype)
        confidence = prior.confidence.to(device=source_xyz.device, dtype=source_xyz.dtype)
        valid = prior.valid.to(device=source_xyz.device)
        method = "legacy_flow_matching"
    else:
        displacement = result.displacement.to(device=source_xyz.device, dtype=source_xyz.dtype)
        probability = result.prior.probability.to(device=source_xyz.device, dtype=source_xyz.dtype)
        confidence = result.prior.confidence.to(device=source_xyz.device, dtype=source_xyz.dtype)
        valid = result.prior.valid.to(device=source_xyz.device)
        method = "current_knn_endpoint_transport"
    velocity = displacement[None] / elapsed
    velocity = velocity.expand(max(1, len(frame_times) - 1), -1, -1).contiguous()
    confidence_t = confidence[None].expand(velocity.shape[0], -1).contiguous()
    summary = {
        "enabled": True,
        "fm_init": fm_init,
        "method": method,
        "endpoint_model_dir": str(endpoint_root),
        "endpoint_ply": str(endpoint_ply),
        "endpoint_iteration": iteration,
        "source_foreground": int(source_xyz.shape[0]),
        "target_gaussians": int(target_xyz.shape[0]),
        "elapsed_seconds": elapsed,
        "mean_displacement_norm": float(displacement.norm(dim=-1).mean()),
        "mean_confidence": float(confidence.mean()),
        "mean_dynamic_probability": float(probability.mean()),
        "ground_truth_used_as_algorithm_input": False,
    }
    print(
        "[closed loop] endpoint FM initialised from "
        f"{endpoint_ply}: mean |d|={summary['mean_displacement_norm']:.5g}, "
        f"mean confidence={summary['mean_confidence']:.4f}"
    )
    return {
        "velocity": velocity,
        "confidence": confidence_t,
        "displacement": displacement,
        "probability": probability,
        "valid": valid,
        "endpoint_xyz": source_xyz + displacement,
        "target_xyz": target_xyz.detach().to(device=source_xyz.device, dtype=source_xyz.dtype),
        "barycentric_target": result.barycentric_target.detach().cpu(),
        "candidate_matches": result.target_indices.detach().cpu() if result.target_indices is not None else None,
        "candidate_weights": result.transport_weights.detach().cpu() if result.transport_weights is not None else None,
        "candidate_costs": (
            -torch.log(result.transport_weights.clamp_min(1.0e-12)).detach().cpu()
            if result.transport_weights is not None else None
        ),
        "summary": summary,
    }


def _correspondence_args(args, root_scene: Path) -> SimpleNamespace:
    """Settings the existing correspondence cache expects.

    The flow resolution is exposed rather than fixed: the audit of the previous
    pipeline found the 256-pixel default smears the moving/static boundary on
    640x480 imagery, and this loop is more sensitive to that than the old one
    because the boundary is exactly where its assignment energies disagree.
    """
    return SimpleNamespace(
        use_motion_correspondence=True,
        dynamic_flow_dir=str(getattr(args, "flow_dir", "") or ""),
        dynamic_sequence_root=str(root_scene),
        correspondence_flow_max_side=int(getattr(args, "correspondence_flow_max_side", 512)),
        correspondence_cycle_threshold=float(getattr(args, "correspondence_cycle_threshold", 1.5)),
        correspondence_dino_min_cosine=float(getattr(args, "correspondence_dino_min_cosine", 0.2)),
        correspondence_min_confidence=float(getattr(args, "correspondence_min_confidence", 0.05)),
        correspondence_alpha_threshold=float(getattr(args, "correspondence_alpha_threshold", 0.05)),
        observed_flow_velocity_ridge=1.0e-4,
        observed_flow_velocity_min_observations=int(
            getattr(args, "observed_flow_velocity_min_observations", 6)
        ),
        observed_flow_velocity_min_cameras=int(
            getattr(args, "observed_flow_velocity_min_cameras", 3)
        ),
        observed_flow_velocity_min_condition=1.0e-4,
        observed_flow_velocity_max_reprojection_px=float(
            getattr(args, "observed_flow_velocity_max_reprojection_px", 0.75)
        ),
        observed_flow_require_static_improvement=False,
        observed_flow_static_improvement_abs=0.05,
        observed_flow_static_improvement_rel=0.15,
        use_dino_feature_field=False,
        fps=float(getattr(args, "fps", 30.0)),
        time_scale=1.0,
        motion_time_mode="seconds",
    )


def load_closed_loop_inputs(args) -> dict:
    """Load the scene and produce every foreground-only input the loop needs."""
    from tools.test_render_denoise_discovery import (
        _ordered_complete_frame_groups,
        build_scene,
    )
    from utils.motion_correspondence import MotionCorrespondenceCache

    scene_args = SimpleNamespace(
        static_model_dir=args.static_model_dir,
        root_scene=args.root_scene,
        iteration=args.static_iteration,
        sh_degree=None,
        resolution=None,
        white_background=None,
        camera_data_device="cpu",
        use_depth=True,
        cameras="",
        fps=float(getattr(args, "fps", 30.0)),
    )
    scene, gaussians, pipe, white_background = build_scene(scene_args)
    device = gaussians.get_canonical_xyz.device

    groups = _ordered_complete_frame_groups(scene.getTrainCameras())
    if len(groups) < 2:
        raise RuntimeError("the closed loop needs at least two complete multi-view frames")
    available = [frame for frame, _ in groups]
    from tools.train_closed_loop_motion import parse_frame_spec

    selected = parse_frame_spec(getattr(args, "frames", "all"), available)
    lookup = dict(groups)
    frame_ids = [frame for frame in selected]
    frame_times = [float(frame) / float(getattr(args, "fps", 30.0)) for frame in frame_ids]
    camera_groups = [lookup[frame] for frame in frame_ids]

    foreground = extract_foreground(gaussians, args)
    canonical_fg = foreground["canonical_fg"]
    print(
        f"[closed loop] foreground {canonical_fg.shape[0]} of "
        f"{foreground['total_gaussians']} Gaussians"
    )

    # --- rasterized multiview velocity, foreground rows only -----------
    # The cache takes (dataset, dyn_args) in that order and reads
    # `dataset.source_path` when no explicit dynamic root is given.  Supplying
    # both keeps it from having to infer either.
    correspondence_args = _correspondence_args(args, Path(args.root_scene))
    correspondence_dataset = SimpleNamespace(
        source_path=str(Path(args.root_scene).expanduser().resolve()),
        white_background=white_background,
        train_test_exp=False,
    )
    cache = MotionCorrespondenceCache(correspondence_dataset, correspondence_args)
    if not cache.enabled:
        raise RuntimeError("the closed loop needs rasterized optical-flow observations")
    intervals = len(frame_ids) - 1
    n = int(canonical_fg.shape[0])
    velocity = torch.zeros((intervals, n, 3), device=device)
    confidence = torch.zeros((intervals, n), device=device)
    normal = torch.eye(3, device=device).expand(intervals, n, 3, 3).clone()
    valid = torch.zeros((intervals, n), device=device, dtype=torch.bool)

    full_xyz = gaussians.get_canonical_xyz.detach()
    index = foreground["foreground_idx"]
    for interval in range(intervals):
        state = cache.observed_velocity_targets(
            full_xyz,
            list(camera_groups[interval]),
            list(camera_groups[interval + 1]),
            frame_times[interval],
            frame_times[interval + 1],
            gaussians=gaussians,
            pipe=pipe,
        )
        # The cache reports per-Gaussian rows over the full cloud; the loop only
        # ever indexes the foreground, so the restriction happens once, here.
        velocity[interval] = state["velocity"][index]
        confidence[interval] = state["confidence"][index]
        normal[interval] = state["normal_matrix"][index]
        valid[interval] = state["valid"][index]
        print(
            f"  interval {frame_ids[interval]}->{frame_ids[interval + 1]}: "
            f"{int(valid[interval].sum())}/{n} valid observations"
        )

    endpoint = _endpoint_transport_init(args, gaussians, foreground, frame_times)
    if endpoint is None:
        print("[closed loop] FM base falls back to rasterized multiview velocity observations")
        fm_base_velocity = velocity.clone()
        fm_base_confidence = confidence.clone()
        speed = (confidence * velocity.norm(dim=-1)).sum(0) / confidence.sum(0).clamp_min(1.0e-8)
        scale = speed[speed > 0].median() if bool((speed > 0).any()) else torch.tensor(1.0, device=device)
        fm_base_probability = (speed / scale.clamp_min(1.0e-8)).clamp(0.0, 1.0)
        fm_base_displacement = (velocity * torch.diff(torch.tensor(frame_times, device=device, dtype=velocity.dtype))[:, None, None]).sum(0)
        fm_base_valid = confidence.sum(0) > 0
        endpoint_metadata = {"endpoint_fm_enabled": False}
    else:
        fm_base_velocity = endpoint["velocity"]
        fm_base_confidence = endpoint["confidence"]
        fm_base_probability = endpoint["probability"]
        fm_base_displacement = endpoint["displacement"]
        fm_base_valid = endpoint["valid"]
        endpoint_metadata = {"endpoint_fm_enabled": True, **endpoint["summary"]}

    render_evidence_fn, render_loss_fn, observation_confidence_fn, render_transport_feedback_fn = _build_render_callbacks(
        scene=scene, gaussians=gaussians, pipe=pipe, foreground=foreground,
        camera_groups=camera_groups, white_background=white_background,
        confidence=confidence, valid=valid, cache=cache, frame_times=frame_times,
        args=args,
    )
    return {
        "scene": scene,
        "gaussians": gaussians,
        "pipe": pipe,
        "white_background": white_background,
        "canonical_fg": canonical_fg,
        "foreground_idx": index,
        "foreground_mask": foreground["foreground_mask"],
        "total_gaussians": foreground["total_gaussians"],
        "frame_ids": frame_ids,
        "frame_times": frame_times,
        "camera_groups": camera_groups,
        "observed_velocity": velocity,
        "observed_confidence": confidence,
        "observed_normal_matrix": normal,
        "observed_valid": valid,
        "fm_base_velocity": fm_base_velocity,
        "fm_base_confidence": fm_base_confidence,
        "fm_base_displacement": fm_base_displacement,
        "fm_base_probability": fm_base_probability,
        "fm_base_valid": fm_base_valid,
        "endpoint_transport": endpoint,
        "metadata": endpoint_metadata,
        "render_evidence_fn": render_evidence_fn,
        "render_loss_fn": render_loss_fn,
        "observation_confidence_fn": observation_confidence_fn,
        "render_transport_feedback_fn": render_transport_feedback_fn,
    }


def _build_render_callbacks(
    *, scene, gaussians, pipe, foreground, camera_groups, white_background,
    confidence, valid, cache, frame_times, args
):
    """Renderer evidence, render loss, and the observation-confidence columns.

    The renderer takes the positions directly through ``means3D_override``,
    which is the API that exists for exactly this and keeps the gradient path
    intact.  The earlier attempt went through ``render_probe_state``, a context
    manager that was created without being entered -- so the canonical geometry
    stayed in place, every render returned the same image whatever positions
    were passed, the critic's render-evidence features were constants, and the
    render loss of spec 19 could not be computed at all.  It also requires the
    dynamic-motion state this pipeline deliberately does not build.

    Only the foreground rows are ever overwritten; background Gaussians stay
    canonical, which is the foreground-only invariant enforced at the one place
    motion reaches the renderer.
    """
    from gaussian_renderer import accumulate_pixel_features_to_gaussians, render
    from utils.closed_loop_motion_losses import RenderLossWeights, render_supervision_loss

    index = foreground["foreground_idx"]
    device = gaussians.get_canonical_xyz.device
    background = torch.tensor(
        [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0], device=device
    )
    canonical_full = gaussians.get_canonical_xyz.detach().clone()
    loss_weights = RenderLossWeights()

    def _install(foreground_positions: torch.Tensor) -> torch.Tensor:
        positions = canonical_full.clone()
        positions[index] = foreground_positions.to(positions)
        return positions

    def _cameras_for(frame: int):
        frame = max(0, min(int(frame), len(camera_groups) - 1))
        return camera_groups[frame]

    @torch.no_grad()
    def render_evidence_fn(foreground_positions: torch.Tensor, frame: int) -> dict:
        """Per-Gaussian image evidence, gathered through the renderer's own ownership."""
        cameras = _cameras_for(frame)
        if not cameras:
            return {}
        camera = cameras[0]
        positions = _install(foreground_positions)
        output = render(camera, gaussians, pipe, background, means3D_override=positions)
        image = output["render"].clamp(0.0, 1.0)
        target = camera.original_image.to(image.device).clamp(0.0, 1.0)
        residual = (image - target).abs().mean(0, keepdim=True)
        gathered = accumulate_pixel_features_to_gaussians(
            camera, gaussians, pipe, residual, torch.ones_like(residual),
            query_positions=positions,
        )
        weight = gathered["weight_sum"]
        per_gaussian = (gathered["feature_sum"][:, 0] / weight.clamp_min(1.0e-8))[index]
        support = weight[index]
        return {
            "rgb_residual": per_gaussian[:, None],
            "raw_support": support[:, None],
            "visible_mass": support[:, None],
            "valid_visible_mass": support[:, None],
            "visibility_ratio": (support > 0).to(image.dtype)[:, None],
            "valid_fraction": (support > 0).to(image.dtype)[:, None],
        }

    def render_loss_fn(foreground_positions: torch.Tensor, frame: int):
        """Differentiable render loss of spec 19, for training the critic.

        The gradient reaches ``foreground_positions`` through the rasterizer, so
        the critic learns the correction the images actually ask for rather than
        only the correction the synthetic corruption asked for.
        """
        cameras = _cameras_for(frame)
        if not cameras:
            return foreground_positions.sum() * 0.0
        camera = cameras[0]
        positions = _install(foreground_positions)
        output = render(camera, gaussians, pipe, background, means3D_override=positions)
        image = output["render"]
        target = camera.original_image.to(image.device)
        reliable = bool(getattr(camera, "depth_reliable", False))
        losses = render_supervision_loss(
            image[None], target[None],
            rendered_depth=output.get("depth") if reliable else None,
            target_depth=camera.invdepthmap.to(image.device) if reliable else None,
            depth_mask=camera.depth_mask.to(image.device) if reliable else None,
            weights=loss_weights,
        )
        return losses["total"]

    @torch.no_grad()
    def observation_confidence_fn(interval: int) -> dict:
        interval = max(0, min(int(interval), confidence.shape[0] - 1))
        return {
            "flow_confidence": confidence[interval][:, None],
            "valid_camera_fraction": valid[interval].to(confidence.dtype)[:, None],
        }

    @torch.no_grad()
    def render_transport_feedback_fn(state, target_frames, output_dir=None) -> dict:
        from utils.motion_correspondence import pair_fixed_cameras
        from utils.rendered_transport_feedback import render_physical_transport_feedback

        labels = state.num_labels
        n = state.num_foreground
        device = gaussians.get_canonical_xyz.device
        energy_fm = torch.zeros((n, labels), device=device)
        energy_obs = torch.zeros_like(energy_fm)
        energy_render = torch.zeros_like(energy_fm)
        rel_fm = torch.zeros_like(energy_fm)
        rel_obs = torch.zeros_like(energy_fm)
        rel_render = torch.zeros_like(energy_fm)
        diagnostics = []
        count = 0
        first_debug = None
        canonical_fg = state.canonical_fg.detach()
        canonical_all = canonical_full
        frames = [max(1, min(int(frame), state.num_frames - 1)) for frame in target_frames]
        span = max(float(frame_times[-1]) - float(frame_times[0]), 1.0e-8)
        max_cameras = max(1, int(getattr(args, "rendered_flow_cameras_per_frame", 1)))
        min_support = float(getattr(args, "rendered_flow_min_support", 1.0e-4))
        target_weight = float(getattr(args, "target_physical_render_weight", 0.0))
        for frame in frames:
            fraction = (float(frame_times[frame]) - float(frame_times[0])) / span
            candidate_fg = torch.cat(
                (
                    canonical_fg[None],
                    state.killing_positions[:, frame].detach().to(canonical_fg),
                ),
                dim=0,
            )
            fm_displacement = state.fm_current_displacement.detach() * float(fraction)
            if frame <= 0:
                fm_conf = state.fm_current_confidence.mean(0)
            else:
                fm_conf = state.fm_current_confidence[:frame].mean(0)
            for camera_index, (source, target) in enumerate(
                pair_fixed_cameras(camera_groups[0], camera_groups[frame])
            ):
                if camera_index >= max_cameras:
                    break
                obs = cache.get(source, target, device)
                feedback = render_physical_transport_feedback(
                    gaussians=gaussians,
                    pipe=pipe,
                    canonical_all=canonical_all,
                    foreground_idx=index,
                    candidate_fg_positions=candidate_fg,
                    source_camera=source,
                    target_camera=target,
                    fm_displacement_fg=fm_displacement,
                    fm_confidence_fg=fm_conf,
                    observed_flow=None if obs is None else obs.flow,
                    observed_confidence=None if obs is None else obs.confidence,
                    background=background,
                    min_support=min_support,
                    target_render_weight=target_weight,
                )
                energy_fm += feedback.per_gaussian_fm_energy
                energy_obs += feedback.per_gaussian_obs_energy
                energy_render += feedback.per_gaussian_render_energy
                rel_fm += feedback.fm_reliability
                rel_obs += feedback.obs_reliability
                rel_render += feedback.render_reliability
                diagnostics.append(feedback.diagnostics)
                count += 1
                if first_debug is None and bool(getattr(args, "save_rendered_flow_debug", False)):
                    first_debug = {
                        "physical_flow": feedback.physical_flow,
                        "physical_flow_weight": feedback.physical_flow_weight,
                        "source": getattr(source, "image_name", ""),
                        "target": getattr(target, "image_name", ""),
                        "frame": int(frame),
                    }
        denom = max(count, 1)
        result = {
            "energy_fm_render": energy_fm / denom,
            "energy_obs_render": energy_obs / denom,
            "energy_target_render": energy_render / denom,
            "reliability_fm_render": (rel_fm / denom).clamp(0.0, 1.0),
            "reliability_obs_render": (rel_obs / denom).clamp(0.0, 1.0),
            "reliability_target_render": (rel_render / denom).clamp(0.0, 1.0),
            "diagnostics": {
                "pairs": count,
                "frames": frames,
                "per_pair": diagnostics,
            },
        }
        if first_debug is not None and output_dir is not None:
            debug_dir = Path(output_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            torch.save(first_debug, debug_dir / "rendered_transport_flow_debug.pt")
        return result

    return render_evidence_fn, render_loss_fn, observation_confidence_fn, render_transport_feedback_fn
