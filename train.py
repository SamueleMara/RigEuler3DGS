#!/usr/bin/env python3
"""Canonical 3DGS followed by the selected dynamic-motion pipeline.

Phases
------
0. Train or load the canonical RGB + semantic + DINO Gaussian model.
1. Discover static plus K rigid Gaussian parts from whole-sequence flow.
2. Fit a frozen physical joint graph.
3. Optimize absolute joint coordinates and export exact SE(3) replay.
"""

from __future__ import annotations

from pathlib import Path
from random import randint
import subprocess
import sys
import uuid

import torch
from argparse import ArgumentParser
from tqdm import tqdm

from arguments import (
    DINOv3_BACKEND,
    DINOv3_HF_LOCAL_DIR,
    DINOv3_HF_MODEL,
    DINOv3_IMAGE_SIZE,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    RIGID_KILLING_REPRESENTATIONS,
    add_dynamic_args,
    apply_dynamic_defaults,
)
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.carving_diagnostics import (
    CarvingDebugWriter,
    evaluate as carving_evaluate,
    initial_state_fingerprint,
)
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import psnr
from utils.loss_utils import (
    _dynamic_root_for_train,
    dino_feature_field_loss,
    l1_loss,
    sam3_feature_field_loss,
    semantic_feature_field_loss,
    ssim,
)
from utils.kpart_rigid_discovery import run_kpart_rigid_discovery
from utils.joint_model_fitting import (
    fit_joint_graph,
    fit_prismatic_joint,
    fit_revolute_joint,
    joint_to_rigid_mode,
)
from utils.joint_trajectory_optimization import (
    derive_qdot_qddot,
    optimize_joint_trajectory,
    render_joint_constrained_motion,
    write_joint_trajectory_html,
)
from utils.unified_dynamic_training import (
    expand_joint_coordinates,
    initialize_gradient_motion_labels,
    initialize_neutral_motion_labels,
    phase1_update_count,
    run_eulerian_phase1_pipeline,
    run_rigid_killing_phase1,
)
from utils.temporal_visibility_pretraining import (
    run_depth_sequence_pretraining,
    run_temporal_visibility_pretraining,
)
from utils.sam3_gaussian_affinity import (
    _root_has_mask_cache,
    _sam3_mask_root_for_train,
    train_sam3_affinity_field,
)
from utils.dynamic_prior import load_dynamic_prior
from utils.killing_prior_adapter import dynamic_prior_to_motion_state


try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    del SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    wandb = None


@torch.no_grad()
def _write_kpart_gt_ablation(stage1, gt_moving_mask, output_dir):
    """Post-solve segmentation ablation. GT never returns to Stage 1."""
    import json

    gt_moving = torch.as_tensor(gt_moving_mask).detach().cpu().bool()
    rows = []
    display = {
        "prior_kpart_initialization": "Prior K+1 initialization",
        "initial_soft": "+ initial soft field unaries",
        "dino_3d": "+ 3D DINO regularization",
        "sam3_3d": "+ 3D SAM3 regularization",
        "endpoint_transport": "+ endpoint transport unary",
        "dino_2d_seam": "+ rendered 2D DINO seam",
        "current_softmax": "Current softmax",
        "rigid_unary": "Rigid unary only",
        "speed_signature": "+ speed signature",
        "dino_graph": "+ DINO graph",
        "render": "+ field-integrated multi-frame render",
        "render_static_rigid_gate": "+ render-denoise static-vs-rigid gate",
        "final_completion": "Final soft completion",
    }
    for key in display:
        if key not in stage1.get("ablation_labels", {}):
            continue
        labels = torch.as_tensor(stage1["ablation_labels"][key]).detach().cpu().long()
        moving = labels > 0
        ambiguous = labels < 0
        static = labels == 0

        def scores(predicted, truth):
            tp = int((predicted & truth).sum())
            fp = int((predicted & ~truth).sum())
            fn = int((~predicted & truth).sum())
            return {
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "iou": tp / max(tp + fp + fn, 1),
            }

        rows.append({
            "key": key, "method": display[key],
            "static": scores(static, ~gt_moving),
            "moving": scores(moving, gt_moving),
            "false_moving_gt_static": int((moving & ~gt_moving).sum()),
            "missed_moving_gt_moving": int(((static | ambiguous) & gt_moving).sum()),
            "ambiguous_count": int(ambiguous.sum()),
            "ambiguous_fraction": float(ambiguous.float().mean()),
        })
    payload = {
        "schema_version": 1,
        "role": "post_solve_evaluation_only",
        "ground_truth_used_as_algorithm_input": False,
        "rows": rows,
    }
    output = Path(output_dir)
    (output / "phase1_kpart_gt_ablation.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    header = "| Method | Static IoU | Moving IoU | FP static→moving | FN moving→static/ambiguous | Ambiguous |\n|---|---:|---:|---:|---:|---:|"
    lines = [header] + [
        f"| {row['method']} | {row['static']['iou']:.4f} | {row['moving']['iou']:.4f} | {row['false_moving_gt_static']} | {row['missed_moving_gt_moving']} | {row['ambiguous_fraction']:.4f} |"
        for row in rows
    ]
    (output / "phase1_kpart_gt_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _ema_postfix(ema, stats, fields, alpha=0.4):
    """Original-3DGS-style loss EMA for compact tqdm reporting."""
    postfix = {}
    for label, key in fields:
        if key not in stats:
            continue
        value = float(stats[key])
        ema[label] = float(alpha) * value + (1.0 - float(alpha)) * ema.get(label, 0.0)
        postfix[label] = f"{ema[label]:.5f}"
    return postfix


def _safe_config(*objects):
    blocked = ("key", "token", "secret", "password", "credential")
    result = {}
    for obj in objects:
        for key, value in vars(obj).items():
            if any(word in str(key).lower() for word in blocked):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[key] = value
            elif isinstance(value, (list, tuple)):
                result[key] = list(value)
    return result


def _prepare_output(dataset, args):
    if not dataset.model_path:
        dataset.model_path = str(Path("output") / uuid.uuid4().hex[:10])
        args.model_path = dataset.model_path
    output = Path(dataset.model_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cfg_args").write_text(str(args), encoding="utf-8")
    writer = None if SummaryWriter is None else SummaryWriter(str(output))
    return output, writer


def _setup_wandb(dataset, opt, pipe, args):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise RuntimeError(
            "W&B logging is enabled by default, but the 'wandb' package is not installed. "
            "Install it with 'pip install wandb', or pass --no_wandb."
        )
    name = args.wandb_name.strip() or Path(dataset.model_path).name
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity.strip() or None,
            id=uuid.uuid4().hex[:12],
            resume="never",
            name=name,
            tags=tags,
            dir=dataset.model_path,
            mode=args.wandb_mode,
            force=args.wandb_mode == "online",
            config=_safe_config(dataset, opt, pipe, args),
        )
        try:
            wandb.define_metric("iteration")
            for pattern in (
                "loss/static/*", "loss/free/*", "diagnostic/free/*",
                "validation/*", "checkpoint/*",
            ):
                wandb.define_metric(pattern, step_metric="iteration")
        except Exception as error:
            print(f"[W&B] metric definitions skipped: {error}")
        print(
            f"[W&B] Logging authenticated entity={run.entity} "
            f"project={run.project} run={run.name} id={run.id}"
        )
        return run
    except Exception as error:
        raise RuntimeError(
            "W&B initialization failed. Run 'wandb login' once for this environment, "
            "or pass --no_wandb to train without W&B."
        ) from error


def _sequence_root(dataset, args):
    explicit = args.root_scene or args.dynamic_sequence_root
    if explicit:
        return Path(explicit).expanduser().resolve()
    source = Path(dataset.source_path).expanduser().resolve()
    return source.parent if source.name == "colmap" else source


def _ensure_dino_targets(dataset, args):
    if not args.use_dino_feature_field:
        return
    root = _sequence_root(dataset, args)
    dino_root = root / args.dinov3_root
    extension = args.dinov3_feature_ext
    if not extension.startswith("."):
        extension = "." + extension
    if dino_root.exists() and any(dino_root.rglob(f"*{extension}")):
        return
    extractor = Path(__file__).resolve().parent / "tools" / "extract_dinov3_features.py"
    if not extractor.exists():
        raise FileNotFoundError(
            f"No DINO targets under {dino_root} and extractor is missing: {extractor}"
        )
    command = [
        sys.executable,
        str(extractor),
        "--sequence_root", str(root),
        "--rgb_dir", args.dynamic_rgb_dir,
        "--out_dir", args.dinov3_root,
        "--backend", DINOv3_BACKEND,
        "--hf_model", DINOv3_HF_MODEL,
        "--hf_local_dir", DINOv3_HF_LOCAL_DIR,
        "--image_size", str(DINOv3_IMAGE_SIZE),
        "--feature_dim", str(args.dino_feature_field_dim),
        "--full_resolution",
    ]
    print(f"[DINO] No precomputed targets found; running {extractor.name}")
    subprocess.run(command, check=True)


def _ensure_sam3_targets(dataset, args):
    if not args.use_sam3_feature_field:
        return
    root = _sequence_root(dataset, args)
    sam3_root = Path(args.sam3_feature_root).expanduser() if args.sam3_feature_root else root / "sam3"
    if not sam3_root.is_absolute():
        sam3_root = root / sam3_root
    if sam3_root.exists() and any(sam3_root.rglob("frame_*.pt")):
        return
    if sam3_root.exists() and (any(sam3_root.rglob("frame_*.npz")) or any(sam3_root.rglob("frame_*.npy"))):
        return
    extractor = Path(__file__).resolve().parent / "tools" / "extract_sam3_features.py"
    local_repo = Path(__file__).resolve().parent / "submodules" / "sam3"
    local_cache = Path("/home/samuelemara/.cache/huggingface/hub/models--facebook--sam3.1")
    raise FileNotFoundError(
        f"No cached SAM3 dense features found under {sam3_root}. "
        "Extract them in the rigeuler3dgs env, for example:\n"
        f"  conda run -n rigeuler3dgs python {extractor} --root {root} "
        f"--frames 0 {args.endpoint_frame if int(args.endpoint_frame) >= 0 else 'LAST'} "
        f"--output {sam3_root} --device cuda\n"
        f"Local SAM3 repo detected: {local_repo if local_repo.exists() else 'not found'}\n"
        f"Local SAM3 HF cache detected: {local_cache if local_cache.exists() else 'not found'}"
    )


def _ensure_sam3_masks(dataset, args):
    if not getattr(args, "use_sam3_affinity_field", False):
        return
    root = _sequence_root(dataset, args)
    mask_root = _sam3_mask_root_for_train(dataset, args)
    if _root_has_mask_cache(mask_root):
        return
    extractor = Path(__file__).resolve().parent / "tools" / "extract_sam3_masks.py"
    suggested_root = mask_root.with_name("sam3_masks") if mask_root.name == "sam3" else mask_root
    raise FileNotFoundError(
        f"No cached SAM3 masks found under {mask_root}. "
        "Extract them offline before training the Gaussian affinity field, for example:\n"
        f"  HF_HUB_OFFLINE=1 python {extractor} --root {root} "
        f"--frames 0 {args.endpoint_frame if int(args.endpoint_frame) >= 0 else 'LAST'} "
        f"--output {suggested_root} --device cuda --debug-overlays"
    )


def _camera_frame_id(camera):
    value = getattr(camera, "frame_id", None)
    if value is not None:
        return int(value)
    import re

    match = re.search(r"frame[_-]?(\d+)", str(getattr(camera, "image_name", "")))
    return int(match.group(1)) if match else 0


def _cameras_at_frame(cameras, frame_idx):
    selected = [camera for camera in cameras if _camera_frame_id(camera) == int(frame_idx)]
    return selected if selected else list(cameras)


def _viewer_step(iteration, total, gaussians, pipe, background, dataset):
    if network_gui is None:
        return
    if network_gui.conn is None:
        network_gui.try_connect()
    while network_gui.conn is not None:
        try:
            image_bytes = None
            camera, train_now, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scale = network_gui.receive()
            if camera is not None:
                image = render(
                    camera, gaussians, pipe, background,
                    scaling_modifier=scale,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                )["render"]
                image_bytes = memoryview(
                    (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                )
            network_gui.send(image_bytes, dataset.source_path)
            if train_now and (iteration < total or not keep_alive):
                break
        except Exception:
            network_gui.conn = None


@torch.no_grad()
def _validate(scene, gaussians, pipe, background, dataset, iteration, writer, wb_run,
              carving=False):
    configurations = (
        ("test", scene.getTestCameras()),
        ("train", scene.getTrainCameras()[:: max(1, len(scene.getTrainCameras()) // 5)][:5]),
    )
    for name, cameras in configurations:
        if not cameras:
            continue
        l1_value = 0.0
        psnr_value = 0.0
        for camera in cameras:
            image = render(
                camera, gaussians, pipe, background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )["render"].clamp(0, 1)
            if carving and getattr(camera, "object_alpha", None) is not None:
                # Composite the reference over the SAME background the render
                # just used. Comparing an object over black against a
                # prediction over white reports a difference that is only the
                # choice of background.
                target = (camera.object_rgb_premult.cuda()
                          + (1.0 - camera.object_alpha.cuda())
                          * background[:, None, None]).clamp(0, 1)
            else:
                target = camera.original_image.cuda().clamp(0, 1)
            l1_value += float(l1_loss(image, target))
            psnr_value += float(psnr(image, target).mean())
        l1_value /= len(cameras)
        psnr_value /= len(cameras)
        print(f"[ITER {iteration}] {name}: L1={l1_value:.6f}, PSNR={psnr_value:.3f}")
        if writer is not None:
            writer.add_scalar(f"validation/{name}_l1", l1_value, iteration)
            writer.add_scalar(f"validation/{name}_psnr", psnr_value, iteration)
        if wb_run is not None:
            wb_run.log({"iteration": iteration, f"validation/{name}_l1": l1_value, f"validation/{name}_psnr": psnr_value}, step=iteration)


def _git_commit():
    """Recorded in initial_state.json so a comparison can be traced to code."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _validate_carving_options(dataset, opt, scene, explicitly_requested=None):
    """Reject the option combinations that would quietly train the wrong thing.

    Carving is on by default, so "carving requested but no masks" is only an
    error when the flag was actually typed. A dataset with no masks has
    nothing to carve against and simply runs the legacy path -- failing those
    runs would break every scene that has no mask directory.
    """
    if explicitly_requested is None:
        explicitly_requested = "--splatfacto_carving" in sys.argv
    carving = bool(getattr(opt, "splatfacto_carving", False))
    debug = bool(getattr(opt, "carving_debug", False))
    has_masks = bool(str(getattr(dataset, "masks", "") or ""))
    active = carving and has_masks

    if explicitly_requested and not has_masks:
        raise ValueError(
            "--splatfacto_carving requires --masks: the mask IS the "
            "ground-truth alpha the objective composites against.")
    if debug and not has_masks:
        raise ValueError(
            "--carving_debug requires --masks. Every diagnostic -- background "
            "leakage above all -- is defined against the foreground mask.")
    if not active:
        return
    if bool(getattr(opt, "random_background", False)):
        raise ValueError(
            "carving and --random_background are two implementations of one "
            "idea and would fight: carving needs the render background and "
            "the composited target to be the SAME sample, and "
            "--random_background draws its own. Drop one, or pass "
            "--no-splatfacto_carving.")
    missing = [c.image_name for c in scene.getTrainCameras()
               if getattr(c, "object_alpha", None) is None]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(scene.getTrainCameras())} training cameras "
            f"have no mask, first: {missing[:3]}. Carving would train those "
            "views on the legacy objective instead, which is two experiments "
            "averaged together rather than one method.")
    print(f"[carving] transparency carving ACTIVE on "
          f"{len(scene.getTrainCameras())} cameras (masks: {dataset.masks})")


def _static_visual_loss(camera, gaussians, pipe, background, dataset, args, opt,
                        depth_weight, bg_generator=None):
    # --splatfacto_carving, and only it, changes the RGB term. --masks alone
    # loads the masks and leaves the objective exactly as it was, so the A/B
    # control can measure background opacity against the same masks the
    # treatment trains on without its own loss being touched. The depth,
    # semantic, DINO and SAM3 terms below are untouched in both branches and
    # still see `background`.
    object_alpha = getattr(camera, "object_alpha", None)
    object_mode = bool(getattr(opt, "splatfacto_carving", False)) and object_alpha is not None
    if object_mode:
        # One random colour per iteration, used by BOTH the render and the
        # ground truth. Over background the target is exactly that colour, so
        # a Gaussian sitting there must reproduce a colour that changes every
        # step. No fixed albedo can, and the only move that lowers the loss is
        # to take its opacity to zero -- which is how the black border splats
        # disappear without a silhouette loss, without touching the CUDA
        # rasteriser, and without masking the prediction.
        #
        # Drawn from a DEDICATED generator when one is supplied. On the global
        # RNG the treatment would consume random numbers the control does not,
        # so the two runs would diverge in camera order and in densify/split
        # decisions, and the comparison would no longer isolate compositing.
        if bg_generator is not None:
            render_background = torch.rand(
                3, device="cuda", dtype=background.dtype, generator=bg_generator)
        else:
            render_background = torch.rand(3, device="cuda")
    else:
        render_background = background
    package = render(
        camera, gaussians, pipe, render_background,
        use_trained_exp=dataset.train_test_exp,
        separate_sh=SPARSE_ADAM_AVAILABLE,
    )
    image = package["render"]
    if object_mode:
        # Deliberately no `image = image * object_alpha` and no masking of the
        # target. Multiplying the prediction by the foreground mask is what let
        # the background keep arbitrary geometry: nothing scored those pixels,
        # so anything could sit there for free. Here the ordinary full-image
        # RGB loss supervises them.
        target = (camera.object_rgb_premult.cuda()
                  + (1.0 - object_alpha.cuda()) * render_background[:, None, None])
    else:
        if camera.alpha_mask is not None:
            image = image * camera.alpha_mask.cuda()
        target = camera.original_image.cuda()
    rgb_l1 = l1_loss(image, target)
    dssim = 1.0 - ssim(image, target)
    total = (1.0 - opt.lambda_dssim) * rgb_l1 + opt.lambda_dssim * dssim
    stats = {"rgb_l1": float(rgb_l1.detach()), "rgb_dssim": float(dssim.detach())}

    if depth_weight > 0 and camera.depth_reliable:
        mask = camera.depth_mask.cuda()
        err = torch.abs(
            (package["depth"] - camera.invdepthmap.cuda()) * mask)
        # Averaged over the MEASURED pixels, not over the frame. These depth
        # maps are sparse -- 1.3% of the frame on blade, 18.3% on oven -- so
        # dividing by the pixel count makes the effective depth weight depend
        # on how much of the image the object happens to fill, and differ by
        # an order of magnitude between two scenes given the same
        # --depth_l1_weight. Dividing by the mask makes the number mean "mean
        # error where depth exists" and comparable across scenes.
        depth_loss = err.sum() / mask.sum().clamp_min(1.0)
        total = total + depth_weight * depth_loss
        stats["depth_l1"] = float(depth_loss.detach())
        stats["depth_pixels"] = float(mask.sum().detach())
    else:
        stats["depth_l1"] = 0.0

    if args.use_semantic_feature_field and args.lambda_semantic_feature_field > 0:
        semantic, semantic_stats = semantic_feature_field_loss(camera, gaussians, pipe, dataset, args)
        if semantic is not None:
            total = total + args.lambda_semantic_feature_field * semantic
        stats.update({f"semantic/{key}": value for key, value in semantic_stats.items()})

    if args.use_dino_feature_field and args.lambda_dino_feature_field > 0:
        feature, feature_stats = dino_feature_field_loss(
            camera, gaussians, pipe, background, dataset, args,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            stage="static",
        )
        if feature is not None:
            total = total + args.lambda_dino_feature_field * feature
        stats.update({f"dino/{key}": value for key, value in feature_stats.items()})
    if args.use_sam3_feature_field and args.lambda_sam3_feature_field > 0:
        feature, feature_stats = sam3_feature_field_loss(
            camera, gaussians, pipe, background, dataset, args,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            stage="static",
        )
        if feature is not None:
            total = total + args.lambda_sam3_feature_field * feature
        stats.update({f"sam3/{key}": value for key, value in feature_stats.items()})
    stats["total"] = float(total.detach())
    return total, stats, package


def train_static(
    scene, gaussians, pipe, background, dataset, args, opt, start_iteration,
    save_iterations, checkpoint_iterations, test_iterations, writer, wb_run,
    viewer_enabled,
):
    total_iterations = int(args.static_train_iters)
    if start_iteration >= total_iterations:
        return
    cameras = scene.getTrainCameras().copy()
    if not cameras:
        raise RuntimeError("No canonical training cameras")
    stack = []
    depth_schedule = get_expon_lr_func(
        opt.depth_l1_weight_init,
        opt.depth_l1_weight_final,
        max_steps=max(1, total_iterations),
    )
    use_sparse = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    ema_losses = {}

    carving = bool(getattr(opt, "splatfacto_carving", False))
    branch = "carving" if carving else "legacy"
    bg_generator = None
    if carving:
        bg_generator = torch.Generator(device="cuda")
        bg_generator.manual_seed(int(getattr(opt, "carving_background_seed", 2026)))
    debug = CarvingDebugWriter(scene.model_path, branch,
                               bool(getattr(opt, "carving_debug", False)))
    debug_interval = max(1, int(getattr(opt, "carving_debug_interval", 1000)))
    if debug.enabled:
        fingerprint = initial_state_fingerprint(gaussians)
        debug.initial_state({
            "branch": branch,
            "splatfacto_carving": carving,
            "carving_background_seed": int(getattr(opt, "carving_background_seed", 2026)),
            "random_background": bool(getattr(opt, "random_background", False)),
            "iterations": total_iterations,
            "start_iteration": int(start_iteration),
            "lambda_dssim": float(opt.lambda_dssim),
            "depth_l1_weight_init": float(opt.depth_l1_weight_init),
            "depth_l1_weight_final": float(opt.depth_l1_weight_final),
            "resolution": int(getattr(dataset, "resolution", -1)),
            "source_path": str(getattr(dataset, "source_path", "")),
            "images": str(getattr(dataset, "images", "")),
            "masks": str(getattr(dataset, "masks", "")),
            "num_initial_gaussians": int(gaussians.get_xyz.shape[0]),
            "num_train_cameras": len(cameras),
            "num_masks": sum(1 for c in cameras
                             if getattr(c, "object_alpha", None) is not None),
            "cameras": [c.image_name for c in cameras],
            "git_commit": _git_commit(),
            "initial_state_sha256": fingerprint,
        })
        # Iteration 0: BEFORE any parameter update, so both runs are measured
        # on identical Gaussians and the row is a true common baseline.
        carving_evaluate(cameras, gaussians, pipe, 0, branch, debug, previews=2)

    progress = tqdm(range(start_iteration + 1, total_iterations + 1), desc="Phase 0: canonical 3DGS")
    for iteration in progress:
        if viewer_enabled:
            _viewer_step(iteration, total_iterations, gaussians, pipe, background, dataset)
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not stack:
            stack = cameras.copy()
        camera = stack.pop(randint(0, len(stack) - 1))
        debug.trace(iteration, camera.image_name)
        bg = torch.rand(3, device="cuda") if opt.random_background else background
        total, stats, package = _static_visual_loss(
            camera, gaussians, pipe, bg, dataset, args, opt, depth_schedule(iteration),
            bg_generator=bg_generator,
        )
        total.backward()

        with torch.no_grad():
            visibility = package["visibility_filter"]
            radii = package["radii"]
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility] = torch.maximum(
                    gaussians.max_radii2D[visibility], radii[visibility]
                )
                gaussians.add_densification_stats(package["viewspace_points"], visibility)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii
                    )
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)
            if use_sparse:
                gaussians.optimizer.step(radii > 0, radii.shape[0])
            else:
                gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if writer is not None:
            for key, value in stats.items():
                writer.add_scalar(f"loss/static/{key}", value, iteration)
        if wb_run is not None and iteration % args.wandb_log_interval == 0:
            wb_run.log({"iteration": iteration, **{f"loss/static/{key}": value for key, value in stats.items()}}, step=iteration)
        postfix = _ema_postfix(
            ema_losses,
            stats,
            (
                ("Loss", "total"),
                ("RGB", "rgb_l1"),
                ("Depth", "depth_l1"),
                ("Sem", "semantic/semantic_bce"),
                ("DINO", "dino/dino_cosine"),
                ("SAM3", "sam3/sam3_cosine"),
            ),
        )
        if iteration % 10 == 0:
            progress.set_postfix(postfix)
        if debug.enabled and (iteration % debug_interval == 0
                              or iteration == total_iterations):
            # After the update, so the row describes the model as it now
            # stands. metrics.csv documents this boundary; iteration 0 is the
            # only pre-update row.
            carving_evaluate(
                cameras, gaussians, pipe, iteration, branch, debug,
                photometric={"L1": stats.get("rgb_l1", ""),
                             "SSIM": 1.0 - float(stats.get("rgb_dssim", 0.0)),
                             "total": stats.get("total", "")},
                previews=2 if iteration == total_iterations else 0,
            )
        if iteration in test_iterations:
            _validate(scene, gaussians, pipe, background, dataset, iteration, writer,
                      wb_run, carving=carving)
        if iteration in save_iterations:
            scene.save(iteration)
        if iteration in checkpoint_iterations:
            torch.save((gaussians.capture(), iteration), Path(scene.model_path) / f"chkpnt{iteration}.pth")
    debug.close()


def _load_start_state(scene, gaussians, opt, args, checkpoint):
    if checkpoint:
        model_state, iteration = torch.load(checkpoint, map_location="cuda", weights_only=False)
        gaussians.restore(model_state, opt)
        return int(iteration), "checkpoint"
    start_ply = Path(args.start_ply).expanduser() if args.start_ply else None
    automatic = False
    if start_ply is None and args.dynamic_motion and not args.force_static_retrain:
        candidate = Path(scene.model_path) / "point_cloud" / f"iteration_{args.static_train_iters}" / "point_cloud.ply"
        if candidate.exists():
            start_ply = candidate
            automatic = True
    if start_ply is not None:
        if not start_ply.exists():
            raise FileNotFoundError(f"Starting PLY does not exist: {start_ply}")
        gaussians.load_ply(str(start_ply), scene._scene_args.train_test_exp)
        return int(args.static_train_iters if automatic else args.start_ply_iteration), "ply"
    return 0, "fresh"


def training(dataset, opt, pipe, args, checkpoint, save_iterations, checkpoint_iterations, test_iterations):
    if opt.optimizer_type == "sparse_adam" and not SPARSE_ADAM_AVAILABLE:
        raise RuntimeError("Sparse Adam requested but the accelerated rasterizer is not installed")
    output, writer = _prepare_output(dataset, args)
    wb_run = _setup_wandb(dataset, opt, pipe, args)
    _ensure_dino_targets(dataset, args)
    _ensure_sam3_targets(dataset, args)
    _ensure_sam3_masks(dataset, args)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, dynamic_args=None)
    _validate_carving_options(dataset, opt, scene)
    dino_dim = args.dino_feature_field_dim if args.use_dino_feature_field else 0
    sam3_dim = args.sam3_feature_dim if args.use_sam3_feature_field else 0
    gaussians.init_semantic_and_feature_fields(
        args.use_semantic_feature_field,
        dino_dim,
        sam3_dim=sam3_dim,
    )
    opt.semantic_feature_lr = args.semantic_feature_lr
    opt.dino_feature_lr = args.dino_feature_lr
    opt.sam3_feature_lr = args.sam3_feature_lr
    gaussians.training_setup(opt)
    first_iteration, load_kind = _load_start_state(scene, gaussians, opt, args, checkpoint)
    if load_kind == "ply":
        # PLY loading replaces Parameter objects; bind a fresh optimizer to the
        # loaded tensors. Checkpoint restore already restores its optimizer.
        gaussians.init_semantic_and_feature_fields(
            args.use_semantic_feature_field,
            dino_dim,
            sam3_dim=sam3_dim,
        )
        gaussians.training_setup(opt)

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    train_static(
        scene, gaussians, pipe, background, dataset, args, opt,
        min(first_iteration, args.static_train_iters),
        set(save_iterations), set(checkpoint_iterations), set(test_iterations),
        writer, wb_run, not args.disable_viewer,
    )
    if first_iteration < args.static_train_iters or args.static_train_iters in save_iterations:
        scene.save(args.static_train_iters)

    dynamic_sequence_enabled = False

    def enable_dynamic_sequence_once():
        nonlocal dynamic_sequence_enabled
        if dynamic_sequence_enabled:
            return
        previous_device = dataset.data_device
        dataset.data_device = args.dynamic_camera_data_device
        try:
            if not scene.enable_dynamic_sequence(dataset, args):
                raise RuntimeError(
                    "Dynamic RGB sequence was not found. Expected ROOT/rgb/cam_*/frame_*.png"
                )
        finally:
            dataset.data_device = previous_device
        dynamic_sequence_enabled = True

    if args.dynamic_motion:
        enable_dynamic_sequence_once()

    phase0_aux_updates = 0

    if getattr(args, "use_sam3_affinity_field", False):
        canonical_cameras = (
            _cameras_at_frame(scene.getTrainCameras(), 0)
            if args.dynamic_motion
            else scene.getTrainCameras()
        )
        sam3_summary = train_sam3_affinity_field(
            gaussians,
            canonical_cameras,
            pipe,
            dataset,
            args,
            scene.model_path,
            writer=writer,
            wb_run=wb_run,
            start_step=int(args.static_train_iters) + int(phase0_aux_updates),
            stage_name="canonical",
            frame_idx=0,
        )
        phase0_aux_updates += int(args.sam3_affinity_iters)
        gaussians._sam3_affinity_training_config = sam3_summary.get("training_config", {})
        gaussians._sam3_affinity_frame_idx = 0
        scene.save(args.static_train_iters)

    if args.dynamic_motion:

        if args.use_endpoint_static_model:
            from utils.endpoint_static_model import prepare_endpoint_static_model

            endpoint_state = prepare_endpoint_static_model(
                scene,
                gaussians,
                pipe,
                background,
                dataset,
                args,
                opt,
                static_visual_loss_fn=_static_visual_loss,
            )
            if getattr(args, "use_sam3_affinity_field", False):
                endpoint_model = endpoint_state["model"]
                endpoint_cameras = endpoint_state["cameras"]
                endpoint_frame = endpoint_state.get("frame", args.endpoint_frame)
                endpoint_summary = train_sam3_affinity_field(
                    endpoint_model,
                    endpoint_cameras,
                    pipe,
                    dataset,
                    args,
                    endpoint_state["model_path"],
                    writer=writer,
                    wb_run=wb_run,
                    start_step=int(args.static_train_iters) + int(phase0_aux_updates),
                    stage_name="endpoint",
                    frame_idx=endpoint_frame,
                )
                phase0_aux_updates += int(args.sam3_affinity_iters)
                endpoint_model._sam3_affinity_training_config = endpoint_summary.get("training_config", {})
                endpoint_model._sam3_affinity_frame_idx = endpoint_frame
                endpoint_dir = (
                    Path(endpoint_state["model_path"])
                    / "point_cloud"
                    / f"iteration_{int(getattr(args, 'endpoint_static_iters', 0))}"
                )
                endpoint_dir.mkdir(parents=True, exist_ok=True)
                endpoint_model.save_ply(str(endpoint_dir / "point_cloud.ply"))
                endpoint_state["sam3_affinity_features"] = endpoint_model.get_sam3_affinity.detach()

        phase1_total_updates = phase1_update_count(scene.getTrainCameras(), args)
        args.phase1_total_updates = phase1_total_updates
        args.iterations = args.static_train_iters + phase0_aux_updates + phase1_total_updates

        def log_dynamic(stage, local_iteration, stats):
            global_iteration = args.static_train_iters + phase0_aux_updates + local_iteration
            if writer is not None:
                for key, value in stats.items():
                    writer.add_scalar(key if "/" in key else f"loss/{stage}/{key}", value, global_iteration)
            if wb_run is not None and local_iteration % args.wandb_log_interval == 0:
                payload = {"iteration": global_iteration}
                for key, value in stats.items():
                    if "/" in key and key.split("/", 1)[0] in {
                        "visibility", "flow_calibration", "assignment_pretrain",
                        "rigid_fit", "assignment_refine", "sequence", "camera",
                        "eulerian", "lagrangian", "comparison", "depth_pretrain",
                        "neutral_discovery",
                        "responsibility", "eulerian_support",
                        "part_discovery", "joint_fit", "joint_compare",
                        "joint_selected", "joint_trajectory", "quaternion_transport",
                        "discovery", "nested_inner", "nested_outer",
                    }:
                        payload[key] = value
                    else:
                        payload[f"diagnostic/{stage}/{key}"] = value
                wb_run.log(payload, step=global_iteration)

        depth_pretrain_only = False
        if not gaussians.dynamic_enabled:
            # Infer material identity on the complete canonical scene before
            # semantic compaction. The rigid branch uses whole-sequence
            # visibility; the Eulerian fallback retains gradient attribution.
            if bool(getattr(args, "enable_nested_killing_part_discovery", False)):
                precompact_motion_state = initialize_neutral_motion_labels(
                    gaussians
                )
            elif str(getattr(args, "dynamic_prior_path", "") or "").strip():
                dynamic_prior = load_dynamic_prior(args.dynamic_prior_path)
                precompact_motion_state = dynamic_prior_to_motion_state(
                    dynamic_prior,
                    gaussians.get_canonical_xyz.detach().cpu(),
                    total_gaussians=int(gaussians.get_canonical_xyz.shape[0]),
                    use_soft_weights=bool(args.prior_use_soft_weights),
                    hard_threshold=float(args.prior_hard_threshold),
                    hard_requires_confidence=not bool(args.freeze_dynamic_prior_labels),
                )
                print(
                    "[Dynamic prior] initialized material probabilities from "
                    f"{dynamic_prior.method}: {int(dynamic_prior.gaussian_ids.numel())} foreground priors, "
                    f"mean_p={float(dynamic_prior.probability.mean()):.4f}, "
                    f"mean_conf={float(dynamic_prior.confidence.mean()):.4f}"
                )
            elif args.kpart_rigid_discovery:
                precompact_motion_state = initialize_neutral_motion_labels(
                    gaussians
                )
            elif args.rigid_responsibility_training:
                precompact_motion_state = initialize_neutral_motion_labels(
                    gaussians
                )
            elif args.phase1_neutral_rigid_discovery:
                precompact_motion_state = initialize_neutral_motion_labels(
                    gaussians
                )
            elif args.phase1_depth_sequence_pretrain:
                precompact_motion_state = run_depth_sequence_pretraining(
                    scene, gaussians, pipe, dataset, args, log_fn=log_dynamic
                )
                depth_pretrain_only = bool(args.phase1_depth_pretrain_only)
            elif (
                args.phase1_motion_representation in RIGID_KILLING_REPRESENTATIONS
                and args.sequence_pretrain_assignment
            ):
                precompact_motion_state = run_temporal_visibility_pretraining(
                    scene, gaussians, pipe, dataset, args, log_fn=log_dynamic
                )
            else:
                precompact_motion_state = initialize_gradient_motion_labels(
                    scene, gaussians, pipe, background, dataset, args, opt,
                )
            if not depth_pretrain_only:
                gaussians.init_dynamic_motion(
                    args,
                    scene.cameras_extent,
                    scene.getTrainCameras(),
                    gradient_motion_state=precompact_motion_state,
                )
        elif (
            gaussians.motion_representation == "eulerian_curl_strain"
            and gaussians.motion_optimizer is None
        ):
            gaussians.setup_dynamic_optimizers(args)
        if not depth_pretrain_only:
            gaussians.freeze_canonical(True, True)

        if (
            not depth_pretrain_only
            and str(getattr(args, "dynamic_discovery_mode", "legacy")) == "render_denoise"
        ):
            from scene.render_guided_motion_probe import run_render_guided_displacement_probe

            stage_a = run_render_guided_displacement_probe(
                scene,
                gaussians,
                pipe,
                dataset,
                args,
                log_fn=log_dynamic,
            )
            gaussians.set_render_discovery_state(stage_a["evidence"], args)

        if depth_pretrain_only:
            print("[Depth pretrain] stopping before rigid fitting as requested")
        elif bool(getattr(args, "enable_nested_killing_part_discovery", False)):
            discovery_updates = (
                int(getattr(args, "part_outer_iters", 1))
                * int(getattr(args, "part_outer_embedding_steps", 1))
            )
            from rigid_parts.nested_pipeline import run_nested_part_embedding_killing_discovery

            stage1 = run_nested_part_embedding_killing_discovery(
                scene,
                gaussians,
                pipe,
                background,
                dataset,
                args,
                Path(scene.model_path) / "dynamic_debug",
                log_fn=log_dynamic,
            )
            if bool(stage1.get("init_only", False)):
                # Spec §28: the initialization ablation stops here, before the
                # first inner Killing solve. Everything below consumes solver
                # state (frame_times, modes, coordinates) that was never built.
                print("[Nested Killing] --init_only: returning after initialization; "
                      "no Killing solve, no Stage 2, no Stage 3")
                return
            if args.fit_joints or args.optimize_joint_trajectory:
                raise RuntimeError(
                    "Nested Killing Stage 1 writes exact SE(3) part transforms, but "
                    "legacy Stage 2 joint fitting is not wired to that new state yet."
                )
        elif args.kpart_rigid_discovery:
            discovery_updates = (
                int(args.kpart_cycles)
                if args.kpart_cycles is not None
                else int(args.kpart_anneal_cycles) + int(args.kpart_plateau_cycles)
            )
            stage1 = run_kpart_rigid_discovery(
                scene,
                gaussians,
                pipe,
                dataset,
                args,
                log_fn=log_dynamic,
            )
            if bool(getattr(args, "enable_rigid_part_discovery", False)):
                from rigid_parts.pipeline import run_exact_closed_loop_rigid_part_discovery
                from utils.temporal_visibility_pretraining import _atomic_torch_save
                from utils.visualize_transported_gaussians_html import write_transported_gaussians_html

                rigid_part_result = run_exact_closed_loop_rigid_part_discovery(
                    stage1,
                    scene,
                    gaussians,
                    pipe,
                    background,
                    dataset,
                    args,
                    Path(scene.model_path) / "dynamic_debug",
                    log_fn=log_dynamic,
                )
                stage1["exact_closed_loop_rigid_part_discovery"] = rigid_part_result["summary"]
                stage1["responsibilities"] = rigid_part_result["responsibilities"]
                stage1["labels"] = rigid_part_result["labels"].to(torch.as_tensor(stage1["labels"]).device)
                stage1["classification"]["labels"] = stage1["labels"]
                refined_tracks = rigid_part_result["rigid_tracks"]
                refined_tracks.update({
                    "motion_membership": stage1["responsibilities"][:, 1:].sum(dim=-1).cpu(),
                    "part_probabilities": stage1["responsibilities"].cpu(),
                    "fixed_cameras": stage1["coarse_tracks"].get("fixed_cameras", []),
                })
                stage1["raw_tracks"] = rigid_part_result["raw_tracks"]
                stage1["rigid_tracks"] = refined_tracks
                stage1["part_rotation"] = rigid_part_result["state"]["rotation"]
                stage1["part_translation"] = rigid_part_result["state"]["translation"]
                stage1["part_omega"] = rigid_part_result["state"]["omega"]
                stage1["part_b"] = rigid_part_result["state"]["b"]
                stage1["part_com"] = rigid_part_result["state"].get("com")
                stage1["part_inertia"] = rigid_part_result["state"].get("inertia")
                stage1["stage2_legacy_modes_are_stale"] = bool(
                    rigid_part_result.get("stage2_legacy_modes_are_stale", False)
                )
                stage1["coarse_tracks"] = refined_tracks
                debug_dir = Path(scene.model_path) / "dynamic_debug"
                _atomic_torch_save(refined_tracks, debug_dir / "rigid_projected_gaussian_tracks.pt")
                write_transported_gaussians_html(
                    refined_tracks,
                    debug_dir / "rigid_projected_motion.html",
                    title="Stage 1 exact closed-loop rigid projection",
                    max_points=int(args.transported_html_max_points),
                    max_vectors=min(250, int(args.transported_html_max_points)),
                )
            # Stage 2 and Stage 3 must agree on which frames they run over, so
            # the subset is resolved once here and handed to both.
            joint_frame_selection = None
            if args.fit_joints or args.optimize_joint_trajectory:
                from utils.unified_dynamic_training import (
                    _complete_frame_groups,
                    _ordered_dynamic_cameras,
                    resolve_joint_frame_selection,
                )

                joint_frame_selection = resolve_joint_frame_selection(
                    _complete_frame_groups(
                        _ordered_dynamic_cameras(scene.getTrainCameras()), args
                    ),
                    args,
                )
                print(
                    f"[Joint frames] '{joint_frame_selection['spec']}' -> "
                    f"{joint_frame_selection['selected_frames']}/"
                    f"{joint_frame_selection['available_frames']} frames "
                    f"{joint_frame_selection['frame_ids'][:1]}..{joint_frame_selection['frame_ids'][-1:]}, "
                    f"max gap {joint_frame_selection['max_gap_seconds']:.3f}s"
                )
            stage2 = None
            if args.fit_joints:
                if bool(stage1.get("stage2_legacy_modes_are_stale", False)):
                    raise RuntimeError(
                        "Exact closed-loop rigid-part discovery produced part_rotation/"
                        "part_translation/part_omega/part_b states. Legacy Stage 2 still "
                        "consumes stale velocity-field modes/coordinates, so joint fitting is "
                        "disabled until Stage 2 is explicitly wired to the new rigid-part state."
                    )
                stage2 = fit_joint_graph(
                    scene,
                    gaussians,
                    pipe,
                    background,
                    dataset,
                    opt,
                    stage1,
                    args,
                    Path(scene.model_path) / "dynamic_debug",
                    frame_selection=joint_frame_selection,
                    log_fn=lambda stage, local, stats: log_dynamic(
                        stage, discovery_updates + local, stats
                    ),
                )
            if stage2 is None:
                provisional = []
                for mode, coordinate in zip(stage1["modes"], stage1["coordinates"]):
                    provisional.append(
                        fit_prismatic_joint(mode, coordinate)
                        if mode.mode_type == "translational"
                        else fit_revolute_joint(mode, coordinate)
                    )
                joints = provisional
                coordinates = torch.stack(stage1["coordinates"])
                velocity, _ = derive_qdot_qddot(
                    coordinates, stage1["frame_times"]
                )
                final_tracks = render_joint_constrained_motion(
                    gaussians.get_canonical_xyz,
                    gaussians.get_rotation,
                    stage1["labels"],
                    joints,
                    coordinates,
                    velocity,
                    stage1["frame_times"],
                )
            elif args.optimize_joint_trajectory:
                stage3 = optimize_joint_trajectory(
                    scene,
                    gaussians,
                    pipe,
                    background,
                    dataset,
                    opt,
                    args,
                    stage1,
                    stage2,
                    frame_selection=joint_frame_selection,
                log_fn=lambda stage, local, stats: log_dynamic(
                    stage,
                    discovery_updates
                    + len(stage2["joints"])
                    * ((2 if str(args.joint_types) == "auto" else 1) * int(args.joint_compare_iters) + 1)
                    + local,
                    stats,
                ),
                )
                joints = stage3["joints"]
                coordinates = stage3["coordinates"]
                velocity = stage3["velocity"]
                final_tracks = stage3["tracks"]
                if joint_frame_selection["is_sparse"]:
                    # Stage 3 solved on a subset; the export replays every frame.
                    coordinates, expansion_scales = expand_joint_coordinates(
                        coordinates, joint_frame_selection["positions"], stage1["coordinates"]
                    )
                    velocity, _ = derive_qdot_qddot(coordinates, stage1["frame_times"])
                    final_tracks = render_joint_constrained_motion(
                        gaussians.get_canonical_xyz,
                        gaussians.get_rotation,
                        stage1["labels"],
                        joints,
                        coordinates,
                        velocity,
                        stage1["frame_times"],
                    )
                    print(
                        "[Joint frames] expanded the sparse Stage-3 coordinate onto all "
                        f"{len(stage1['frame_times'])} frames, profile scale "
                        f"{[round(value, 5) for value in expansion_scales]}"
                    )
            else:
                joints = stage2["joints"]
                coordinates = torch.stack(
                    [torch.as_tensor(joint["coarse_q"]) for joint in joints]
                )
                if joint_frame_selection["is_sparse"]:
                    coordinates, _ = expand_joint_coordinates(
                        coordinates, joint_frame_selection["positions"], stage1["coordinates"]
                    )
                velocity, _ = derive_qdot_qddot(
                    coordinates, stage1["frame_times"]
                )
                final_tracks = render_joint_constrained_motion(
                    gaussians.get_canonical_xyz,
                    gaussians.get_rotation,
                    stage1["labels"],
                    joints,
                    coordinates,
                    velocity,
                    stage1["frame_times"],
                )
            gaussians.set_kpart_joint_state(
                joints,
                coordinates,
                velocity,
                stage1["frame_times"],
                stage1["responsibilities"],
                stage1["labels"],
                diagnostics={
                    "stage1_support": args.kpart_support,
                    "stage2_fitted": bool(args.fit_joints),
                    "stage3_optimized": bool(args.optimize_joint_trajectory),
                },
            )
            if args.rigid_analytic_validation:
                if len(joints) != 1:
                    print("[K-part GT validation] skipped: the current validator maps one configured moving part")
                else:
                    from utils.rigid_killing_validation import (
                        MEASURED_FLOW_SOURCE_ID,
                        OCCLUSION_CALIBRATED_FLOW_SOURCE_ID,
                        load_rigid_dataset_ground_truth,
                        validate_frozen_rigid_solution,
                    )

                    # GT is imported only after every estimator artifact and the
                    # frozen replay state have been produced.
                    ground_truth = load_rigid_dataset_ground_truth(
                        _dynamic_root_for_train(dataset, args),
                        gaussians.get_canonical_xyz,
                        moving_part=int(args.rigid_analytic_moving_part),
                        gaussian_part_labels=getattr(gaussians, "_gt_part_labels", None),
                    )
                    if stage1.get("ablation_labels"):
                        _write_kpart_gt_ablation(
                            stage1,
                            ground_truth.moving_mask,
                            Path(scene.model_path) / "dynamic_debug",
                        )
                    if stage2 is not None:
                        import json
                        import math

                        candidate_evaluation = []
                        for part in stage2.get("comparison", []):
                            for candidate in part["candidates"]:
                                parameter = candidate["joint_parameters"]
                                axis = torch.as_tensor(parameter["axis"], dtype=torch.float64)
                                cosine = abs(float(torch.dot(axis, ground_truth.axis)))
                                angle_error = math.degrees(
                                    math.acos(min(1.0, max(-1.0, cosine)))
                                )
                                pivot = parameter.get("pivot")
                                if pivot is None:
                                    axis_line_error = None
                                else:
                                    pivot = torch.as_tensor(pivot, dtype=torch.float64)
                                    cross = torch.cross(axis, ground_truth.axis, dim=-1)
                                    if float(torch.linalg.norm(cross)) > 1.0e-8:
                                        axis_line_error = abs(
                                            float(torch.dot(ground_truth.reference_point - pivot, cross))
                                        ) / float(torch.linalg.norm(cross))
                                    else:
                                        offset = ground_truth.reference_point - pivot
                                        axis_line_error = float(
                                            torch.linalg.norm(offset - torch.dot(offset, axis) * axis)
                                        )
                                candidate_evaluation.append({
                                    "part": int(part["part"]),
                                    "joint_type": parameter["joint_type"],
                                    "selected": parameter["joint_type"] == part["selected"],
                                    "axis_cosine": cosine,
                                    "axis_angular_error_degrees": angle_error,
                                    "axis_line_distance_to_gt": axis_line_error,
                                    "pitch_error_from_revolute_gt": abs(
                                        float(parameter.get("pitch", 0.0))
                                    ),
                                    "role": "post_solve_evaluation_only",
                                })
                        with (
                            Path(scene.model_path)
                            / "dynamic_debug"
                            / "phase2_joint_candidates_gt_evaluation.json"
                        ).open("w", encoding="utf-8") as stream:
                            json.dump({
                                "schema_version": 1,
                                "ground_truth_used_as_algorithm_input": False,
                                "candidates": candidate_evaluation,
                            }, stream, indent=2, allow_nan=False)
                    interval_speed = torch.diff(torch.as_tensor(coordinates[0]).double()) / torch.diff(
                        torch.as_tensor(stage1["frame_times"]).double()
                    )
                    report = validate_frozen_rigid_solution(
                        ground_truth,
                        joint_to_rigid_mode(joints[0]),
                        interval_speed,
                        stage1["frame_ids"],
                        stage1["frame_times"],
                        gaussians.get_canonical_xyz,
                        list(final_tracks["positions"]),
                        torch.as_tensor(stage1["labels"]) > 0,
                        solver_observation_source_id=(
                            OCCLUSION_CALIBRATED_FLOW_SOURCE_ID
                            if args.optimize_joint_trajectory
                            else MEASURED_FLOW_SOURCE_ID
                        ),
                        static_tolerance=float(args.rigid_static_displacement_tolerance),
                    )
                    report["ground_truth_used_as_algorithm_input"] = False
                    validation_name = (
                        "phase3_joint_gt_validation.json"
                        if args.optimize_joint_trajectory
                        else "phase1_kpart_gt_validation.json"
                    )
                    import json
                    with (Path(scene.model_path) / "dynamic_debug" / validation_name).open("w", encoding="utf-8") as stream:
                        json.dump(report, stream, indent=2, allow_nan=False)
                    if args.optimize_joint_trajectory:
                        gt_q = torch.tensor(
                            [ground_truth.frame_values[int(frame)] for frame in report["frame_ids"]],
                            dtype=torch.float64,
                        )
                        gt_q = gt_q - gt_q[0]
                        gt_interval_velocity = torch.diff(gt_q) / torch.diff(
                            torch.as_tensor(stage1["frame_times"], dtype=torch.float64)
                        )
                        gt_velocity = torch.cat((gt_interval_velocity, gt_interval_velocity[-1:]))
                        write_joint_trajectory_html(
                            coordinates,
                            velocity,
                            stage1["frame_times"],
                            Path(scene.model_path) / "dynamic_debug" / "phase3_joint_trajectory_gt_evaluation.html",
                            gt_coordinates=gt_q,
                            gt_velocity=gt_velocity,
                        )
                    print(f"[K-part GT validation] passed={report['acceptance_passed']}")
        elif args.phase1_motion_representation in RIGID_KILLING_REPRESENTATIONS:
            run_rigid_killing_phase1(
                scene,
                gaussians,
                pipe,
                background,
                dataset,
                args,
                opt,
                log_fn=log_dynamic,
            )
        else:
            run_eulerian_phase1_pipeline(
                scene, gaussians, pipe, background, dataset, args, opt,
                log_fn=log_dynamic,
            )

        if args.render_after_training:
            from utils.render_dynamic_motion import render_dynamic_sequence
            render_dynamic_sequence(
                scene, gaussians, pipe, background, args, args.iterations,
                split="all",
                use_tracks=False,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )

    final_iteration = args.iterations
    scene.save(final_iteration)
    torch.save((gaussians.capture(), final_iteration), output / f"chkpnt{final_iteration}.pth")
    if writer is not None:
        writer.close()
    if wb_run is not None:
        wb_run.finish()
    print(f"Training complete: {output}")


def main():
    parser = ArgumentParser(description="Rigid/Eulerian dynamic 3D Gaussian Splatting")
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    add_dynamic_args(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--disable_viewer", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default="")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    args = apply_dynamic_defaults(parser.parse_args())
    if bool(getattr(args, "run_dynamic_prior_comparison", False)):
        comparison_output = (
            Path(args.comparison_output_dir).expanduser().resolve()
            if str(getattr(args, "comparison_output_dir", "") or "").strip()
            else Path(args.model_path).expanduser().resolve() / "dynamic_prior_comparison"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "tools" / "run_dynamic_prior_comparison.py"),
            "--root-scene",
            str(Path(args.source_path).expanduser().resolve()),
            "--static-model-dir",
            str(Path(args.model_path).expanduser().resolve()),
            "--static-iteration",
            str(int(args.static_train_iters)),
            "--output",
            str(comparison_output),
            "--methods",
            str(args.comparison_methods),
            "--semantic-dynamic-threshold",
            str(float(args.semantic_dynamic_threshold)),
            "--semantic-dynamic-min-points",
            str(int(args.semantic_dynamic_min_points)),
            "--seed",
            str(int(args.comparison_seed)),
            "--comparison-denoise-prob-bias",
            str(float(args.comparison_denoise_prob_bias)),
            "--comparison-denoise-prob-temperature",
            str(float(args.comparison_denoise_prob_temperature)),
            "--prior-smooth-knn",
            str(int(args.comparison_prior_smooth_knn)),
            "--prior-smooth-iters",
            str(int(args.comparison_prior_smooth_iters)),
            "--prior-smooth-alpha",
            str(float(args.comparison_prior_smooth_alpha)),
            "--kpart-k",
            str(int(args.comparison_kpart_k)),
            "--kpart-dynamic-prior-weight",
            str(float(args.kpart_dynamic_prior_weight)),
            "--kpart-anneal-cycles",
            str(int(args.kpart_anneal_cycles)),
            "--kpart-plateau-cycles",
            str(int(args.kpart_plateau_cycles)),
            "--kpart-confidence-threshold",
            str(float(args.kpart_confidence_threshold)),
            "--fm-endpoint-knn",
            str(int(args.fm_endpoint_knn)),
            "--fm-transport-temperature",
            str(float(args.fm_transport_temperature)),
            "--fm-transport-robust-delta",
            str(float(args.fm_transport_robust_delta)),
            "--fm-motion-scale",
            str(args.fm_motion_scale),
            "--fm-confidence-entropy-scale",
            str(float(args.fm_confidence_entropy_scale)),
            "--fm-prior-prob-bias",
            str(float(args.fm_prior_prob_bias)),
            "--fm-prior-prob-temperature",
            str(float(args.fm_prior_prob_temperature)),
            "--fm-transport-chunk-size",
            str(int(args.fm_transport_chunk_size)),
            "--fm-hidden-dim",
            str(int(args.fm_hidden_dim)),
            "--fm-layers",
            str(int(args.fm_layers)),
            "--fm-lr",
            str(float(args.fm_lr)),
            "--fm-iters",
            str(int(args.fm_iters)),
            "--fm-batch-size",
            str(int(args.fm_batch_size)),
            "--fm-identity-weight",
            str(float(args.fm_identity_weight)),
            "--fm-smooth-weight",
            str(float(args.fm_smooth_weight)),
            "--fm-integrator",
            str(args.fm_integrator),
            "--fm-integration-steps",
            str(int(args.fm_integration_steps)),
            "--phase1-motion-representation",
            str(args.phase1_motion_representation)
            if str(args.phase1_motion_representation) in RIGID_KILLING_REPRESENTATIONS
            else "rigid_killing_eulerian",
            "--prior-hard-threshold",
            str(float(args.prior_hard_threshold)),
            "--curl-grid-target-cells",
            str(int(args.curl_grid_target_cells)),
            "--endpoint-frame",
            str(int(args.endpoint_frame)),
            "--endpoint-iteration",
            str(int(args.endpoint_static_iters)),
            "--endpoint-init",
            str(args.endpoint_init),
            "--endpoint-depth-voxel-size",
            str(float(args.endpoint_depth_voxel_size)),
        ]
        if args.kpart_cycles is not None:
            command.extend(["--kpart-cycles", str(int(args.kpart_cycles))])
        if bool(args.comparison_fit_joints or args.fit_joints):
            command.extend(
                [
                    "--fit-joints",
                    "--joint-types",
                    str(args.joint_types),
                    "--joint-topology",
                    str(args.joint_topology),
                    "--joint-compare-iters",
                    str(int(args.joint_compare_iters)),
                    "--joint-compare-batch-size",
                    str(int(args.joint_compare_batch_size)),
                    "--joint-compare-validation-cameras",
                    str(int(args.joint_compare_validation_cameras)),
                ]
            )
        if bool(args.comparison_optimize_joint_trajectory or args.optimize_joint_trajectory):
            command.extend(
                [
                    "--optimize-joint-trajectory",
                    "--joint-trajectory-mode",
                    str(args.joint_trajectory_mode),
                    "--joint-trajectory-iters",
                    str(int(args.joint_trajectory_iters)),
                    "--joint-trajectory-batch-size",
                    str(int(args.joint_trajectory_batch_size)),
                    "--joint-trajectory-acceleration-weight",
                    str(float(args.joint_trajectory_acceleration_weight)),
                    "--joint-trajectory-jerk-weight",
                    str(float(args.joint_trajectory_jerk_weight)),
                    "--joint-trajectory-reverse-weight",
                    str(float(args.joint_trajectory_reverse_weight)),
                ]
            )
        command.append(
            "--prior-use-soft-weights"
            if bool(args.prior_use_soft_weights)
            else "--no-prior-use-soft-weights"
        )
        if bool(getattr(args, "force_static_retrain", False)):
            command.append("--force-static-train")
        denoise_targets = str(getattr(args, "comparison_denoise_target_frames", "") or "").strip()
        if denoise_targets:
            command.extend(["--denoise-train-targets", denoise_targets, "--denoise-eval-targets", denoise_targets])
        if bool(getattr(args, "use_sam3_feature_field", False)):
            command.extend(
                [
                    "--ensure-sam3-features",
                    "--train-sam3-feature-field",
                    "--sam3-feature-dim",
                    str(int(args.sam3_feature_dim)),
                ]
            )
            if str(getattr(args, "sam3_feature_root", "") or "").strip():
                command.extend(["--sam3-feature-root", str(Path(args.sam3_feature_root).expanduser().resolve())])
        if bool(getattr(args, "use_sam3_affinity_field", False)):
            command.extend(
                [
                    "--ensure-sam3-affinity",
                    "--sam3-affinity-iters",
                    str(int(args.sam3_affinity_iters)),
                    "--sam3-affinity-dim",
                    str(int(args.sam3_affinity_dim)),
                ]
            )
            if str(getattr(args, "sam3_mask_root", "") or "").strip():
                command.extend(["--sam3-mask-root", str(Path(args.sam3_mask_root).expanduser().resolve())])
        endpoint_model = str(getattr(args, "endpoint_model_path", "") or "").strip()
        if endpoint_model:
            endpoint_path = Path(endpoint_model).expanduser()
            if endpoint_path.suffix.lower() == ".ply":
                command.extend(["--endpoint-ply", str(endpoint_path.resolve())])
            else:
                command.extend(["--endpoint-model-dir", str(endpoint_path.resolve())])
        if bool(args.dynamic_motion):
            command.append("--run-killing")
        if bool(args.wandb):
            command.append("--killing-wandb")
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parent)
        sys.exit(completed.returncode)
    args.save_iterations.append(args.iterations)

    safe_state(args.quiet)
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        model.extract(args),
        optimization.extract(args),
        pipeline.extract(args),
        args,
        args.start_checkpoint,
        args.save_iterations,
        args.checkpoint_iterations,
        args.test_iterations,
    )


if __name__ == "__main__":
    main()
