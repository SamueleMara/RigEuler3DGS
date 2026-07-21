#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Adaptive Eulerian-motion extension: this keeps the original CUDA renderer and
# static 3DGS optimization path intact. When --dynamic_motion is enabled,
# canonical Gaussians are transported by a world-fixed Eulerian velocity field
# before being passed to the standard renderer.
#

import os
import json
import subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import (
    l1_loss,
    ssim,
    semantic_feature_field_loss,
    dino_feature_field_loss,
)
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, add_dynamic_args, apply_dynamic_defaults
from utils.dynamic_loss_utils import (
    get_camera_timestamp,
    static_velocity_loss,
    dynamic_sparsity_loss,
    dynamic_binarization_loss,
    cycle_consistency_loss,
    dino_weighted_rigidity_loss,
    compute_cell_incoherence_scores,
    dynamic_motion_target_loss,
    windowed_dynamic_motion_target_loss,  # deprecated: kept for compatibility, not used by default
    model_based_dynamic_motion_target_loss,
    motion_aware_dynamic_label_coherence_loss,
    motion_contrastive_dynamic_label_loss,
    rigid_model_contrastive_dynamic_label_loss,
    cell_dynamic_label_coherence_loss,
    global_twist_velocity_loss,
    global_twist_residual_scores,
    joint_single_rigid_losses,
)

DINOv3_DEFAULT_HF_LOCAL_DIR = os.environ.get(
    "DINOV3_HF_LOCAL_DIR",
    str(Path.home() / ".cache" / "huggingface" / "dinov3-vitl16-pretrain-lvd1689m"),
)
DINOv3_DEFAULT_HF_MODEL = os.environ.get(
    "DINOV3_HF_MODEL",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
)
DINOv3_DEFAULT_BACKEND = os.environ.get("DINOV3_BACKEND", "transformers")
DINOv3_DEFAULT_IMAGE_SIZE = int(os.environ.get("DINOV3_IMAGE_SIZE", "512"))

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False








def _wandb_safe_config(*namespaces):
    """Build a flat W&B config while avoiding accidental secret logging."""
    blocked = ("key", "token", "secret", "password", "credential")
    cfg = {}
    for ns in namespaces:
        if ns is None:
            continue
        for k, v in vars(ns).items():
            lk = str(k).lower()
            if any(b in lk for b in blocked):
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                cfg[k] = v
            elif isinstance(v, (list, tuple)):
                cfg[k] = list(v)
            else:
                cfg[k] = str(v)
    return cfg


def setup_wandb(dataset, opt, pipe, dyn):
    """Initialize W&B logging if --wandb is active.

    Authentication is intentionally not handled here. Use either:
        wandb login
    or:
        export WANDB_API_KEY=...
    outside the script. Do not hardcode API keys in this repository.
    """
    if not getattr(dyn, "wandb", False):
        return None
    if not WANDB_AVAILABLE:
        print("[W&B] wandb package not installed. Run: pip install wandb")
        return None

    mode = str(getattr(dyn, "wandb_mode", "online") or "online")
    if mode == "disabled":
        return None

    run_name = str(getattr(dyn, "wandb_name", "") or "").strip()
    if not run_name:
        run_name = Path(str(dataset.model_path)).name

    tags = []
    raw_tags = str(getattr(dyn, "wandb_tags", "") or "").strip()
    if raw_tags:
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

    try:
        run = wandb.init(
            project=str(getattr(dyn, "wandb_project", "RigEuler3DGS") or "RigEuler3DGS"),
            entity=str(getattr(dyn, "wandb_entity", "") or None) or None,
            name=run_name,
            dir=str(dataset.model_path),
            mode=mode,
            tags=tags,
            config=_wandb_safe_config(dataset, opt, pipe, dyn),
        )
        if run is not None:
            try:
                wandb.define_metric("iteration")
                wandb.define_metric("loss/static/*", step_metric="iteration")
                wandb.define_metric("loss/dynamic/*", step_metric="iteration")
                wandb.define_metric("diagnostic/dynamic/*", step_metric="iteration")
                wandb.define_metric("diagnostic/grid/*", step_metric="iteration")
                wandb.define_metric("loss/validation/*", step_metric="iteration")
            except Exception as exc:
                print(f"[W&B] metric definition failed, continuing: {exc}")
            print(f"[W&B] Logging run: project={run.project}, name={run.name}")
        return run
    except Exception as exc:
        print(f"[W&B] init failed: {exc}")
        return None



def sync_feature_field_lrs_to_opt(opt, dyn):
    """GaussianModel.training_setup only receives opt; copy aux LR fields there."""
    for name in ("semantic_feature_lr", "dino_feature_lr"):
        if hasattr(dyn, name):
            setattr(opt, name, getattr(dyn, name))


def _dynamic_foreground_mask_for_losses(gaussians):
    if not hasattr(gaussians, "get_dynamic_foreground_mask"):
        return None
    mask = gaussians.get_dynamic_foreground_mask
    if mask is None or not torch.is_tensor(mask) or mask.numel() != gaussians.get_canonical_xyz.shape[0]:
        return None
    return mask.bool()


def wandb_log_training(wb_run, iteration, stage, euler_iter, loss, Ll1, Ll1depth, dynamic_stats, scene, gaussians, dyn, elapsed_ms):
    """Log only losses to W&B, separated by static and dynamic stage.

    No speed, cell count, dynamic-probability, PSNR, timing, artifact, or scene
    diagnostics are logged here. The namespace is intentionally small:

        loss/static/*   for canonical 3DGS stage
        loss/dynamic/*  for Eulerian velocity-field stage

    Dynamic auxiliary losses are logged under loss/dynamic/<loss_name>.
    Values in dynamic_stats are the raw auxiliary losses as computed by
    add_dynamic_losses(); the total entry is the final optimized scalar.
    """
    if wb_run is None:
        return
    interval = max(1, int(getattr(dyn, "wandb_log_interval", 10)))
    if iteration % interval != 0 and iteration != int(getattr(dyn, "static_train_iters", -1)):
        return

    total = float(loss.detach().item() if hasattr(loss, "detach") else loss)
    l1 = float(Ll1.detach().item() if hasattr(Ll1, "detach") else Ll1)
    depth = float(Ll1depth)

    data = {"iteration": int(iteration)}
    if str(stage) == "static":
        data.update(
            {
                "loss/static/total": total,
                "loss/static/l1": l1,
                "loss/static/depth": depth,
            }
        )
    else:
        data.update(
            {
                "loss/dynamic/total": total,
                "loss/dynamic/l1": l1,
                "loss/dynamic/depth": depth,
                "loss/dynamic/euler_iter": int(euler_iter),
            }
        )
        for k, v in dynamic_stats.items():
            sk = str(k)
            # Keep actual losses in loss/dynamic.  Auxiliary counters/residual
            # diagnostics are streamed separately so W&B plots stay readable.
            if sk.startswith("L_"):
                data["loss/dynamic/" + sk] = float(v)
            else:
                data["diagnostic/dynamic/" + sk] = float(v)

    try:
        wb_run.log(data, step=int(iteration))
    except Exception as exc:
        print(f"[W&B] loss log failed at iter {iteration}: {exc}")

def wandb_log_validation(wb_run, iteration, split_name, l1_value, psnr_value):
    """Log only validation loss to W&B; PSNR is intentionally not logged."""
    if wb_run is None:
        return
    try:
        wb_run.log(
            {
                "iteration": int(iteration),
                f"loss/validation/{split_name}_l1": float(l1_value),
            },
            step=int(iteration),
        )
    except Exception as exc:
        print(f"[W&B] validation loss log failed at iter {iteration}: {exc}")

def wandb_log_grid_update(wb_run, iteration, grid):
    if wb_run is None or grid is None:
        return
    info = getattr(grid, "last_update_info", None) or {}
    data = {
        "iteration": int(iteration),
        "diagnostic/grid/active_cells": int(grid.active_mask.sum().item()) if hasattr(grid, "active_mask") else int(grid.num_cells),
        "diagnostic/grid/total_cells": int(grid.num_cells),
        "diagnostic/grid/split_count": float(info.get("split_count", 0)),
        "diagnostic/grid/new_child_count": float(info.get("new_child_count", 0)),
        "diagnostic/grid/merge_count": float(info.get("merge_count", 0)),
        "diagnostic/grid/current_score_max": float(info.get("current_score_max", 0.0)),
        "diagnostic/grid/decision_score_max": float(info.get("decision_score_max", 0.0)),
        "diagnostic/grid/split_cutoff": float(info.get("split_cutoff", 0.0)),
        "diagnostic/grid/ema_updates": float(info.get("score_ema_updates", 0)),
    }
    try:
        wb_run.log(data, step=int(iteration))
    except Exception as exc:
        print(f"[W&B] grid-update log failed at iter {iteration}: {exc}")


def wandb_log_artifacts(wb_run, model_path, dyn):
    """No-op: W&B integration is intentionally loss-only."""
    return

def dynamic_stage(iteration, dyn):
    """Return the staged dynamic-training phase.

    static:
        vanilla canonical 3DGS stage.
    discovery:
        first dynamic phase; discover static vs dynamic Gaussians using RGB/depth,
        static velocity suppression, sparse/binary logits, and a weak speed target.
    motion:
        second dynamic phase; understand the moving part motion with the Eulerian
        twist prior, cycle consistency, feature supervision, and adaptive-grid
        refinement.  No local SE(3), global SE(3), speed, rollout, temporal, or
        acceleration losses are used.
    """
    if not dyn.dynamic_motion:
        return "static", 0
    static_iters = int(getattr(dyn, "static_train_iters", 30000))
    if iteration <= static_iters:
        return "static", 0
    euler_iter = iteration - static_iters
    if euler_iter >= int(getattr(dyn, "finetune_from_iter", 2000000000)) and getattr(dyn, "allow_dynamic_gaussian_finetune", False):
        return "finetune", euler_iter
    if bool(getattr(dyn, "joint_single_rigid_training", False)):
        return "joint", euler_iter
    motion_from = int(getattr(dyn, "motion_understanding_from_iter", -1))
    if motion_from < 0:
        motion_from = int(getattr(dyn, "dynamic_discovery_iters", 1000))
    if euler_iter >= motion_from:
        return "motion", euler_iter
    return "discovery", euler_iter

def compute_base_render_loss(image, gt_image, opt):
    Ll1 = l1_loss(image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
    return loss, Ll1, ssim_value




def _dino_root_for_train(dataset, dyn):
    root = _dynamic_root_for_train(dataset, dyn)
    return root / getattr(dyn, "dinov3_root", "dinov3")


def _find_first_dino_feature_map(dataset, dyn):
    dino_root = _dino_root_for_train(dataset, dyn)
    if not dino_root.exists():
        return None
    for pat in ("cam_*/*.pt", "*.pt"):
        found = sorted(dino_root.glob(pat))
        if found:
            return found[0]
    return None


def _read_dino_feature_tensor(path):
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        for key in ("features", "dino", "dino_feat", "feature", "feat"):
            if key in obj:
                obj = obj[key]
                break
    if not torch.is_tensor(obj):
        raise RuntimeError(f"DINO feature file has unsupported object type: {type(obj)!r}")
    feat = obj.float()
    if feat.ndim == 4:
        feat = feat[0]
    if feat.ndim != 3:
        raise RuntimeError(f"DINO feature map must be [C,H,W] or [1,C,H,W], got {tuple(feat.shape)}")
    # If stored as H,W,C, move channels first.
    if feat.shape[0] > feat.shape[-1] and feat.shape[-1] <= 4096:
        # Ambiguous; DINO maps are usually C,H,W with C large and H/W small.
        pass
    elif feat.shape[-1] > feat.shape[0] and feat.shape[0] <= 256 and feat.shape[1] <= 256:
        feat = feat.permute(2, 0, 1).contiguous()
    return feat



def _infer_dino_feature_dim(dataset, dyn):
    first = _find_first_dino_feature_map(dataset, dyn)
    if first is None:
        return 0
    try:
        feat = _read_dino_feature_tensor(first)
    except Exception as exc:
        print(f"[DINO feature field] Could not infer feature dim from {first}: {exc}")
        return 0
    src_dim = int(feat.shape[0])
    requested = int(getattr(dyn, "dino_feature_field_dim", 32))
    return src_dim if requested <= 0 else min(src_dim, requested)


def init_gaussian_aux_feature_fields(gaussians, dataset, dyn):
    """Initialize semantic and DINO per-Gaussian fields before optimizer setup."""
    semantic_on = bool(getattr(dyn, "use_semantic_feature_field", False))
    dino_on = bool(getattr(dyn, "use_dino_feature_field", False))
    dino_dim = _infer_dino_feature_dim(dataset, dyn) if dino_on else 0
    if dino_on and dino_dim <= 0:
        print("[DINO feature field] No dense DINO maps found/readable; DINO field disabled.")
        setattr(dyn, "use_dino_feature_field", False)
    if hasattr(gaussians, "init_semantic_and_feature_fields"):
        gaussians.init_semantic_and_feature_fields(
            semantic=semantic_on,
            dino_dim=dino_dim,
            init_dino_random=True,
        )
        if dino_dim > 0:
            dyn.dino_feature_field_dim = dino_dim


def init_dino_feature_field(gaussians, dataset, dyn):
    """Compatibility wrapper.

    The field is now initialized by init_gaussian_aux_feature_fields() before
    training_setup(), so it is part of the normal Gaussian optimizer and follows
    densification/pruning. This function intentionally does not create a
    separate optimizer anymore.
    """
    init_gaussian_aux_feature_fields(gaussians, dataset, dyn)


def add_dynamic_losses(loss, photometric_loss, gaussians, iteration, euler_iter, dyn, stage):
    """Add staged dynamic losses.

    Discovery phase: learn the static/dynamic split.
        - L_static on low-confidence/static points
        - L_dyn_sparse, L_dyn_bin
        - L_dyn_motion_target only during warmup

    Motion-understanding phase: learn the rigid motion of the moving part.
        - L_global_twist on the Eulerian velocity field
        - L_cycle for reversible integration
        - L_velocity_rigid for local instantaneous rigidity

    Removed from the active objective:
        L_speed, L_rollout, L_velocity_temporal, L_velocity_accel, L_se3,
        L_global_se3.
    """
    stats = {}
    if not dyn.dynamic_motion or not gaussians.dynamic_enabled:
        return loss, stats

    x0_all = gaussians.get_canonical_xyz
    xt_all = gaussians.get_xyz
    vel_all = gaussians.get_current_velocity
    cell_id_all = gaussians.get_cell_id
    dyn_prob_all = gaussians.get_dynamic_probability
    dino_feat_all = getattr(gaussians, "_dino_feature_field", None)
    if dino_feat_all is None:
        dino_feat_all = gaussians.get_dino_features

    fg_mask = _dynamic_foreground_mask_for_losses(gaussians)
    if bool(getattr(dyn, "joint_single_rigid_training", False)):
        if fg_mask is None:
            raise RuntimeError(
                "Joint rigid training requires a valid semantic foreground mask; "
                "refusing to estimate omega,u from an unmasked scene."
            )
        part_field_check = getattr(gaussians, "joint_part_field", None)
        if part_field_check is None:
            raise RuntimeError("JointPartField is missing during joint rigid training")
        expected_ids = torch.where(fg_mask)[0].to(part_field_check.gaussian_ids.device)
        actual_ids = torch.sort(part_field_check.gaussian_ids.reshape(-1)).values
        expected_ids = torch.sort(expected_ids.reshape(-1)).values
        if not torch.equal(actual_ids, expected_ids):
            leaked = sorted(set(actual_ids.detach().cpu().tolist()) - set(expected_ids.detach().cpu().tolist()))
            raise RuntimeError(
                "Background contamination detected before rigid fitting. "
                f"Leaked Gaussian IDs: {leaked[:16]}"
            )
        stats["joint_background_points_in_rigid_estimation"] = 0.0
        stats["joint_foreground_ids_verified"] = float(actual_ids.numel())

    if fg_mask is not None:
        fg_count = int(fg_mask.sum().detach().item())
        stats["dynamic_foreground_points"] = float(fg_count)
        stats["dynamic_foreground_fraction"] = float(fg_count) / max(1.0, float(fg_mask.numel()))
        stats["dynamic_background_points"] = float(fg_mask.numel() - fg_count)
        if fg_count <= 0:
            return loss, stats
        x0 = x0_all[fg_mask]
        xt = xt_all[fg_mask]
        vel = vel_all[fg_mask] if vel_all is not None else None
        cell_id = cell_id_all[fg_mask] if cell_id_all is not None and torch.is_tensor(cell_id_all) and cell_id_all.numel() == fg_mask.numel() else cell_id_all
        dyn_prob = dyn_prob_all[fg_mask] if dyn_prob_all is not None and dyn_prob_all.numel() == fg_mask.numel() else dyn_prob_all
        dino_feat = dino_feat_all[fg_mask] if dino_feat_all is not None and torch.is_tensor(dino_feat_all) and dino_feat_all.shape[0] == fg_mask.numel() else dino_feat_all
    else:
        x0, xt, vel, cell_id, dyn_prob, dino_feat = x0_all, xt_all, vel_all, cell_id_all, dyn_prob_all, dino_feat_all

    joint_cluster_all = getattr(gaussians, "get_joint_cluster_id", None)
    if joint_cluster_all is not None and torch.is_tensor(joint_cluster_all):
        if fg_mask is not None and joint_cluster_all.numel() == fg_mask.numel():
            joint_cluster = joint_cluster_all[fg_mask]
        else:
            joint_cluster = joint_cluster_all
    else:
        joint_cluster = None

    # Joint optimization is active at every dynamic iteration. The label field,
    # cosine clustering, proposal field, and explicit omega(t),u(t) are all
    # updated together with deliberately separated gradient routes.
    if bool(getattr(dyn, "joint_single_rigid_training", False)):
        rigid_all = getattr(gaussians, "get_current_rigid_velocity", None)
        proposal_all = getattr(gaussians, "get_current_proposal_velocity", None)
        if fg_mask is not None:
            rigid = rigid_all[fg_mask] if rigid_all is not None else None
            proposal = proposal_all[fg_mask] if proposal_all is not None else None
        else:
            rigid, proposal = rigid_all, proposal_all
        current_t = float(gaussians._current_time if gaussians._current_time is not None else 0.0)
        if rigid is None or proposal is None or dyn_prob is None:
            raise RuntimeError("Foreground rigid/proposal/assignment tensors are missing")
        expected_joint_n = int(fg_mask.sum().item()) if fg_mask is not None else int(x0.shape[0])
        if not (rigid.shape[0] == proposal.shape[0] == dyn_prob.reshape(-1).shape[0] == expected_joint_n):
            raise RuntimeError(
                "Rigid fitting tensor mismatch: expected semantic foreground only, "
                f"got rigid={rigid.shape[0]}, proposal={proposal.shape[0]}, "
                f"p={dyn_prob.reshape(-1).shape[0]}, expected={expected_joint_n}"
            )
        joint_losses, joint_info = joint_single_rigid_losses(
            total_velocity=vel,
            rigid_velocity=rigid,
            proposal_velocity=proposal,
            dynamic_prob=dyn_prob,
            part_field=getattr(gaussians, "joint_part_field", None),
            rigid_twist_field=getattr(gaussians, "rigid_twist_field", None),
            current_time=current_t,
            robust_delta=float(getattr(dyn, "joint_robust_delta", 0.05)),
            label_temperature=float(getattr(dyn, "joint_label_temperature", -1.0)),
            label_temperature_quantile=float(getattr(dyn, "joint_label_temperature_quantile", 0.50)),
            label_temperature_min=float(getattr(dyn, "joint_label_temperature_min", 1e-3)),
            label_temperature_max=float(getattr(dyn, "joint_label_temperature_max", 5e-2)),
            confident_static_threshold=float(getattr(dyn, "joint_confident_static_threshold", 0.20)),
            confident_dynamic_threshold=float(getattr(dyn, "joint_confident_dynamic_threshold", 0.80)),
            twist_dt=float(getattr(dyn, "joint_twist_smooth_dt", 0.02)),
            field_pair_samples=int(getattr(dyn, "joint_field_pair_samples", 1024)),
            field_different_margin=float(getattr(dyn, "joint_field_different_margin", 0.20)),
            minimum_part_fraction=float(getattr(dyn, "joint_minimum_part_fraction", 0.02)),
        )
        weighted = {
            "twist_fit": float(getattr(dyn, "lambda_joint_twist_fit", 0.10)),
            "proposal_rigid": float(getattr(dyn, "lambda_joint_proposal_rigid", 0.05)),
            "proposal_static": float(getattr(dyn, "lambda_joint_proposal_static", 0.02)),
            "label": float(getattr(dyn, "lambda_joint_label", 0.05)),
            "field_compact": float(getattr(dyn, "lambda_joint_field_compact", 0.02)),
            "field_pairwise": float(getattr(dyn, "lambda_joint_field_pairwise", 0.01)),
            "prototype_separation": float(getattr(dyn, "lambda_joint_prototype_separation", 0.01)),
            "assignment_entropy": float(getattr(dyn, "lambda_joint_assignment_entropy", 0.002)),
            "part_occupancy": float(getattr(dyn, "lambda_joint_part_occupancy", 0.02)),
            "twist_smooth": float(getattr(dyn, "lambda_joint_twist_smooth", 0.01)),
        }
        for name, value in joint_losses.items():
            w = weighted.get(name, 0.0)
            if w > 0:
                loss = loss + w * value
            stats["L_joint_" + name] = float(value.detach().item())
            stats["W_joint_" + name] = float(w)
        stats["phase_joint"] = 1.0
        stats["joint_rigid_handover"] = float(getattr(gaussians, "joint_rigid_handover", 0.0))
        for key, value in getattr(gaussians, "_last_joint_occupancy_info", {}).items():
            if isinstance(value, (int, float)):
                stats["joint_hard_occupancy_" + str(key)] = float(value)
        for key, value in joint_info.items():
            if isinstance(value, (int, float)):
                stats[str(key)] = float(value)

        if float(getattr(dyn, "lambda_cycle", 0.0)) > 0:
            x_back, _ = gaussians.integrate_positions(
                xt, current_t, 0.0,
                steps=int(getattr(dyn, "motion_integration_steps", 4)),
                gaussian_cell_id=cell_id,
            )
            Lcycle = cycle_consistency_loss(x0, x_back)
            loss = loss + float(dyn.lambda_cycle) * Lcycle
            stats["L_cycle"] = float(Lcycle.detach().item())
        return loss, stats

    motion_from = int(getattr(dyn, "motion_understanding_from_iter", -1))
    if motion_from < 0:
        motion_from = int(getattr(dyn, "dynamic_discovery_iters", 1000))
    in_discovery = euler_iter < motion_from
    in_motion = not in_discovery
    stats["phase_discovery"] = float(in_discovery)
    stats["phase_motion"] = float(in_motion)

    # --- Phase 1: dynamic/static discerning ---------------------------------
    if in_discovery:
        if float(getattr(dyn, "lambda_static", 0.0)) > 0:
            Lstatic = static_velocity_loss(vel, dyn_prob)
            loss = loss + float(dyn.lambda_static) * Lstatic
            stats["L_static"] = float(Lstatic.detach().item())

        if dyn_prob is not None:
            if float(getattr(dyn, "lambda_dyn_sparse", 0.0)) > 0:
                Lds = dynamic_sparsity_loss(dyn_prob)
                loss = loss + float(dyn.lambda_dyn_sparse) * Lds
                stats["L_dyn_sparse"] = float(Lds.detach().item())
            if float(getattr(dyn, "lambda_dyn_bin", 0.0)) > 0:
                Ldb = dynamic_binarization_loss(dyn_prob)
                loss = loss + float(dyn.lambda_dyn_bin) * Ldb
                stats["L_dyn_bin"] = float(Ldb.detach().item())
            # The old windowed displacement/speed target is intentionally disabled.
            # For articulated objects it labels high-displacement points as dynamic
            # and hinge-near moving points as static. Dynamic logits are now
            # supervised by the model-based identity-vs-rigid target after rigid
            # fitting starts in the motion phase.
            if float(getattr(dyn, "lambda_dyn_motion_target", 0.0)) > 0:
                stats["deprecated_windowed_motion_target_disabled"] = 1.0

            if float(getattr(dyn, "lambda_motion_aware_dyn_coherence", 0.0)) > 0:
                Lmadc, madc_info = motion_aware_dynamic_label_coherence_loss(
                    dynamic_prob=dyn_prob,
                    x0=x0,
                    motion_field=gaussians.motion_field,
                    grid=gaussians.motion_grid,
                    gaussian_cell_id=cell_id,
                    current_velocity=vel,
                    dino_feat=dino_feat,
                    times=getattr(dyn, "dynamic_target_times", "0,0.25,0.5,0.75,1.0"),
                    integration_steps=int(getattr(dyn, "dynamic_target_integration_steps", getattr(dyn, "motion_integration_steps", 4))),
                    max_points=int(getattr(dyn, "motion_aware_coherence_max_points", 2048)),
                    k=int(getattr(dyn, "motion_aware_coherence_knn", 8)),
                    sigma_x=float(getattr(dyn, "motion_aware_coherence_sigma_x", -1.0)),
                    sigma_motion=float(getattr(dyn, "motion_aware_coherence_sigma_motion", 0.25)),
                    sigma_f=float(getattr(dyn, "motion_aware_coherence_sigma_dino", 0.25)),
                    boundary_weight=float(getattr(dyn, "motion_aware_coherence_boundary_weight", 0.25)),
                    boundary_margin=float(getattr(dyn, "motion_aware_coherence_boundary_margin", 0.50)),
                )
                loss = loss + float(dyn.lambda_motion_aware_dyn_coherence) * Lmadc
                stats["L_motion_aware_dyn_coherence"] = float(Lmadc.detach().item())
                for _k, _v in madc_info.items():
                    if isinstance(_v, (int, float)):
                        stats[str(_k)] = float(_v)

            if float(getattr(dyn, "lambda_motion_contrastive_label", 0.0)) > 0:
                Lcontrast, contrast_info = motion_contrastive_dynamic_label_loss(
                    dynamic_prob=dyn_prob,
                    x0=x0,
                    motion_field=gaussians.motion_field,
                    grid=gaussians.motion_grid,
                    gaussian_cell_id=cell_id,
                    times=getattr(dyn, "motion_contrastive_times", getattr(dyn, "dynamic_target_times", "0,0.25,0.5,0.75,1.0")),
                    integration_steps=int(getattr(dyn, "motion_contrastive_integration_steps", getattr(dyn, "motion_integration_steps", 4))),
                    max_points=int(getattr(dyn, "motion_contrastive_max_points", 2048)),
                    num_pairs=int(getattr(dyn, "motion_contrastive_num_pairs", 32768)),
                    target_quantile=float(getattr(dyn, "motion_contrastive_target_quantile", 0.95)),
                    target_gamma=float(getattr(dyn, "motion_contrastive_target_gamma", 1.0)),
                    loss_type=str(getattr(dyn, "motion_contrastive_loss_type", "smooth_l1")),
                    huber_beta=float(getattr(dyn, "motion_contrastive_huber_beta", 0.10)),
                )
                loss = loss + float(dyn.lambda_motion_contrastive_label) * Lcontrast
                stats["L_motion_contrastive_label"] = float(Lcontrast.detach().item())
                for _k, _v in contrast_info.items():
                    if isinstance(_v, (int, float)):
                        stats[str(_k)] = float(_v)

    # Keep static regularization in motion phase only for very confident static
    # Gaussians. This avoids damping slow, true dynamic Gaussians near the hinge.
    if in_motion and float(getattr(dyn, "lambda_static_motion", 0.0)) > 0 and vel is not None:
        if dyn_prob is None:
            LstaticM = static_velocity_loss(vel, None)
        else:
            thr = float(getattr(dyn, "static_confident_threshold", 0.05))
            mask = (dyn_prob.detach() < thr).float()
            per = (vel ** 2).sum(dim=-1, keepdim=True) * mask
            LstaticM = per.sum() / mask.sum().clamp_min(1.0)
        loss = loss + float(dyn.lambda_static_motion) * LstaticM
        stats["L_static_motion"] = float(LstaticM.detach().item())

    # --- Phase 2: motion understanding --------------------------------------
    if in_motion:
        # L_velocity_rigid removed: pairwise radial-strain is not a reliable
        # rigid-motion formulation for mixed static/dynamic cells. Rigid motion
        # is now learned with L_global_twist.

        if (
            float(getattr(dyn, "lambda_global_twist", 0.0)) > 0
            and euler_iter >= int(getattr(dyn, "global_twist_from_iter", motion_from))
        ):
            LglobalTwist, twist_info = global_twist_velocity_loss(
                x=xt,
                velocity=vel,
                dynamic_prob=dyn_prob,
                num_parts=int(getattr(dyn, "num_moving_parts", 1)),
                dynamic_threshold=float(getattr(dyn, "global_twist_dynamic_threshold", 0.5)),
                min_points=int(getattr(dyn, "global_twist_min_points", 128)),
                max_points=int(getattr(dyn, "global_twist_max_points", 8192)),
                cluster_iters=int(getattr(dyn, "global_twist_cluster_iters", 8)),
                robust_delta=float(getattr(dyn, "global_twist_robust_delta", 0.03)),
                detach_fit=bool(getattr(dyn, "global_twist_detach_fit", True)),
                ridge=float(getattr(dyn, "global_twist_ridge", 1e-6)),
            )
            loss = loss + float(dyn.lambda_global_twist) * LglobalTwist
            stats["L_global_twist"] = float(LglobalTwist.detach().item())
            stats["global_twist_parts_used"] = float(twist_info.get("parts_used", 0))
            stats["global_twist_points_used"] = float(twist_info.get("points_used", 0))
            stats["global_twist_omega_norm_mean"] = float(twist_info.get("omega_norm_mean", 0.0))
            stats["global_twist_residual_mean"] = float(twist_info.get("residual_mean", 0.0))

        if (
            dyn_prob is not None
            and vel is not None
            and float(getattr(dyn, "lambda_rigid_model_contrastive_label", 0.0)) > 0
            and euler_iter >= int(getattr(dyn, "rigid_model_contrastive_from_iter", int(getattr(dyn, "global_twist_from_iter", motion_from)) + int(getattr(dyn, "rigid_model_contrastive_delay_iters", 1000))))
        ):
            LrigCon, rigcon_info = rigid_model_contrastive_dynamic_label_loss(
                x=xt,
                velocity=vel,
                dynamic_prob=dyn_prob,
                num_parts=int(getattr(dyn, "num_moving_parts", 1)),
                dynamic_threshold=float(getattr(dyn, "rigid_model_contrastive_dynamic_threshold", getattr(dyn, "global_twist_dynamic_threshold", 0.5))),
                min_fit_points=int(getattr(dyn, "rigid_model_contrastive_min_fit_points", getattr(dyn, "global_twist_min_points", 128))),
                max_fit_points=int(getattr(dyn, "rigid_model_contrastive_max_fit_points", getattr(dyn, "global_twist_max_points", 8192))),
                max_points=int(getattr(dyn, "rigid_model_contrastive_max_points", 2048)),
                num_pairs=int(getattr(dyn, "rigid_model_contrastive_num_pairs", 32768)),
                cluster_iters=int(getattr(dyn, "global_twist_cluster_iters", 8)),
                ridge=float(getattr(dyn, "global_twist_ridge", 1e-6)),
                temperature=float(getattr(dyn, "rigid_model_contrastive_temperature", -1.0)),
                loss_type=str(getattr(dyn, "rigid_model_contrastive_loss_type", "smooth_l1")),
                huber_beta=float(getattr(dyn, "rigid_model_contrastive_huber_beta", 0.10)),
            )
            loss = loss + float(dyn.lambda_rigid_model_contrastive_label) * LrigCon
            stats["L_rigid_model_contrastive_label"] = float(LrigCon.detach().item())
            for _k, _v in rigcon_info.items():
                if isinstance(_v, (int, float)):
                    stats[str(_k)] = float(_v)

        if (
            dyn_prob is not None
            and float(getattr(dyn, "lambda_motion_contrastive_label_motion", 0.0)) > 0
            and euler_iter >= int(getattr(dyn, "motion_contrastive_from_iter", motion_from))
        ):
            LcontrastM, contrast_info_m = motion_contrastive_dynamic_label_loss(
                dynamic_prob=dyn_prob,
                x0=x0,
                motion_field=gaussians.motion_field,
                grid=gaussians.motion_grid,
                gaussian_cell_id=cell_id,
                times=getattr(dyn, "motion_contrastive_times", getattr(dyn, "dynamic_target_times", "0,0.25,0.5,0.75,1.0")),
                integration_steps=int(getattr(dyn, "motion_contrastive_integration_steps", getattr(dyn, "motion_integration_steps", 4))),
                max_points=int(getattr(dyn, "motion_contrastive_max_points", 2048)),
                num_pairs=int(getattr(dyn, "motion_contrastive_num_pairs", 32768)),
                target_quantile=float(getattr(dyn, "motion_contrastive_target_quantile", 0.95)),
                target_gamma=float(getattr(dyn, "motion_contrastive_target_gamma", 1.0)),
                loss_type=str(getattr(dyn, "motion_contrastive_loss_type", "smooth_l1")),
                huber_beta=float(getattr(dyn, "motion_contrastive_huber_beta", 0.10)),
            )
            loss = loss + float(dyn.lambda_motion_contrastive_label_motion) * LcontrastM
            stats["L_motion_contrastive_label_motion"] = float(LcontrastM.detach().item())
            for _k, _v in contrast_info_m.items():
                if isinstance(_v, (int, float)):
                    stats["motion_" + str(_k)] = float(_v)

        cycle_from = int(getattr(dyn, "cycle_from_iter", motion_from))
        if euler_iter >= cycle_from and float(getattr(dyn, "lambda_cycle", 0.0)) > 0:
            t = float(gaussians._current_time if gaussians._current_time is not None else 0.0)
            x_back, _ = gaussians.integrate_positions(xt, t, 0.0, steps=dyn.motion_integration_steps, gaussian_cell_id=cell_id)
            Lcycle = cycle_consistency_loss(x0, x_back)
            loss = loss + float(dyn.lambda_cycle) * Lcycle
            stats["L_cycle"] = float(Lcycle.detach().item())

        # Model-based dynamic target: identity/null velocity vs fitted moving
        # rigid velocity fields. This replaces the old displacement-amplitude
        # target and can label hinge-near moving points as dynamic.
        model_target_from = int(getattr(dyn, "model_dynamic_target_from_iter", int(getattr(dyn, "global_twist_from_iter", motion_from))))
        target_until = int(getattr(dyn, "dyn_motion_target_until_iter", -1))
        if target_until < 0:
            target_until = int(getattr(dyn, "euler_field_train_iters", euler_iter))
        if (
            dyn_prob is not None
            and vel is not None
            and euler_iter >= model_target_from
            and euler_iter <= target_until
            and float(getattr(dyn, "lambda_model_dynamic_target", 0.0)) > 0
        ):
            LmodelTarget, model_target_info = model_based_dynamic_motion_target_loss(
                dynamic_prob=dyn_prob,
                x=xt,
                velocity=vel,
                num_parts=int(getattr(dyn, "num_moving_parts", 1)),
                dynamic_threshold=float(getattr(dyn, "model_dynamic_target_dynamic_threshold", getattr(dyn, "global_twist_dynamic_threshold", 0.5))),
                min_fit_points=int(getattr(dyn, "model_dynamic_target_min_fit_points", getattr(dyn, "global_twist_min_points", 128))),
                max_fit_points=int(getattr(dyn, "model_dynamic_target_max_fit_points", getattr(dyn, "global_twist_max_points", 8192))),
                max_points=int(getattr(dyn, "model_dynamic_target_max_points", getattr(dyn, "dynamic_target_max_points", 4096))),
                cluster_iters=int(getattr(dyn, "global_twist_cluster_iters", 8)),
                ridge=float(getattr(dyn, "global_twist_ridge", 1e-6)),
                temperature=float(getattr(dyn, "model_dynamic_target_temperature", -1.0)),
                target_power=float(getattr(dyn, "model_dynamic_target_power", 1.0)),
                loss_type=str(getattr(dyn, "model_dynamic_target_loss_type", "bce")),
                huber_beta=float(getattr(dyn, "model_dynamic_target_huber_beta", 0.10)),
                fit_score_mode=str(getattr(dyn, "model_dynamic_target_fit_score_mode", "prob_or_speed")),
            )
            loss = loss + float(dyn.lambda_model_dynamic_target) * LmodelTarget
            stats["L_model_dynamic_target"] = float(LmodelTarget.detach().item())
            for _k, _v in model_target_info.items():
                if isinstance(_v, (int, float)):
                    stats[str(_k)] = float(_v)

        if dyn_prob is not None and float(getattr(dyn, "lambda_motion_aware_dyn_coherence_motion", 0.0)) > 0:
            LmadcM, madc_info_m = motion_aware_dynamic_label_coherence_loss(
                dynamic_prob=dyn_prob,
                x0=x0,
                motion_field=gaussians.motion_field,
                grid=gaussians.motion_grid,
                gaussian_cell_id=cell_id,
                current_velocity=vel,
                dino_feat=dino_feat,
                times=getattr(dyn, "dynamic_target_times", "0,0.25,0.5,0.75,1.0"),
                integration_steps=int(getattr(dyn, "dynamic_target_integration_steps", getattr(dyn, "motion_integration_steps", 4))),
                max_points=int(getattr(dyn, "motion_aware_coherence_max_points", 2048)),
                k=int(getattr(dyn, "motion_aware_coherence_knn", 8)),
                sigma_x=float(getattr(dyn, "motion_aware_coherence_sigma_x", -1.0)),
                sigma_motion=float(getattr(dyn, "motion_aware_coherence_sigma_motion", 0.25)),
                sigma_f=float(getattr(dyn, "motion_aware_coherence_sigma_dino", 0.25)),
            )
            loss = loss + float(dyn.lambda_motion_aware_dyn_coherence_motion) * LmadcM
            stats["L_motion_aware_dyn_coherence_motion"] = float(LmadcM.detach().item())
            for _k, _v in madc_info_m.items():
                if isinstance(_v, (int, float)):
                    stats["motion_" + str(_k)] = float(_v)

        # Optional label cleanup after the split is learned. Default is off; the
        # adaptive grid should expose mixed cells rather than forcing them to
        # share one label prematurely.
        if float(getattr(dyn, "lambda_cell_label_coherence", 0.0)) > 0 and dyn_prob is not None:
            LlabelCell = cell_dynamic_label_coherence_loss(dyn_prob, cell_id, max_cells=dyn.rigid_max_cells)
            loss = loss + float(dyn.lambda_cell_label_coherence) * LlabelCell
            stats["L_cell_label_coherence"] = float(LlabelCell.detach().item())

    return loss, stats

def _per_gaussian_viewspace_gradient_score(viewspace_point_tensor, visibility_filter, num_gaussians, dyn):
    """Return the actual backpropagated screen/viewspace gradient per Gaussian.

    This is NOT a residual proxy. It is the same signal used by original 3DGS
    densification: ||dL / d viewspace_xy||, read from
    ``viewspace_point_tensor.grad`` after ``loss.backward()``.

    Static 3DGS uses this gradient to clone/split Gaussians. Dynamic training
    must keep Gaussian identity fixed, so we use the exact same gradient to
    split the motion-grid cells that control ambiguous/high-loss Gaussians.

    Robustness note: depending on the rasterizer path, ``viewspace_point_tensor``
    can contain either all Gaussians [N, 3] or only the visible subset [Nv, 3].
    Likewise ``visibility_filter`` can be [N], [N, 1], byte/bool, or absent.
    This function maps every supported case back to a full [N] vector.
    """
    device = viewspace_point_tensor.device if viewspace_point_tensor is not None else torch.device("cuda")
    N = int(num_gaussians)
    out = torch.zeros(N, device=device)

    if viewspace_point_tensor is None or getattr(viewspace_point_tensor, "grad", None) is None:
        return out

    grad = viewspace_point_tensor.grad.detach()
    if grad.ndim != 2 or grad.shape[0] == 0:
        return out

    g = torch.norm(grad[:, :2], dim=-1)
    g = torch.where(torch.isfinite(g), g, torch.zeros_like(g))

    vis = None
    if visibility_filter is not None:
        vis = visibility_filter.detach().to(device=device)
        # radii/visibility masks sometimes arrive as [N,1] or non-bool tensors.
        vis = vis.reshape(-1).bool()
        if vis.numel() != N:
            vis = None

    # Case A: gradient was retained for all Gaussians.
    if g.numel() == N:
        out[:] = g
        if vis is not None:
            out[~vis] = 0.0

    # Case B: gradient was retained only for visible Gaussians.  Scatter it back
    # into the full Gaussian indexing using the visibility mask.
    elif vis is not None and int(vis.sum().item()) == int(g.numel()):
        out[vis] = g

    # Fallback: preserve the signal instead of crashing.  This should only be hit
    # by unusual rasterizer paths; it is still better than killing the run.
    else:
        n = min(N, int(g.numel()))
        out[:n] = g[:n]

    nz = out[out > 0]
    if nz.numel() > 8:
        q = float(getattr(dyn, "grid_render_grad_clip_percentile", 95.0)) / 100.0
        q = min(max(q, 0.5), 0.999)
        scale = torch.quantile(nz, q).clamp_min(1e-12)
        out = (out / scale).clamp(0.0, 10.0)

    return out.detach()


def _print_motion_grid_update(iteration, euler_iter, gaussians, scores):
    grid = getattr(gaussians, "motion_grid", None)
    if grid is None:
        return
    info = getattr(grid, "last_update_info", None) or {}
    active = int(grid.active_mask.sum().item()) if hasattr(grid, "active_mask") else int(grid.num_cells)
    msg = (
        f"\n[ITER {iteration}] Adaptive motion-grid update "
        f"(euler_iter={euler_iter}): active={active}/{grid.num_cells}, "
        f"split={info.get('split_count', 0)}, children={info.get('new_child_count', 0)}, "
        f"merged={info.get('merge_count', 0)}, "
        f"cur_max={info.get('current_score_max', float(scores.max().item()) if scores.numel() else 0.0):.6g}, "
        f"ema_max={info.get('decision_score_max', float(scores.max().item()) if scores.numel() else 0.0):.6g}, "
        f"cutoff={info.get('split_cutoff', 0.0):.6g}, "
        f"ema_updates={info.get('score_ema_updates', 0)}"
    )
    print(msg)
    top = info.get("top_cells", [])
    if top:
        pretty = ", ".join([
            f"cell {c['cell_id']}: ema={c['score']:.4g}, cur={c.get('current_score', 0.0):.4g}, n={c['points']}"
            for c in top
        ])
        print(f"[ITER {iteration}] Adaptive-grid top cells: {pretty}")


def maybe_update_motion_grid(
    gaussians,
    dyn,
    iteration,
    euler_iter,
    viewspace_point_tensor=None,
    visibility_filter=None,
    wb_run=None,
):
    if not dyn.dynamic_motion or not gaussians.dynamic_enabled:
        return False
    if not getattr(dyn, "enable_adaptive_grid", False):
        return False
    if euler_iter < int(getattr(dyn, "grid_from_iter", 0)):
        return False
    if int(getattr(dyn, "grid_update_interval", 0)) <= 0 or iteration % int(dyn.grid_update_interval) != 0:
        return False
    if gaussians.motion_grid is None:
        return False

    with torch.no_grad():
        viewspace_grad_all = _per_gaussian_viewspace_gradient_score(
            viewspace_point_tensor=viewspace_point_tensor,
            visibility_filter=visibility_filter,
            num_gaussians=gaussians.get_canonical_xyz.shape[0],
            dyn=dyn,
        )
        dyn_prob_all = gaussians.get_dynamic_probability.detach() if gaussians.get_dynamic_probability is not None else None
        vel_all = gaussians.get_current_velocity.detach() if gaussians.get_current_velocity is not None else None
        x0_all = gaussians.get_canonical_xyz.detach()
        xt_all = gaussians.get_xyz.detach()
        cell_id_all = gaussians.get_cell_id.detach()
        dino_all = (gaussians._dino_feature_field.detach() if hasattr(gaussians, "_dino_feature_field") and gaussians._dino_feature_field is not None else (gaussians.get_dino_features.detach() if gaussians.get_dino_features is not None else None))

        fg_mask = _dynamic_foreground_mask_for_losses(gaussians)
        if fg_mask is not None and fg_mask.numel() == x0_all.shape[0]:
            if int(fg_mask.sum().item()) <= 0:
                return False
            x0_score = x0_all[fg_mask]
            xt_score = xt_all[fg_mask]
            vel_det = vel_all[fg_mask] if vel_all is not None else None
            cell_id_score = cell_id_all[fg_mask]
            dyn_prob_det = dyn_prob_all[fg_mask] if dyn_prob_all is not None else None
            dino_score = dino_all[fg_mask] if dino_all is not None and torch.is_tensor(dino_all) and dino_all.shape[0] == fg_mask.numel() else dino_all
            viewspace_grad = viewspace_grad_all[fg_mask]
        else:
            x0_score = x0_all
            xt_score = xt_all
            vel_det = vel_all
            cell_id_score = cell_id_all
            dyn_prob_det = dyn_prob_all
            dino_score = dino_all
            viewspace_grad = viewspace_grad_all

        twist_residual = None
        if float(getattr(dyn, "grid_lambda_twist_residual", 0.0)) > 0 and vel_det is not None:
            if bool(getattr(dyn, "joint_single_rigid_training", False)) and gaussians.get_current_rigid_velocity is not None:
                rigid_all = gaussians.get_current_rigid_velocity.detach()
                rigid_score = rigid_all[fg_mask] if fg_mask is not None and fg_mask.numel() == rigid_all.shape[0] else rigid_all
                twist_residual = torch.linalg.norm(vel_det - rigid_score, dim=-1)
            else:
                twist_residual = global_twist_residual_scores(
                    x=xt_score,
                    velocity=vel_det,
                    dynamic_prob=dyn_prob_det,
                    num_parts=int(getattr(dyn, "num_moving_parts", 1)),
                    dynamic_threshold=float(getattr(dyn, "global_twist_dynamic_threshold", 0.5)),
                    min_points=int(getattr(dyn, "global_twist_min_points", 128)),
                    max_points=int(getattr(dyn, "global_twist_max_points", 8192)),
                    cluster_iters=int(getattr(dyn, "global_twist_cluster_iters", 8)),
                    ridge=float(getattr(dyn, "global_twist_ridge", 1e-6)),
                )
        scores = compute_cell_incoherence_scores(
            x0=x0_score,
            xt=xt_score,
            velocity=vel_det,
            cell_id=cell_id_score,
            dino_feat=dino_score,
            render_residual_per_gaussian=viewspace_grad,  # actual ||dL/d viewspace_xy||, not a residual proxy
            dynamic_prob=dyn_prob_det,
            twist_residual_per_gaussian=twist_residual,
            num_cells=gaussians.motion_grid.num_cells,
            lambda_vel=float(getattr(dyn, "grid_lambda_vel", 1.0)),
            lambda_disp=float(getattr(dyn, "grid_lambda_disp", 0.5)),
            lambda_dino=float(getattr(dyn, "grid_lambda_dino", 0.1)),
            lambda_img=float(getattr(dyn, "grid_lambda_render_grad", 0.25)),  # historical CLI name; uses real viewspace gradient
            lambda_dyn_entropy=float(getattr(dyn, "grid_lambda_dyn_entropy", 0.25)),
            lambda_dyn_var=float(getattr(dyn, "grid_lambda_dyn_var", 0.25)),
            lambda_twist=float(getattr(dyn, "grid_lambda_twist_residual", 0.0)),
            lambda_strain=float(getattr(dyn, "grid_lambda_strain", 0.0)),
            strain_max_points=int(getattr(dyn, "grid_strain_max_points", 64)),
        )
        changed = gaussians.motion_grid.update_topology(
            scores=scores,
            xyz0=gaussians.get_canonical_xyz.detach(),
            split_threshold=float(getattr(dyn, "grid_split_threshold", 1e-4)),
            split_percentile=float(getattr(dyn, "grid_split_percentile", -1.0)),
            score_ema_decay=float(getattr(dyn, "grid_score_ema_decay", 0.95)),
            merge_threshold=float(getattr(dyn, "grid_merge_threshold", 0.0)),
            max_splits=int(getattr(dyn, "grid_max_splits_per_update", 32)),
            min_points=int(getattr(dyn, "grid_min_points", 16)),
            topk=int(getattr(dyn, "grid_print_topk", 5)),
        )
    _print_motion_grid_update(iteration, euler_iter, gaussians, scores)
    wandb_log_grid_update(wb_run, iteration, gaussians.motion_grid)
    if changed:
        gaussians.rebuild_dynamic_optimizers(dyn)
    return changed



def _dynamic_root_for_train(dataset, dyn):
    explicit = getattr(dyn, "root_scene", "") or getattr(dyn, "dynamic_sequence_root", "")
    if explicit:
        return Path(explicit).expanduser().resolve()
    src = Path(dataset.source_path).expanduser().resolve()
    if src.name == "colmap":
        return src.parent
    if (src / "colmap").exists():
        return src
    return src


def maybe_extract_dinov3_features(dataset, dyn):
    """Ensure dense DINO/DINOv3 feature maps exist when requested.

    This is triggered by either --use_dinov3 or --use_dino_feature_field.  The
    previous code only looked at --use_dinov3, so --use_dino_feature_field could
    disable itself before extraction was even attempted.
    """
    needs_dino = bool(getattr(dyn, "use_dinov3", False)) or bool(getattr(dyn, "use_dino_feature_field", False))
    if not needs_dino:
        return
    root = _dynamic_root_for_train(dataset, dyn)
    dino_root = root / getattr(dyn, "dinov3_root", "dinov3")
    ext = getattr(dyn, "dinov3_feature_ext", ".pt")
    existing = list(dino_root.glob("cam_*/*" + ext)) if dino_root.exists() else []
    if existing:
        print(f"[DINOv3] Found {len(existing)} precomputed feature maps under {dino_root}")
        setattr(dyn, "use_dinov3", True)
        return

    # Feature-field training needs the maps. Try extracting automatically unless
    # the user explicitly disabled extraction by disabling the feature field.
    extract = bool(getattr(dyn, "dinov3_extract_on_start", False)) or bool(getattr(dyn, "use_dino_feature_field", False))
    if not extract:
        print(f"[DINOv3] No feature maps found under {dino_root}. Run tools/extract_dinov3_features.py or pass --extract_dinov3.")
        return

    script = Path(__file__).resolve().parent / "tools" / "extract_dinov3_features.py"
    backend = str(getattr(dyn, "dinov3_backend", DINOv3_DEFAULT_BACKEND) or DINOv3_DEFAULT_BACKEND)
    cmd = [
        sys.executable, str(script),
        "--sequence_root", str(root),
        "--rgb_dir", getattr(dyn, "dynamic_rgb_dir", "rgb"),
        "--out_dir", getattr(dyn, "dinov3_root", "dinov3"),
        "--backend", backend,
        "--image_size", str(getattr(dyn, "dinov3_image_size", DINOv3_DEFAULT_IMAGE_SIZE if backend == "transformers" else 518)),
    ]
    if backend == "transformers":
        cmd.extend(["--hf_model", getattr(dyn, "dinov3_hf_model", DINOv3_DEFAULT_HF_MODEL)])
        hf_local_dir = str(getattr(dyn, "dinov3_hf_local_dir", DINOv3_DEFAULT_HF_LOCAL_DIR) or "").strip()
        if hf_local_dir:
            cmd.extend(["--hf_local_dir", hf_local_dir])
    else:
        cmd.extend([
            "--repo", getattr(dyn, "dinov3_repo", "submodules/dinov3"),
            "--model_name", getattr(dyn, "dinov3_model_name", "dinov3_vitl16"),
        ])
    dinov3_weights = str(getattr(dyn, "dinov3_weights", "") or "").strip()
    if dinov3_weights:
        cmd.extend(["--weights", dinov3_weights])
    dinov3_cache_dir = str(getattr(dyn, "dinov3_cache_dir", "") or "").strip()
    if dinov3_cache_dir:
        cmd.extend(["--cache_dir", dinov3_cache_dir])
    if getattr(dyn, "dinov3_allow_download", False):
        cmd.append("--allow_download")
    if bool(getattr(dyn, "dinov3_full_resolution", True)) and not bool(getattr(dyn, "dinov3_patch_resolution", False)):
        cmd.append("--full_resolution")
    target_dim = int(getattr(dyn, "dinov3_target_dim", getattr(dyn, "dino_feature_field_dim", 32)))
    if target_dim > 0:
        cmd.extend(["--feature_dim", str(target_dim)])
    if getattr(dyn, "dinov3_save_fp32", False):
        cmd.append("--fp32")
    if getattr(dyn, "dinov3_overwrite", False):
        cmd.append("--overwrite")
    print("[DINOv3] Extracting dense feature tensors:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    existing = list(dino_root.glob("cam_*/*" + ext)) if dino_root.exists() else []
    if result.returncode == 0 and existing:
        print(f"[DINOv3] Extraction complete: {len(existing)} feature maps under {dino_root}")
        setattr(dyn, "use_dinov3", True)
        return

    msg = (
        "[DINOv3] Feature extraction failed or produced no maps. "
        "Pass --dinov3_weights /path/to/local_checkpoint.pth, put the checkpoint in "
        "~/.cache/torch/hub/checkpoints, or run tools/extract_dinov3_features.py manually. "
        "Online download is only attempted with --dinov3_allow_download."
    )
    if getattr(dyn, "dinov3_required", False):
        raise RuntimeError(msg)
    print(msg)
    print("[DINOv3] Disabling only DINO feature-field supervision; semantic/static RGB training can continue.")
    setattr(dyn, "use_dinov3", False)
    setattr(dyn, "use_dino_feature_field", False)
    setattr(dyn, "dinov3_extract_on_start", False)
    return



def _collect_track_export_times(scene, dyn):
    cams = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    items = {}
    stride = max(1, int(getattr(dyn, "track_frame_stride", 1)))
    for cam in cams:
        fid = int(getattr(cam, "frame_id", round(float(getattr(cam, "timestamp", 0.0)) * float(getattr(dyn, "fps", 30.0)))))
        if fid % stride != 0:
            continue
        t = float(getattr(cam, "timestamp", fid / max(float(getattr(dyn, "fps", 30.0)), 1e-8)))
        items[fid] = t
    if not items:
        return [], []
    frame_ids = sorted(items.keys())
    times = [items[f] for f in frame_ids]
    return frame_ids, times


def _safe_tensor_stats(x):
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return {}
    y = x.detach().float().reshape(-1)
    finite = torch.isfinite(y)
    if finite.any():
        y = y[finite]
    else:
        return {"numel": int(x.numel()), "finite_fraction": 0.0}
    return {
        "numel": int(x.numel()),
        "finite_fraction": float(finite.float().mean().item()),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "mean": float(y.mean().item()),
        "median": float(torch.quantile(y, 0.5).item()) if y.numel() > 1 else float(y.item()),
        "p95": float(torch.quantile(y, 0.95).item()) if y.numel() > 1 else float(y.item()),
        "p99": float(torch.quantile(y, 0.99).item()) if y.numel() > 1 else float(y.item()),
    }


def _build_track_summary(tracks, dyn, phase_info=None):
    summary = {
        "num_gaussians": int(tracks["positions"].shape[1]) if "positions" in tracks and tracks["positions"].ndim >= 2 else 0,
        "num_frames": int(tracks["positions"].shape[0]) if "positions" in tracks else 0,
        "positions_shape": list(tracks["positions"].shape) if "positions" in tracks else [],
        "velocities_shape": list(tracks["velocities"].shape) if "velocities" in tracks else [],
        "dtype": str(tracks["positions"].dtype) if "positions" in tracks else "",
        "uses_dinov3": bool(getattr(dyn, "use_dinov3", False)),
        "adaptive_grid_enabled": bool(getattr(dyn, "enable_adaptive_grid", False)),
    }
    if phase_info:
        summary.update(phase_info)

    pos = tracks.get("positions", None)
    vel = tracks.get("velocities", None)
    x0 = tracks.get("canonical_positions", None)
    if torch.is_tensor(pos) and pos.numel() > 0:
        summary["position_min"] = [float(v) for v in pos.detach().float().amin(dim=(0, 1)).cpu().tolist()]
        summary["position_max"] = [float(v) for v in pos.detach().float().amax(dim=(0, 1)).cpu().tolist()]
        if torch.is_tensor(x0) and x0.ndim == 2:
            disp = torch.linalg.norm(pos.detach().float() - x0.detach().float().unsqueeze(0), dim=-1)
            max_disp = disp.max(dim=0).values
            summary["max_displacement_stats"] = _safe_tensor_stats(max_disp)
            summary["max_displacement_gt_1e-4"] = int((max_disp > 1e-4).sum().item())
            summary["max_displacement_gt_1e-2"] = int((max_disp > 1e-2).sum().item())
            summary["max_displacement_gt_1e-1"] = int((max_disp > 1e-1).sum().item())
    if torch.is_tensor(vel) and vel.numel() > 0:
        speed = torch.linalg.norm(vel.detach().float(), dim=-1)
        max_speed = speed.max(dim=0).values
        mean_speed = speed.mean(dim=0)
        summary["speed_stats_all_frames"] = _safe_tensor_stats(speed)
        summary["max_speed_stats"] = _safe_tensor_stats(max_speed)
        summary["mean_speed_stats"] = _safe_tensor_stats(mean_speed)
        summary["max_speed_gt_1e-4"] = int((max_speed > 1e-4).sum().item())
        summary["max_speed_gt_1e-2"] = int((max_speed > 1e-2).sum().item())
        summary["max_speed_gt_1e-1"] = int((max_speed > 1e-1).sum().item())

    if "rigid_velocities" in tracks:
        rv = tracks["rigid_velocities"].detach().float()
        summary["rigid_speed_stats"] = _safe_tensor_stats(torch.linalg.norm(rv, dim=-1))
    if "proposal_velocities" in tracks or "residual_velocities" in tracks:
        resv = tracks.get("proposal_velocities", tracks.get("residual_velocities")).detach().float()
        summary["proposal_speed_stats"] = _safe_tensor_stats(torch.linalg.norm(resv, dim=-1))
    if "rigid_omega" in tracks:
        summary["rigid_omega_stats"] = _safe_tensor_stats(torch.linalg.norm(tracks["rigid_omega"].detach().float(), dim=-1))
    if "rigid_translation" in tracks:
        summary["rigid_translation_stats"] = _safe_tensor_stats(torch.linalg.norm(tracks["rigid_translation"].detach().float(), dim=-1))
    if "joint_cluster_id" in tracks:
        cid = tracks["joint_cluster_id"].detach().long()
        summary["joint_current_cluster_count"] = int(cid[cid >= 0].max().item() + 1) if (cid >= 0).any() else 0
        summary["joint_current_cluster_sizes"] = torch.bincount(cid.clamp_min(0).cpu(), minlength=max(1, summary["joint_current_cluster_count"])).tolist()
    if "joint_initial_cluster_id" in tracks:
        cid0 = tracks["joint_initial_cluster_id"].detach().long()
        summary["joint_initial_cluster_count"] = int(cid0[cid0 >= 0].max().item() + 1) if (cid0 >= 0).any() else 0
        summary["joint_initial_cluster_sizes"] = torch.bincount(cid0.clamp_min(0).cpu(), minlength=max(1, summary["joint_initial_cluster_count"])).tolist()
        if "joint_cluster_id" in tracks and cid0.numel() == tracks["joint_cluster_id"].numel():
            summary["joint_assignment_changed_count"] = int((cid0 != tracks["joint_cluster_id"].detach().long()).sum().item())
    if "joint_part_assignments" in tracks:
        assign = tracks["joint_part_assignments"].detach().float()
        summary["joint_part_mass"] = assign.mean(dim=0).cpu().tolist()
        summary["joint_part_confidence_stats"] = _safe_tensor_stats(assign.max(dim=-1).values)
    if "joint_part_prototypes" in tracks:
        proto = torch.nn.functional.normalize(tracks["joint_part_prototypes"].detach().float(), dim=-1)
        summary["joint_part_prototype_cosine"] = proto.matmul(proto.t()).cpu().tolist()
    if "joint_rigid_handover" in tracks:
        h = tracks["joint_rigid_handover"]
        summary["joint_rigid_handover"] = float(h.item()) if torch.is_tensor(h) else float(h)

    if "dynamic_prob" in tracks:
        p = tracks["dynamic_prob"].detach().float().reshape(-1)
        thr = float(getattr(dyn, "track_html_dynamic_threshold", 0.5))
        summary["dynamic_prob_stats"] = _safe_tensor_stats(p)
        summary["dynamic_prob_threshold"] = thr
        summary["dynamic_prob_ge_threshold"] = int((p >= thr).sum().item())
    if "semantic_prob" in tracks:
        sem = tracks["semantic_prob"].detach().float().reshape(-1)
        summary["semantic_prob_stats"] = _safe_tensor_stats(sem)
    if "dynamic_foreground_mask" in tracks:
        fg = tracks["dynamic_foreground_mask"].detach().bool().reshape(-1)
        summary["dynamic_foreground_count"] = int(fg.sum().item())
        summary["dynamic_foreground_fraction"] = float(fg.float().mean().item()) if fg.numel() else 0.0
        summary["dynamic_foreground_threshold"] = float(tracks.get("dynamic_foreground_threshold", torch.tensor(float(getattr(dyn, "semantic_dynamic_threshold", 0.5)))).item())
    return summary


def _phase_debug_tensor_stats(x):
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return {"numel": 0, "finite_fraction": 0.0}
    v = x.detach().float().reshape(-1)
    finite = torch.isfinite(v)
    out = {"numel": int(v.numel()), "finite_fraction": float(finite.float().mean().item()) if v.numel() else 0.0}
    if finite.any():
        vf = v[finite]
        out.update({
            "min": float(vf.min().item()),
            "max": float(vf.max().item()),
            "mean": float(vf.mean().item()),
            "median": float(torch.quantile(vf, 0.50).item()),
            "p05": float(torch.quantile(vf, 0.05).item()),
            "p25": float(torch.quantile(vf, 0.25).item()),
            "p75": float(torch.quantile(vf, 0.75).item()),
            "p95": float(torch.quantile(vf, 0.95).item()),
            "p99": float(torch.quantile(vf, 0.99).item()),
        })
    return out


def _phase_debug_skew_matrix(x):
    z = torch.zeros_like(x[..., 0])
    return torch.stack([
        torch.stack([z, -x[..., 2], x[..., 1]], dim=-1),
        torch.stack([x[..., 2], z, -x[..., 0]], dim=-1),
        torch.stack([-x[..., 1], x[..., 0], z], dim=-1),
    ], dim=-2)


def _phase_debug_fit_weighted_twist(x, velocity, weights=None, ridge=1e-6):
    x = x.detach().float()
    velocity = velocity.detach().float()
    n = int(x.shape[0])
    if n < 4:
        return torch.zeros(3, dtype=x.dtype), torch.zeros(3, dtype=x.dtype)
    if weights is None:
        w = torch.ones(n, dtype=x.dtype)
    else:
        w = weights.detach().float().reshape(-1).clamp_min(0.0)
        if w.numel() != n:
            w = torch.ones(n, dtype=x.dtype)
    sqrt_w = torch.sqrt(w.clamp_min(1e-8))
    A = torch.zeros(n, 3, 6, dtype=x.dtype)
    A[:, :, :3] = -_phase_debug_skew_matrix(x)
    eye = torch.eye(3, dtype=x.dtype).view(1, 3, 3).expand(n, 3, 3)
    A[:, :, 3:] = eye
    A_flat = A.reshape(-1, 6)
    b_flat = velocity.reshape(-1)
    ww = sqrt_w.repeat_interleave(3)
    Aw = A_flat * ww[:, None]
    bw = b_flat * ww
    AtA = Aw.t().matmul(Aw)
    Atb = Aw.t().matmul(bw)
    reg = float(ridge) * torch.eye(6, dtype=x.dtype)
    try:
        xi = torch.linalg.solve(AtA + reg, Atb)
    except RuntimeError:
        xi = torch.linalg.pinv(AtA + reg).matmul(Atb)
    return xi[:3].detach(), xi[3:].detach()


def _phase_debug_twist_predict_velocity(x, omega, trans):
    return torch.cross(omega.reshape(1, 3).expand_as(x), x, dim=-1) + trans.reshape(1, 3)


def _phase_debug_group_stats(name, mask, tensors):
    mask = mask.detach().bool().reshape(-1)
    row = {"name": name, "count": int(mask.sum().item())}
    if not mask.any():
        return row
    for key, val in tensors.items():
        if torch.is_tensor(val) and val.reshape(-1).numel() == mask.numel():
            row[key] = _phase_debug_tensor_stats(val.reshape(-1)[mask])
    return row


def _phase_debug_scatter_mean(values, cell_id, num_cells):
    if values is None or not torch.is_tensor(values) or cell_id is None or num_cells <= 0:
        return torch.empty(0)
    v = values.detach().float().reshape(-1)
    cid = cell_id.detach().long().reshape(-1)
    if v.numel() != cid.numel():
        return torch.empty(0)
    valid = (cid >= 0) & (cid < int(num_cells)) & torch.isfinite(v)
    sums = torch.zeros(num_cells, dtype=torch.float32)
    counts = torch.zeros(num_cells, dtype=torch.float32)
    if valid.any():
        sums.scatter_add_(0, cid[valid], v[valid])
        counts.scatter_add_(0, cid[valid], torch.ones_like(v[valid]))
    return sums / counts.clamp_min(1.0)


def _build_deep_phase_diagnostics_from_tracks(tracks, dyn, max_cells_json=48):
    """Build detached per-Gaussian and per-cell diagnostics for phase snapshots.

    This is diagnostic-only. It does not create or modify losses. It is meant to
    answer why a visually moving part is stuck/static:
      * no early motion seed;
      * too far from high-motion seeds;
      * DINO-disconnected from high-motion seeds;
      * dead/mixed adaptive-grid cell;
      * identity/static residual beats fitted rigid residual.
    """
    if tracks is None or not isinstance(tracks, dict):
        return {}, None, None
    pos = tracks.get("positions", None)
    vel = tracks.get("velocities", None)
    x0 = tracks.get("canonical_positions", None)
    if not (torch.is_tensor(pos) and torch.is_tensor(vel) and torch.is_tensor(x0)):
        return {}, None, None
    if pos.ndim != 3 or vel.ndim != 3 or x0.ndim != 2:
        return {}, None, None

    # Use CPU float32 diagnostics to avoid touching training state/GPU memory.
    pos = pos.detach().float().cpu()
    vel = vel.detach().float().cpu()
    x0 = x0.detach().float().cpu()
    explicit_rigid_vel = tracks.get("rigid_velocities", None)
    explicit_residual_vel = tracks.get("proposal_velocities", tracks.get("residual_velocities", None))
    if torch.is_tensor(explicit_rigid_vel):
        explicit_rigid_vel = explicit_rigid_vel.detach().float().cpu()
    if torch.is_tensor(explicit_residual_vel):
        explicit_residual_vel = explicit_residual_vel.detach().float().cpu()
    T, N = int(pos.shape[0]), int(pos.shape[1])
    if N == 0 or T == 0:
        return {}, None, None

    p = tracks.get("dynamic_prob", torch.zeros(N, 1)).detach().float().cpu().reshape(-1)
    if p.numel() != N:
        p = torch.zeros(N, dtype=torch.float32)
    sem = tracks.get("semantic_prob", torch.zeros(N, 1)).detach().float().cpu().reshape(-1)
    if sem.numel() != N:
        sem = torch.zeros(N, dtype=torch.float32)
    fg = tracks.get("dynamic_foreground_mask", torch.ones(N, dtype=torch.bool)).detach().bool().cpu().reshape(-1)
    if fg.numel() != N:
        fg = torch.ones(N, dtype=torch.bool)
    cid = tracks.get("cell_id", torch.full((N,), -1, dtype=torch.long)).detach().long().cpu().reshape(-1)
    if cid.numel() != N:
        cid = torch.full((N,), -1, dtype=torch.long)
    joint_cluster_id = tracks.get("joint_cluster_id", torch.full((N,), -1, dtype=torch.long))
    joint_cluster_id = joint_cluster_id.detach().long().cpu().reshape(-1) if torch.is_tensor(joint_cluster_id) else torch.full((N,), -1, dtype=torch.long)
    if joint_cluster_id.numel() != N:
        joint_cluster_id = torch.full((N,), -1, dtype=torch.long)
    joint_initial_cluster_id = tracks.get("joint_initial_cluster_id", joint_cluster_id)
    joint_initial_cluster_id = joint_initial_cluster_id.detach().long().cpu().reshape(-1) if torch.is_tensor(joint_initial_cluster_id) else joint_cluster_id.clone()
    if joint_initial_cluster_id.numel() != N:
        joint_initial_cluster_id = joint_cluster_id.clone()
    part_assignments = tracks.get("joint_part_assignments", None)
    if torch.is_tensor(part_assignments) and part_assignments.ndim == 2 and part_assignments.shape[0] == N:
        part_assignments = part_assignments.detach().float().cpu()
        part_confidence = part_assignments.max(dim=-1).values
    else:
        part_assignments = None
        part_confidence = torch.maximum(p, 1.0 - p)
    part_changed_from_initial = (joint_cluster_id != joint_initial_cluster_id) & fg

    speed = torch.linalg.norm(vel, dim=-1)  # transported velocity [T,N]
    mean_speed = speed.mean(dim=0)
    max_speed = speed.max(dim=0).values
    if torch.is_tensor(explicit_residual_vel) and explicit_residual_vel.shape == vel.shape:
        proposal_speed_frames = torch.linalg.norm(explicit_residual_vel, dim=-1)
    else:
        proposal_speed_frames = speed
    mean_proposal_speed = proposal_speed_frames.mean(dim=0)
    max_proposal_speed = proposal_speed_frames.max(dim=0).values
    disp = torch.linalg.norm(pos - x0.unsqueeze(0), dim=-1)
    max_disp = disp.max(dim=0).values

    dyn_thr = float(getattr(dyn, "track_html_dynamic_threshold", 0.5))
    seed_quantile = float(getattr(dyn, "phase_debug_seed_quantile", 0.80))
    seed_min = int(getattr(dyn, "phase_debug_seed_min_points", 128))
    seed_max = int(getattr(dyn, "phase_debug_seed_max_points", 1024))
    ridge = float(getattr(dyn, "phase_debug_twist_ridge", getattr(dyn, "global_twist_ridge", 1e-6)))
    nearest_chunk = int(getattr(dyn, "phase_debug_nearest_chunk", 2048))
    if bool(getattr(dyn, "joint_single_rigid_training", False)):
        temp_arg = float(getattr(dyn, "joint_label_temperature", -1.0))
        temp_quantile = float(getattr(dyn, "joint_label_temperature_quantile", 0.50))
        temp_min = float(getattr(dyn, "joint_label_temperature_min", 1e-3))
        temp_max = float(getattr(dyn, "joint_label_temperature_max", 5e-2))
    else:
        temp_arg = float(getattr(dyn, "phase_debug_q_temperature", -1.0))
        temp_quantile = 0.75
        temp_min = 1e-4
        temp_max = float("inf")

    # Use high motion as seeds, not p_dyn alone, because p_dyn can be
    # untrained/degenerate at end_dynamic_labeling. This tells whether stuck
    # points were ever connected to a genuine motion seed.
    seed_score = max_proposal_speed.clone()
    valid_seed_pool = fg & torch.isfinite(seed_score) & (seed_score > 1e-8)
    seed_mask = torch.zeros(N, dtype=torch.bool)
    if valid_seed_pool.any():
        vals = seed_score[valid_seed_pool]
        q = min(max(seed_quantile, 0.0), 0.999)
        thr = torch.quantile(vals, q)
        seed_mask = valid_seed_pool & (seed_score >= thr)
        if int(seed_mask.sum().item()) < seed_min:
            pool_idx = torch.where(valid_seed_pool)[0]
            k = min(int(pool_idx.numel()), int(seed_min))
            order = torch.argsort(seed_score[pool_idx], descending=True)
            seed_mask[:] = False
            seed_mask[pool_idx[order[:k]]] = True
        if seed_max > 0 and int(seed_mask.sum().item()) > seed_max:
            seed_idx_all = torch.where(seed_mask)[0]
            order = torch.argsort(seed_score[seed_idx_all], descending=True)
            keep = seed_idx_all[order[:seed_max]]
            seed_mask[:] = False
            seed_mask[keep] = True
    seed_idx = torch.where(seed_mask)[0]

    nearest_seed_distance = torch.full((N,), float("inf"), dtype=torch.float32)
    nearest_seed_index = torch.full((N,), -1, dtype=torch.long)
    nearest_seed_dino_similarity = torch.full((N,), float("nan"), dtype=torch.float32)
    if seed_idx.numel() > 0:
        seed_xyz = x0[seed_idx]
        for start in range(0, N, max(1, nearest_chunk)):
            end = min(N, start + max(1, nearest_chunk))
            d = torch.cdist(x0[start:end], seed_xyz)
            md, mi = d.min(dim=1)
            nearest_seed_distance[start:end] = md
            nearest_seed_index[start:end] = seed_idx[mi]
        dino = tracks.get("dino_feature_field", None)
        if torch.is_tensor(dino) and dino.ndim == 2 and dino.shape[0] == N and dino.numel() > 0:
            f = torch.nn.functional.normalize(dino.detach().float().cpu(), dim=-1)
            valid_nn = nearest_seed_index >= 0
            nearest_seed_dino_similarity[valid_nn] = (f[valid_nn] * f[nearest_seed_index[valid_nn]]).sum(dim=-1).clamp(-1.0, 1.0)

    # Joint assignment is trained from the unrestricted proposal field versus
    # the explicit rigid field, not from the already gated transported velocity.
    e_static_frames = proposal_speed_frames.clone()
    e_rigid_frames = torch.zeros_like(proposal_speed_frames)
    omega_norm = torch.zeros(T, dtype=torch.float32)
    trans_norm = torch.zeros(T, dtype=torch.float32)
    using_explicit_rigid = bool(
        torch.is_tensor(explicit_rigid_vel)
        and explicit_rigid_vel.shape == vel.shape
        and torch.is_tensor(explicit_residual_vel)
        and explicit_residual_vel.shape == vel.shape
    )
    if using_explicit_rigid:
        e_rigid_frames = torch.linalg.norm(explicit_residual_vel - explicit_rigid_vel, dim=-1)
        if torch.is_tensor(tracks.get("rigid_omega", None)):
            omega_norm = torch.linalg.norm(tracks["rigid_omega"].detach().float().cpu(), dim=-1)
        if torch.is_tensor(tracks.get("rigid_translation", None)):
            trans_norm = torch.linalg.norm(tracks["rigid_translation"].detach().float().cpu(), dim=-1)
    elif seed_idx.numel() >= 4:
        seed_weights = seed_score[seed_idx].clamp_min(1e-4)
        for ti in range(T):
            proposal_t = explicit_residual_vel[ti] if torch.is_tensor(explicit_residual_vel) and explicit_residual_vel.shape == vel.shape else vel[ti]
            omega, trans = _phase_debug_fit_weighted_twist(pos[ti, seed_idx], proposal_t[seed_idx], weights=seed_weights, ridge=ridge)
            pred = _phase_debug_twist_predict_velocity(pos[ti], omega, trans)
            e_rigid_frames[ti] = torch.linalg.norm(proposal_t - pred, dim=-1)
            omega_norm[ti] = torch.linalg.norm(omega)
            trans_norm[ti] = torch.linalg.norm(trans)
    else:
        e_rigid_frames[:] = float("nan")

    e_static = e_static_frames.mean(dim=0)
    e_rigid = torch.nanmean(e_rigid_frames, dim=0) if torch.isfinite(e_rigid_frames).any() else torch.full((N,), float("nan"))
    residual_diff = e_static - e_rigid
    finite_diff = residual_diff[torch.isfinite(residual_diff)]
    if temp_arg > 0:
        q_temp = torch.tensor(temp_arg, dtype=torch.float32)
    elif finite_diff.numel() > 0:
        q_temp = torch.quantile(finite_diff.abs(), min(max(temp_quantile, 0.0), 1.0))
        q_temp = q_temp.clamp_min(temp_min)
        if torch.isfinite(torch.tensor(temp_max)):
            q_temp = q_temp.clamp_max(temp_max)
    else:
        q_temp = torch.tensor(max(temp_min, 1e-4))
    q_model = torch.sigmoid(residual_diff / q_temp).clamp(0.0, 1.0)

    stuck_speed_thr = float(getattr(dyn, "phase_debug_stuck_speed_threshold", 0.05))
    stuck_disp_thr = float(getattr(dyn, "phase_debug_stuck_disp_threshold", 0.02))
    moving_disp_thr = float(getattr(dyn, "phase_debug_moving_disp_threshold", 0.10))
    moving_speed_thr = float(getattr(dyn, "phase_debug_moving_speed_threshold", 0.10))

    final_static_fg = fg & (p < dyn_thr)
    final_dynamic_fg = fg & (p >= dyn_thr)
    dead_static_fg = final_static_fg & (max_speed < stuck_speed_thr) & (max_disp < stuck_disp_thr)
    moving_but_static = final_static_fg & ((max_speed > moving_speed_thr) | (max_disp > moving_disp_thr))
    model_says_dynamic_but_label_static = final_static_fg & torch.isfinite(q_model) & (q_model >= 0.5)
    close_to_seed_static = final_static_fg & torch.isfinite(nearest_seed_distance) & (nearest_seed_distance <= float(getattr(dyn, "phase_debug_close_seed_distance", 0.10)))

    explicit_rigid_mean_speed = torch.full((N,), float("nan"))
    explicit_residual_mean_speed = torch.full((N,), float("nan"))
    if using_explicit_rigid:
        explicit_rigid_mean_speed = torch.linalg.norm(explicit_rigid_vel, dim=-1).mean(dim=0)
    if torch.is_tensor(explicit_residual_vel) and explicit_residual_vel.shape == vel.shape:
        explicit_residual_mean_speed = torch.linalg.norm(explicit_residual_vel, dim=-1).mean(dim=0)

    per_point = {
        "canonical_xyz": x0,
        "semantic_prob": sem,
        "dynamic_prob": p,
        "dynamic_foreground_mask": fg,
        "cell_id": cid,
        "joint_cluster_id": joint_cluster_id,
        "joint_initial_cluster_id": joint_initial_cluster_id,
        "joint_part_confidence": part_confidence,
        "joint_part_changed_from_initial_mask": part_changed_from_initial,
        "mean_speed": mean_speed,
        "mean_proposal_speed": mean_proposal_speed,
        "max_proposal_speed": max_proposal_speed,
        "explicit_rigid_mean_speed": explicit_rigid_mean_speed,
        "explicit_residual_mean_speed": explicit_residual_mean_speed,
        "max_speed": max_speed,
        "max_displacement": max_disp,
        "seed_score": seed_score,
        "high_motion_seed_mask": seed_mask,
        "nearest_high_motion_seed_distance": nearest_seed_distance,
        "nearest_high_motion_seed_index": nearest_seed_index,
        "nearest_high_motion_seed_dino_similarity": nearest_seed_dino_similarity,
        "e_static_mean_speed": e_static,
        "e_rigid_mean_residual": e_rigid,
        "residual_diff_static_minus_rigid": residual_diff,
        "q_model_identity_vs_rigid": q_model,
        "final_static_foreground_mask": final_static_fg,
        "final_dynamic_foreground_mask": final_dynamic_fg,
        "dead_static_foreground_mask": dead_static_fg,
        "moving_but_static_mask": moving_but_static,
        "model_says_dynamic_but_label_static_mask": model_says_dynamic_but_label_static,
        "close_to_seed_static_mask": close_to_seed_static,
        "omega_norm_per_frame": omega_norm,
        "trans_norm_per_frame": trans_norm,
    }

    if part_assignments is not None:
        per_point["joint_part_assignments"] = part_assignments

    tensors = {
        "semantic_prob": sem,
        "dynamic_prob": p,
        "mean_speed": mean_speed,
        "mean_proposal_speed": mean_proposal_speed,
        "max_proposal_speed": max_proposal_speed,
        "joint_part_confidence": part_confidence,
        "max_speed": max_speed,
        "max_displacement": max_disp,
        "nearest_seed_distance": nearest_seed_distance,
        "nearest_seed_dino_similarity": nearest_seed_dino_similarity,
        "e_static": e_static,
        "e_rigid": e_rigid,
        "residual_diff": residual_diff,
        "q_model": q_model,
    }
    groups = [
        _phase_debug_group_stats("all", torch.ones(N, dtype=torch.bool), tensors),
        _phase_debug_group_stats("foreground", fg, tensors),
        _phase_debug_group_stats("high_motion_seeds", seed_mask, tensors),
        _phase_debug_group_stats("final_dynamic_foreground", final_dynamic_fg, tensors),
        _phase_debug_group_stats("final_static_foreground", final_static_fg, tensors),
        _phase_debug_group_stats("dead_static_foreground", dead_static_fg, tensors),
        _phase_debug_group_stats("moving_but_static", moving_but_static, tensors),
        _phase_debug_group_stats("model_says_dynamic_but_label_static", model_says_dynamic_but_label_static, tensors),
        _phase_debug_group_stats("close_to_seed_static", close_to_seed_static, tensors),
        _phase_debug_group_stats("part_changed_from_initial", part_changed_from_initial, tensors),
        _phase_debug_group_stats("initial_static_slot", fg & (joint_initial_cluster_id == 0), tensors),
        _phase_debug_group_stats("initial_moving_slot", fg & (joint_initial_cluster_id > 0), tensors),
        _phase_debug_group_stats("current_static_slot", fg & (joint_cluster_id == 0), tensors),
        _phase_debug_group_stats("current_moving_slot", fg & (joint_cluster_id > 0), tensors),
    ]

    num_cells = int(cid[cid >= 0].max().item()) + 1 if (cid >= 0).any() else 0
    per_cell = {}
    per_cell_summary_rows = []
    if num_cells > 0:
        counts = torch.bincount(cid[cid >= 0], minlength=num_cells).long()
        per_cell["cell_id"] = torch.arange(num_cells, dtype=torch.long)
        per_cell["count"] = counts
        for name, values in [
            ("foreground_fraction", fg.float()),
            ("dynamic_fraction", (p >= dyn_thr).float()),
            ("semantic_prob_mean", sem),
            ("dynamic_prob_mean", p),
            ("joint_part_confidence_mean", part_confidence),
            ("part_changed_from_initial_fraction", part_changed_from_initial.float()),
            ("mean_speed_mean", mean_speed),
            ("mean_proposal_speed_mean", mean_proposal_speed),
            ("max_proposal_speed_mean", max_proposal_speed),
            ("max_speed_mean", max_speed),
            ("max_displacement_mean", max_disp),
            ("nearest_seed_distance_mean", nearest_seed_distance),
            ("nearest_seed_dino_similarity_mean", nearest_seed_dino_similarity),
            ("e_static_mean", e_static),
            ("e_rigid_mean", e_rigid),
            ("q_model_mean", q_model),
            ("explicit_rigid_mean_speed", explicit_rigid_mean_speed),
            ("explicit_residual_mean_speed", explicit_residual_mean_speed),
            ("seed_fraction", seed_mask.float()),
            ("dead_static_fraction", dead_static_fg.float()),
            ("moving_but_static_fraction", moving_but_static.float()),
            ("model_says_dynamic_but_label_static_fraction", model_says_dynamic_but_label_static.float()),
        ]:
            per_cell[name] = _phase_debug_scatter_mean(values, cid, num_cells)

        # Problem cells: foreground cells with many static/dead points and low motion.
        problem_score = (
            per_cell.get("dead_static_fraction", torch.zeros(num_cells)) * 2.0
            + per_cell.get("moving_but_static_fraction", torch.zeros(num_cells)) * 2.0
            + per_cell.get("model_says_dynamic_but_label_static_fraction", torch.zeros(num_cells)) * 1.5
            + per_cell.get("foreground_fraction", torch.zeros(num_cells)) * 0.5
        ) * counts.float().clamp_min(1.0).log1p()
        topk = min(int(max_cells_json), num_cells)
        top_ids = torch.argsort(problem_score, descending=True)[:topk]
        for c in top_ids.tolist():
            row = {"cell_id": int(c), "count": int(counts[c].item()), "problem_score": float(problem_score[c].item())}
            for k, v in per_cell.items():
                if k in ("cell_id", "count"):
                    continue
                if torch.is_tensor(v) and v.numel() > c:
                    val = v[c]
                    row[k] = None if not torch.isfinite(val) else float(val.item())
            per_cell_summary_rows.append(row)

    summary = {
        "num_gaussians": int(N),
        "num_frames": int(T),
        "dynamic_threshold": float(dyn_thr),
        "foreground_count": int(fg.sum().item()),
        "dynamic_count": int((p >= dyn_thr).sum().item()),
        "high_motion_seed_count": int(seed_mask.sum().item()),
        "seed_quantile": float(seed_quantile),
        "seed_min_points": int(seed_min),
        "seed_max_points": int(seed_max),
        "stuck_speed_threshold": float(stuck_speed_thr),
        "stuck_displacement_threshold": float(stuck_disp_thr),
        "moving_speed_threshold": float(moving_speed_thr),
        "moving_displacement_threshold": float(moving_disp_thr),
        "q_model_temperature": float(q_temp.item()) if torch.is_tensor(q_temp) else float(q_temp),
        "using_explicit_rigid_twist": bool(using_explicit_rigid),
        "joint_initial_cluster_count": int(joint_initial_cluster_id[joint_initial_cluster_id >= 0].max().item() + 1) if (joint_initial_cluster_id >= 0).any() else 0,
        "joint_current_cluster_count": int(joint_cluster_id[joint_cluster_id >= 0].max().item() + 1) if (joint_cluster_id >= 0).any() else 0,
        "joint_part_changed_from_initial_count": int(part_changed_from_initial.sum().item()),
        "joint_part_changed_from_initial_fraction": float(part_changed_from_initial.float().mean().item()),
        "joint_part_confidence_stats": _phase_debug_tensor_stats(part_confidence),
        "joint_part_mass": (part_assignments.mean(dim=0).tolist() if part_assignments is not None else None),
        "joint_part_prototype_cosine": (tracks["joint_part_prototypes"].detach().float().cpu().matmul(tracks["joint_part_prototypes"].detach().float().cpu().t()).tolist() if torch.is_tensor(tracks.get("joint_part_prototypes", None)) else None),
        "joint_rigid_handover": float(tracks.get("joint_rigid_handover", 0.0).item()) if torch.is_tensor(tracks.get("joint_rigid_handover", None)) else float(tracks.get("joint_rigid_handover", 0.0)),
        "explicit_rigid_mean_speed_stats": _phase_debug_tensor_stats(explicit_rigid_mean_speed),
        "proposal_mean_speed_stats": _phase_debug_tensor_stats(explicit_residual_mean_speed),
        "omega_norm_per_frame_stats": _phase_debug_tensor_stats(omega_norm),
        "trans_norm_per_frame_stats": _phase_debug_tensor_stats(trans_norm),
        "group_stats": groups,
        "top_problem_cells": per_cell_summary_rows,
        "interpretation_hints": {
            "dead_static_foreground": "foreground + p_dyn<threshold + near-zero max speed/displacement; these were never activated as moving.",
            "moving_but_static": "foreground + p_dyn<threshold but nontrivial speed/displacement; likely visualization/threshold or label target problem.",
            "model_says_dynamic_but_label_static": "q_model>=0.5 but p_dyn<threshold; label optimization/threshold problem, not motion fitting.",
            "close_to_seed_static": "static-labeled foreground close to high-motion seeds; candidate for missing propagation/connectivity issue.",
            "part_changed_from_initial": "Gaussians whose optimized cosine-field assignment differs from the initial full XYZ+DINO k-medoids assignment.",
            "proposal_vs_rigid": "q_model and residuals use the unrestricted proposal velocity versus the explicit rigid velocity, exactly as the joint label loss does.",
        }
    }
    return summary, per_point, per_cell


def _write_deep_phase_diagnostics(out_dir, tracks, dyn):
    if not bool(getattr(dyn, "phase_debug_deep_diagnostics", True)):
        return
    summary, per_point, per_cell = _build_deep_phase_diagnostics_from_tracks(
        tracks,
        dyn,
        max_cells_json=int(getattr(dyn, "phase_debug_max_problem_cells_json", 48)),
    )
    if not summary:
        return
    out_dir = Path(out_dir)
    with open(out_dir / "deep_diagnostics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if per_point is not None:
        torch.save(per_point, out_dir / "per_point_diagnostics.pt")
    if per_cell is not None:
        torch.save(per_cell, out_dir / "per_cell_diagnostics.pt")

    if bool(getattr(dyn, "phase_debug_write_csv", True)) and per_point is not None:
        try:
            import csv
            cols = [
                "idx", "x", "y", "z", "semantic_prob", "dynamic_prob", "foreground", "cell_id",
                "joint_initial_cluster_id", "joint_cluster_id", "joint_part_confidence", "joint_part_changed_from_initial",
                "mean_speed", "mean_proposal_speed", "max_proposal_speed", "explicit_rigid_mean_speed", "proposal_mean_speed", "max_speed", "max_displacement", "high_motion_seed",
                "nearest_seed_distance", "nearest_seed_index", "nearest_seed_dino_similarity",
                "e_static", "e_rigid", "residual_diff", "q_model",
                "dead_static_foreground", "moving_but_static", "model_says_dynamic_but_label_static", "close_to_seed_static",
            ]
            with open(out_dir / "per_point_diagnostics.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                N = int(per_point["canonical_xyz"].shape[0])
                for i in range(N):
                    xyz = per_point["canonical_xyz"][i]
                    writer.writerow([
                        i, float(xyz[0]), float(xyz[1]), float(xyz[2]),
                        float(per_point["semantic_prob"][i]),
                        float(per_point["dynamic_prob"][i]),
                        int(bool(per_point["dynamic_foreground_mask"][i])),
                        int(per_point["cell_id"][i]),
                        int(per_point["joint_initial_cluster_id"][i]),
                        int(per_point["joint_cluster_id"][i]),
                        float(per_point["joint_part_confidence"][i]),
                        int(bool(per_point["joint_part_changed_from_initial_mask"][i])),
                        float(per_point["mean_speed"][i]),
                        float(per_point["mean_proposal_speed"][i]),
                        float(per_point["max_proposal_speed"][i]),
                        float(per_point["explicit_rigid_mean_speed"][i]) if torch.isfinite(per_point["explicit_rigid_mean_speed"][i]) else "nan",
                        float(per_point["explicit_residual_mean_speed"][i]) if torch.isfinite(per_point["explicit_residual_mean_speed"][i]) else "nan",
                        float(per_point["max_speed"][i]),
                        float(per_point["max_displacement"][i]),
                        int(bool(per_point["high_motion_seed_mask"][i])),
                        float(per_point["nearest_high_motion_seed_distance"][i]),
                        int(per_point["nearest_high_motion_seed_index"][i]),
                        float(per_point["nearest_high_motion_seed_dino_similarity"][i]) if torch.isfinite(per_point["nearest_high_motion_seed_dino_similarity"][i]) else "nan",
                        float(per_point["e_static_mean_speed"][i]),
                        float(per_point["e_rigid_mean_residual"][i]) if torch.isfinite(per_point["e_rigid_mean_residual"][i]) else "nan",
                        float(per_point["residual_diff_static_minus_rigid"][i]) if torch.isfinite(per_point["residual_diff_static_minus_rigid"][i]) else "nan",
                        float(per_point["q_model_identity_vs_rigid"][i]) if torch.isfinite(per_point["q_model_identity_vs_rigid"][i]) else "nan",
                        int(bool(per_point["dead_static_foreground_mask"][i])),
                        int(bool(per_point["moving_but_static_mask"][i])),
                        int(bool(per_point["model_says_dynamic_but_label_static_mask"][i])),
                        int(bool(per_point["close_to_seed_static_mask"][i])),
                    ])
        except Exception as exc:
            print(f"[Dynamic 3DGS] Deep phase CSV failed: {exc}")


def _build_grid_debug_from_tracks(gaussians, tracks, max_cells_json=32):
    grid = getattr(gaussians, "motion_grid", None)
    if grid is None:
        return {}, None

    state = {
        "cell_centers": grid.cell_centers.detach().cpu(),
        "cell_level": grid.cell_level.detach().cpu(),
        "active_mask": grid.active_mask.detach().cpu(),
        "parent_id": grid.parent_id.detach().cpu(),
        "gaussian_cell_id": grid.gaussian_cell_id.detach().cpu(),
        "features": grid.features.detach().cpu(),
        "score_ema": getattr(grid, "score_ema", torch.empty(0, device=grid.device)).detach().cpu(),
        "score_ema_updates": getattr(grid, "score_ema_updates", torch.zeros(1, device=grid.device)).detach().cpu(),
        "last_update_info": getattr(grid, "last_update_info", {}) or {},
        "feature_dim": int(getattr(grid, "feature_dim", 0)),
        "voxel_size": float(getattr(grid, "voxel_size", 0.0)),
        "joint_cluster_id": tracks.get("joint_cluster_id", torch.empty(0, dtype=torch.long)).detach().cpu() if torch.is_tensor(tracks.get("joint_cluster_id", None)) else torch.empty(0, dtype=torch.long),
        "rigid_center": tracks.get("rigid_center", torch.empty(0)).detach().cpu() if torch.is_tensor(tracks.get("rigid_center", None)) else torch.empty(0),
        "rigid_omega": tracks.get("rigid_omega", torch.empty(0)).detach().cpu() if torch.is_tensor(tracks.get("rigid_omega", None)) else torch.empty(0),
        "rigid_translation": tracks.get("rigid_translation", torch.empty(0)).detach().cpu() if torch.is_tensor(tracks.get("rigid_translation", None)) else torch.empty(0),
    }

    summary = {
        "num_cells_total": int(grid.num_cells),
        "num_cells_active": int(grid.active_mask.detach().sum().item()),
        "voxel_size": float(getattr(grid, "voxel_size", 0.0)),
        "feature_dim": int(getattr(grid, "feature_dim", 0)),
        "last_update_info": getattr(grid, "last_update_info", {}) or {},
    }
    levels = grid.cell_level.detach().cpu().long()
    if levels.numel() > 0:
        summary["cell_level_min"] = int(levels.min().item())
        summary["cell_level_max"] = int(levels.max().item())

    cid = tracks.get("cell_id", None)
    if cid is None:
        cid = grid.gaussian_cell_id.detach().cpu()
    else:
        cid = cid.detach().cpu().long().reshape(-1)
    valid = cid >= 0
    N = int(cid.numel())
    num_cells = int(grid.num_cells)
    counts = torch.bincount(cid[valid].clamp(max=max(num_cells - 1, 0)), minlength=num_cells) if num_cells > 0 else torch.empty(0, dtype=torch.long)
    state["per_cell_count"] = counts
    if counts.numel() > 0:
        active_counts = counts[grid.active_mask.detach().cpu().bool()[:counts.numel()]]
        summary["cell_count_stats_active"] = _safe_tensor_stats(active_counts.float())
        summary["empty_active_cells"] = int((active_counts == 0).sum().item())

    # Per-cell aggregates for diagnosing dead/stuck regions.
    per_cell = {"cell_id": torch.arange(num_cells, dtype=torch.long), "count": counts}
    def scatter_mean(values, name):
        if values is None or not torch.is_tensor(values) or values.numel() != N or num_cells <= 0:
            return
        v = values.detach().cpu().float().reshape(-1)
        sums = torch.zeros(num_cells, dtype=torch.float32)
        sums.scatter_add_(0, cid[valid].clamp(max=num_cells - 1), v[valid])
        per_cell[name] = sums / counts.clamp_min(1).float()

    if "dynamic_prob" in tracks:
        scatter_mean(tracks["dynamic_prob"].reshape(-1), "dynamic_prob_mean")
    if "semantic_prob" in tracks:
        scatter_mean(tracks["semantic_prob"].reshape(-1), "semantic_prob_mean")
    if "dynamic_foreground_mask" in tracks:
        scatter_mean(tracks["dynamic_foreground_mask"].float().reshape(-1), "foreground_fraction")
    if "velocities" in tracks and tracks["velocities"].numel() > 0:
        speed = torch.linalg.norm(tracks["velocities"].float(), dim=-1)
        scatter_mean(speed.mean(dim=0), "mean_speed_mean")
        scatter_mean(speed.max(dim=0).values, "max_speed_mean")
    if "positions" in tracks and "canonical_positions" in tracks and tracks["positions"].numel() > 0:
        disp = torch.linalg.norm(tracks["positions"].float() - tracks["canonical_positions"].float().unsqueeze(0), dim=-1)
        scatter_mean(disp.max(dim=0).values, "max_displacement_mean")

    state["per_cell_debug"] = per_cell
    if counts.numel() > 0:
        # Compact JSON list for quick inspection; full tensors are in motion_grid_state.pt.
        score = per_cell.get("max_displacement_mean", counts.float())
        topk = min(int(max_cells_json), int(counts.numel()))
        ids = torch.argsort(score, descending=True)[:topk]
        rows = []
        for cid_i in ids.tolist():
            row = {"cell_id": int(cid_i), "count": int(counts[cid_i].item())}
            for k, v in per_cell.items():
                if k in ("cell_id", "count"):
                    continue
                row[k] = float(v[cid_i].item())
            rows.append(row)
        summary["top_cells_by_max_displacement_mean"] = rows
    return summary, state


def _write_motion_grid_octomap_debug(grid_state, dyn, tag, out_dir):
    """Write octomap-style HTML for the adaptive motion grid at a phase boundary.

    This is a visualization only. It does not affect training, losses, optimizer
    state, grid topology, or grid features.
    """
    if not bool(getattr(dyn, "phase_debug_make_html", True)):
        return
    if not bool(getattr(dyn, "phase_debug_grid_octomap_html", True)):
        return
    if grid_state is None:
        return
    try:
        from utils.visualize_motion_grid_octomap_html import write_motion_grid_octomap_html

        write_motion_grid_octomap_html(
            grid_state=grid_state,
            out_html=str(out_dir / "motion_grid_octomap.html"),
            title=f"Adaptive Motion Grid / Feature Field - {tag}",
            color_mode=str(getattr(dyn, "phase_debug_grid_octomap_color_mode", "feature_pca")),
            max_cells=int(getattr(dyn, "phase_debug_grid_octomap_max_cells", 5000)),
            active_only=bool(getattr(dyn, "phase_debug_grid_octomap_active_only", True)),
            min_count=int(getattr(dyn, "phase_debug_grid_octomap_min_count", 1)),
            opacity=float(getattr(dyn, "phase_debug_grid_octomap_opacity", 0.24)),
        )
    except Exception as exc:
        print(f"[Dynamic 3DGS] Phase debug motion-grid octomap HTML failed for {tag}: {exc}")


def _write_track_visual_debug(model_path, tracks_path, dyn, tag, out_dir):
    if not bool(getattr(dyn, "phase_debug_make_html", True)):
        return
    cameras_json = Path(model_path) / "cameras.json"
    cameras_json_str = str(cameras_json) if cameras_json.exists() else ""
    try:
        from utils.visualize_gaussian_tracks_html import write_track_html_from_file
        write_track_html_from_file(
            tracks_path=str(tracks_path),
            out_html=str(out_dir / "gaussian_tracks_visualization.html"),
            cameras_json=cameras_json_str,
            max_points=int(getattr(dyn, "phase_debug_html_max_points", getattr(dyn, "track_html_max_points", 30000))),
            dynamic_threshold=float(getattr(dyn, "track_html_dynamic_threshold", 0.5)),
            frame_stride=int(getattr(dyn, "phase_debug_html_frame_stride", getattr(dyn, "track_html_frame_stride", 1))),
            title=f"Dynamic Eulerian 3DGS Tracks - {tag}",
            label_mode=str(getattr(dyn, "track_html_label_mode", "auto")),
            speed_percentile=float(getattr(dyn, "track_html_speed_percentile", 80.0)),
            frustum_scale=float(getattr(dyn, "html_frustum_scale", 0.12)),
            camera_deduplicate=bool(getattr(dyn, "html_camera_deduplicate", True)),
            aspectmode=getattr(dyn, "html_aspectmode", "cube"),
            axis_scope=getattr(dyn, "html_axis_scope", "all"),
        )
    except Exception as exc:
        print(f"[Dynamic 3DGS] Phase debug track HTML failed for {tag}: {exc}")
    try:
        from utils.visualize_velocity_field_html import write_velocity_field_html_from_file
        write_velocity_field_html_from_file(
            tracks_path=str(tracks_path),
            out_html=str(out_dir / "velocity_field_visualization.html"),
            cameras_json=cameras_json_str,
            max_points=int(getattr(dyn, "phase_debug_html_max_points", getattr(dyn, "velocity_html_max_points", 30000))),
            max_arrows=int(getattr(dyn, "phase_debug_velocity_max_arrows", getattr(dyn, "velocity_html_max_arrows", 3000))),
            frame_stride=int(getattr(dyn, "phase_debug_html_frame_stride", getattr(dyn, "velocity_html_frame_stride", 1))),
            speed_clip_percentile=float(getattr(dyn, "velocity_html_speed_clip_percentile", 98.0)),
            arrow_scale=float(getattr(dyn, "velocity_html_arrow_scale", 0.0)),
            title=f"Dynamic Eulerian 3DGS Velocity Field - {tag}",
            frustum_scale=float(getattr(dyn, "html_frustum_scale", 0.12)),
            camera_deduplicate=bool(getattr(dyn, "html_camera_deduplicate", True)),
            aspectmode=getattr(dyn, "html_aspectmode", "cube"),
            axis_scope=getattr(dyn, "html_axis_scope", "all"),
        )
    except Exception as exc:
        print(f"[Dynamic 3DGS] Phase debug velocity HTML failed for {tag}: {exc}")
    try:
        from utils.visualize_rigid_twist_html import write_rigid_twist_html_from_file
        write_rigid_twist_html_from_file(
            tracks_path=str(tracks_path),
            out_html=str(out_dir / "rigid_twist_trajectory.html"),
            title=f"Explicit learned rigid twist omega(t), u(t) - {tag}",
        )
    except Exception as exc:
        print(f"[Dynamic 3DGS] Phase debug rigid-twist HTML failed for {tag}: {exc}")


def export_gaussian_tracks_to_path(scene, gaussians, dyn, out_path, summary_path=None, phase_info=None):
    frame_ids, times = _collect_track_export_times(scene, dyn)
    if not times:
        return None, None
    dtype = torch.float16 if getattr(dyn, "track_export_dtype", "float32") == "float16" else torch.float32
    tracks = gaussians.export_time_tracks(
        times=times,
        frame_ids=frame_ids,
        steps=int(getattr(dyn, "motion_integration_steps", 4)),
        dtype=dtype,
        device="cpu",
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tracks, out_path)
    summary = _build_track_summary(tracks, dyn, phase_info=phase_info)
    summary["track_file"] = str(out_path.name)
    if summary_path is not None:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    return out_path, tracks


def export_phase_debug_snapshot(scene, gaussians, dyn, iteration, euler_iter, stage, tag):
    """Export a full diagnostic snapshot at a dynamic phase boundary.

    The snapshot is intentionally not a training loss. It writes the current
    dynamic state so we can answer: did the stuck region already fail at the end
    of dynamic labelling, or did it become stuck after rigid/twist losses?
    """
    if not bool(getattr(dyn, "export_phase_debug_snapshots", True)):
        return None
    if bool(getattr(dyn, "disable_phase_debug_snapshots", False)):
        return None
    if not getattr(dyn, "dynamic_motion", False) or not getattr(gaussians, "dynamic_enabled", False):
        return None
    root = Path(scene.model_path) / str(getattr(dyn, "phase_debug_dir", "debug_phase_snapshots"))
    safe_tag = str(tag).replace("/", "_").replace(" ", "_")
    out_dir = root / f"iter_{int(iteration):06d}_euler_{int(euler_iter):06d}_{safe_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_info = {
        "debug_tag": safe_tag,
        "iteration": int(iteration),
        "euler_iter": int(euler_iter),
        "stage": str(stage),
        "motion_understanding_from_iter": int(getattr(dyn, "motion_understanding_from_iter", -1)),
        "global_twist_from_iter": int(getattr(dyn, "global_twist_from_iter", -1)),
        "model_dynamic_target_from_iter": int(getattr(dyn, "model_dynamic_target_from_iter", -1)),
        "euler_field_train_iters": int(getattr(dyn, "euler_field_train_iters", -1)),
        "joint_single_rigid_training": bool(getattr(dyn, "joint_single_rigid_training", False)),
        "joint_debug_early_iter": int(getattr(dyn, "joint_debug_early_iter", -1)),
        "joint_debug_mid_iter": int(getattr(dyn, "joint_debug_mid_iter", -1)),
    }
    with torch.no_grad():
        tracks_path, tracks = export_gaussian_tracks_to_path(
            scene=scene,
            gaussians=gaussians,
            dyn=dyn,
            out_path=out_dir / "gaussian_tracks.pt",
            summary_path=out_dir / "dynamic_summary.json",
            phase_info=phase_info,
        )
        if tracks is not None:
            grid_summary, grid_state = _build_grid_debug_from_tracks(gaussians, tracks)
            grid_summary.update(phase_info)
            with open(out_dir / "motion_grid_summary.json", "w") as f:
                json.dump(grid_summary, f, indent=2)
            if grid_state is not None:
                torch.save(grid_state, out_dir / "motion_grid_state.pt")
                _write_motion_grid_octomap_debug(grid_state, dyn, safe_tag, out_dir)
            _write_deep_phase_diagnostics(out_dir, tracks, dyn)
            _write_track_visual_debug(scene.model_path, tracks_path, dyn, safe_tag, out_dir)
    print(f"[Dynamic 3DGS] Phase debug snapshot saved: {out_dir}")
    return out_dir


def export_gaussian_tracks(scene, gaussians, dyn):
    out_path = Path(scene.model_path) / getattr(dyn, "track_export_name", "gaussian_tracks.pt")
    summary_path = Path(scene.model_path) / "dynamic_summary.json"
    tracks_path, _ = export_gaussian_tracks_to_path(scene, gaussians, dyn, out_path=out_path, summary_path=summary_path)
    if tracks_path is not None:
        print(f"[Dynamic 3DGS] Exported Gaussian tracks to {tracks_path}")
    return tracks_path


def _load_gaussians_from_ply_for_resume(gaussians, ply_path: Path, opt):
    """Load a saved GraphDeco point_cloud.ply and rebuild train-time tensors.

    --start_checkpoint is only for chkpnt*.pth files. A saved static stage from
    scene.save(iter) is a PLY under point_cloud/iteration_XXXXX/point_cloud.ply;
    this helper loads that representation instead.
    """
    ply_path = Path(ply_path).expanduser().resolve()
    if not ply_path.exists():
        raise FileNotFoundError(f"Static PLY not found: {ply_path}")
    print(f"[Dynamic 3DGS] Loading static Gaussian PLY: {ply_path}")
    try:
        gaussians.load_ply(str(ply_path))
    except TypeError:
        # Some newer 3DGS forks add extra optional arguments; keep this robust.
        gaussians.load_ply(str(ply_path), False)
    # Important: the optimizer created before loading the PLY points to the old
    # COLMAP-initialized tensors. Rebuild it so all tensor sizes/parameters match
    # the loaded static Gaussians.
    gaussians.training_setup(opt)
    print(f"[Dynamic 3DGS] Loaded {gaussians.get_xyz.shape[0]} Gaussians from static PLY")


def _resolve_start_ply(scene, dyn, checkpoint) -> tuple[Path | None, int]:
    """Resolve an explicit or automatic static PLY resume point.

    Priority:
      1. --start_ply PATH
      2. if --start_checkpoint is missing and this is dynamic training, use
         <model_path>/point_cloud/iteration_<static_train_iters>/point_cloud.ply
         when it exists.
    """
    if not getattr(dyn, "dynamic_motion", False):
        return None, 0

    static_iter = int(getattr(dyn, "static_train_iters", 30000))
    explicit_ply = str(getattr(dyn, "start_ply", "") or "").strip()
    if explicit_ply:
        iter_hint = int(getattr(dyn, "start_ply_iteration", -1))
        return Path(explicit_ply).expanduser(), (iter_hint if iter_hint >= 0 else static_iter)

    if checkpoint:
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if checkpoint_path.exists():
            return None, 0
        auto_ply = Path(scene.model_path) / "point_cloud" / f"iteration_{static_iter}" / "point_cloud.ply"
        if auto_ply.exists():
            print(
                f"[Dynamic 3DGS] Requested checkpoint does not exist: {checkpoint_path}\n"
                f"[Dynamic 3DGS] Found saved static PLY instead and will resume from it: {auto_ply}\n"
                f"[Dynamic 3DGS] Use --start_ply explicitly next time; --start_checkpoint is only for chkpnt*.pth files."
            )
            return auto_ply, static_iter

    # Proper two-stage workflow: if the static checkpoint already exists in the
    # same output folder, automatically skip the expensive static 3DGS phase and
    # run only the dynamic Eulerian stage.  Use --force_static_retrain to ignore it.
    if bool(getattr(dyn, "auto_resume_static_ply", True)) and not bool(getattr(dyn, "force_static_retrain", False)):
        auto_ply = Path(scene.model_path) / "point_cloud" / f"iteration_{static_iter}" / "point_cloud.ply"
        if auto_ply.exists():
            print(
                f"[Dynamic 3DGS] Auto-resuming dynamic stage from existing static PLY:\n"
                f"  {auto_ply}\n"
                f"[Dynamic 3DGS] To retrain static from COLMAP, pass --force_static_retrain."
            )
            return auto_ply, static_iter

    return None, 0

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, dyn=None):

    if dyn is None:
        dyn = Namespace(dynamic_motion=False)

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    wb_run = setup_wandb(dataset, opt, pipe, dyn)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    # DINO maps are needed already in static training because the learnable
    # Gaussian feature field is supervised in both static and dynamic phases.
    maybe_extract_dinov3_features(dataset, dyn)
    # Two-stage mode starts with the vanilla COLMAP camera list. The dynamic
    # sequence is enabled only after static_train_iters, so frame-000000
    # canonical 3DGS optimization stays as close as possible to GraphDeco 3DGS.
    scene = Scene(dataset, gaussians, dynamic_args=None if dyn.dynamic_motion else dyn)
    init_gaussian_aux_feature_fields(gaussians, dataset, dyn)
    sync_feature_field_lrs_to_opt(opt, dyn)
    gaussians.training_setup(opt)

    start_ply, start_ply_iter = _resolve_start_ply(scene, dyn, checkpoint)
    if start_ply is not None:
        _load_gaussians_from_ply_for_resume(gaussians, start_ply, opt)
        init_gaussian_aux_feature_fields(gaussians, dataset, dyn)
        sync_feature_field_lrs_to_opt(opt, dyn)
        gaussians.training_setup(opt)
        first_iter = int(start_ply_iter)
        checkpoint = None
    elif checkpoint:
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Use --start_checkpoint only with chkpnt*.pth files. "
                f"For a saved GraphDeco point cloud, use --start_ply "
                f"<model_path>/point_cloud/iteration_<iter>/point_cloud.ply."
            )
        (model_params, first_iter) = torch.load(str(checkpoint_path))
        gaussians.restore(model_params, opt)

    dynamic_sequence_enabled = False

    def initialize_dynamic_stage_once():
        nonlocal dynamic_sequence_enabled, viewpoint_stack, viewpoint_indices
        if not dyn.dynamic_motion or dynamic_sequence_enabled:
            return
        maybe_extract_dinov3_features(dataset, dyn)

        # Initialize the exact per-Gaussian XYZ+DINO clustering before creating
        # hundreds of dynamic camera objects. This keeps the one-time CPU N^2
        # computation independent of dynamic-sequence image storage.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if not gaussians.dynamic_enabled:
            gaussians.init_dynamic_motion(dyn, scene.cameras_extent, cameras=scene.getTrainCameras())
        else:
            gaussians.setup_dynamic_optimizers(dyn)

        # Expanded sequence images are stored on CPU by default. The training
        # loop already copies only the sampled frame to CUDA via .cuda().
        previous_data_device = getattr(dataset, "data_device", "cuda")
        dynamic_camera_device = str(getattr(dyn, "dynamic_camera_data_device", "cpu"))
        dataset.data_device = dynamic_camera_device
        try:
            scene.enable_dynamic_sequence(dataset, dyn)
        finally:
            dataset.data_device = previous_data_device
        print(
            f"[Dynamic sequence] Frame image storage device: {dynamic_camera_device}; "
            "only the sampled frame is transferred to CUDA each iteration"
        )
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))
        init_dino_feature_field(gaussians, dataset, dyn)
        if bool(getattr(dyn, "joint_single_rigid_training", False)) and hasattr(gaussians, "export_joint_initialization"):
            gaussians.export_joint_initialization(Path(scene.model_path) / "joint_initialization")
        gaussians.freeze_canonical(
            freeze_geometry=dyn.freeze_canonical_geometry,
            freeze_appearance=dyn.freeze_canonical_appearance,
        )
        dynamic_sequence_enabled = True
        if bool(getattr(dyn, "joint_single_rigid_training", False)):
            nclusters = int(gaussians.get_joint_cluster_id.max().item() + 1) if getattr(gaussians, "get_joint_cluster_id", None) is not None and gaussians.get_joint_cluster_id.numel() else 0
            print(
                "[Dynamic 3DGS] Enabled exact JOINT one-rigid-part training: "
                f"full per-Gaussian XYZ+DINO k-medoids initialization, {nclusters} cosine-field slots, "
                "proposal + explicit omega(t),u(t) optimized every dynamic iteration"
            )
        else:
            print("[Dynamic 3DGS] Enabled Eulerian velocity-field transport")
        if hasattr(gaussians, "get_dynamic_foreground_mask"):
            fg = gaussians.get_dynamic_foreground_mask
            if bool(getattr(dyn, "joint_single_rigid_training", False)):
                print(
                    f"[Dynamic 3DGS] Exact joint assignment uses all {int(fg.numel())} starting Gaussians; "
                    "semantic filtering is not used to omit points from initialization or optimization"
                )
            elif bool(getattr(dyn, "filter_sem", False)):
                print(f"[Dynamic 3DGS] Semantic dynamic filter ON: {int(fg.sum().item())}/{int(fg.numel())} gaussians, threshold={float(getattr(gaussians, 'dynamic_foreground_threshold', 0.5)):.3f}")
            else:
                print(f"[Dynamic 3DGS] Semantic dynamic filter OFF: using all {int(fg.numel())} gaussians as dynamic candidates")
        print(f"[Dynamic 3DGS] Motion cells: {gaussians.motion_grid.num_cells}, voxel_size={gaussians.motion_grid.voxel_size:.6f}")

    if dyn.dynamic_motion and first_iter >= int(getattr(dyn, "static_train_iters", 30000)):
        initialize_dynamic_stage_once()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_dynamic_for_log = 0.0
    phase_debug_done = set()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    if dyn.dynamic_motion and dynamic_sequence_enabled:
                        gaussians.set_time(0.0, detach_canonical=True, steps=dyn.motion_integration_steps)
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    if dyn.dynamic_motion and dynamic_sequence_enabled:
                        gaussians.clear_time()
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree.
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        stage, euler_iter = dynamic_stage(iteration, dyn)
        if dyn.dynamic_motion and stage != "static" and not dynamic_sequence_enabled:
            initialize_dynamic_stage_once()

        # Pick a random Camera. In two-stage mode this samples vanilla COLMAP
        # cameras during the static stage and dynamic (cam, frame) clones during
        # the Euler stage.
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render.
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if dyn.dynamic_motion and stage != "static":
            if bool(getattr(dyn, "joint_single_rigid_training", False)) and hasattr(gaussians, "set_joint_training_progress"):
                gaussians.set_joint_training_progress(euler_iter, dyn)
            detach_canonical = not (stage == "finetune" and dyn.allow_dynamic_gaussian_finetune)
            if stage == "finetune" and dyn.allow_dynamic_gaussian_finetune:
                gaussians.unfreeze_canonical_appearance()
            t = get_camera_timestamp(viewpoint_cam, fps=dyn.fps, time_scale=dyn.time_scale)
            gaussians.set_time(t, detach_canonical=detach_canonical, steps=dyn.motion_integration_steps)

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss.
        gt_image = viewpoint_cam.original_image.cuda()
        photometric_loss, Ll1, ssim_value = compute_base_render_loss(image, gt_image, opt)
        loss = photometric_loss

        # Depth regularization.
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        dynamic_stats = {}
        # Semantic foreground is supervised only in the static canonical stage.
        # DINO feature-field supervision is used in both static and dynamic stages.
        if (
            getattr(dyn, "use_semantic_feature_field", False)
            and float(getattr(dyn, "lambda_semantic_feature_field", 0.0)) > 0
            and stage == "static"
        ):
            Lsem, sem_stats = semantic_feature_field_loss(viewpoint_cam, gaussians, pipe, dataset, dyn)
            if Lsem is not None:
                loss = loss + float(dyn.lambda_semantic_feature_field) * Lsem
            dynamic_stats.update(sem_stats)

        dino_weight = (
            float(getattr(dyn, "lambda_dino_feature_field", 0.0))
            if stage == "static"
            else float(getattr(dyn, "lambda_dino_feature_field_dynamic", 0.003))
        )
        dino_from = (
            int(getattr(dyn, "dino_feature_from_iter", 0))
            if stage == "static"
            else int(getattr(dyn, "dino_feature_dynamic_from_iter", 0))
        )
        if (
            getattr(dyn, "use_dino_feature_field", False)
            and dino_weight > 0
            and (stage == "static" or euler_iter >= dino_from)
        ):
            Lfeat, feat_stats = dino_feature_field_loss(
                viewpoint_cam=viewpoint_cam,
                gaussians=gaussians,
                pipe=pipe,
                background=bg,
                dataset=dataset,
                dyn=dyn,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                stage=stage,
            )
            if Lfeat is not None:
                loss = loss + dino_weight * Lfeat
                dynamic_stats["dino_loss_weight"] = float(dino_weight)
            dynamic_stats.update(feat_stats)

        if dyn.dynamic_motion and stage != "static":
            loss, dyn_loss_stats = add_dynamic_losses(loss, photometric_loss, gaussians, iteration, euler_iter, dyn, stage)
            dynamic_stats.update(dyn_loss_stats)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar.
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            dyn_scalar = sum(float(v) for k, v in dynamic_stats.items() if str(k).startswith("L_")) if dynamic_stats else 0.0
            ema_dynamic_for_log = 0.4 * dyn_scalar + 0.6 * ema_dynamic_for_log

            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"}
                if dyn.dynamic_motion:
                    postfix.update({"Dyn": f"{ema_dynamic_for_log:.{7}f}", "Stage": stage})
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer and dyn.dynamic_motion:
                tb_writer.add_scalar('dynamic/stage_id', {"static": 0, "joint": 1, "discovery": 2, "motion": 3, "finetune": 4}.get(stage, 0), iteration)
                if gaussians.dynamic_enabled and gaussians.motion_grid is not None:
                    tb_writer.add_scalar('dynamic/motion_cells', gaussians.motion_grid.num_cells, iteration)
                if gaussians.get_current_velocity is not None:
                    tb_writer.add_scalar('dynamic/mean_speed', torch.linalg.norm(gaussians.get_current_velocity.detach(), dim=-1).mean().item(), iteration)
                for k, v in dynamic_stats.items():
                    tb_writer.add_scalar('dynamic_losses/' + k, v, iteration)

            # Log and save.
            elapsed_ms = iter_start.elapsed_time(iter_end)
            wandb_log_training(wb_run, iteration, stage, euler_iter, loss, Ll1, Ll1depth, dynamic_stats, scene, gaussians, dyn, elapsed_ms)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed_ms, testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, dyn, wb_run)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification. During dynamic training, keep Gaussian identity fixed;
            # adaptive refinement is applied to the motion grid instead.
            if (not dyn.dynamic_motion or stage == "static") and iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step.
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)

                if dyn.dynamic_motion and stage != "static":
                    if gaussians.motion_optimizer is not None:
                        if bool(getattr(dyn, "joint_single_rigid_training", False)) and float(getattr(dyn, "joint_grad_clip", 0.0)) > 0:
                            motion_params = list(gaussians.motion_field.parameters()) + [gaussians.motion_grid.features]
                            if getattr(gaussians, "rigid_twist_field", None) is not None:
                                motion_params += list(gaussians.rigid_twist_field.parameters())
                            torch.nn.utils.clip_grad_norm_(motion_params, float(dyn.joint_grad_clip))
                        gaussians.motion_optimizer.step()
                        gaussians.motion_optimizer.zero_grad(set_to_none=True)
                    if gaussians.dynamic_optimizer is not None:
                        if hasattr(gaussians, "mask_dynamic_gradients"):
                            gaussians.mask_dynamic_gradients()
                        if bool(getattr(dyn, "joint_single_rigid_training", False)) and float(getattr(dyn, "joint_grad_clip", 0.0)) > 0:
                            part_field = getattr(gaussians, "joint_part_field", None)
                            if part_field is not None:
                                torch.nn.utils.clip_grad_norm_(list(part_field.parameters()), float(dyn.joint_grad_clip))
                            elif gaussians._dynamic_logit is not None and gaussians._dynamic_logit.numel() > 0:
                                torch.nn.utils.clip_grad_norm_([gaussians._dynamic_logit], float(dyn.joint_grad_clip))
                        gaussians.dynamic_optimizer.step()
                        if bool(getattr(dyn, "joint_single_rigid_training", False)):
                            part_field = getattr(gaussians, "joint_part_field", None)
                            if part_field is not None:
                                occupancy_info = part_field.enforce_nonempty_parts(
                                    minimum_part_fraction=float(getattr(dyn, "joint_minimum_part_fraction", 0.02))
                                )
                                gaussians._last_joint_occupancy_info = occupancy_info
                        gaussians.dynamic_optimizer.zero_grad(set_to_none=True)
                    # DINO Gaussian descriptors are frozen in the dynamic stage.
                    # L_dino still backpropagates through the transported render,
                    # but its gradients train motion_field/motion_grid, not z_i.
                    if stage == "finetune" and dyn.allow_dynamic_gaussian_finetune:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)
                    changed = maybe_update_motion_grid(
                        gaussians,
                        dyn,
                        iteration,
                        euler_iter,
                        viewspace_point_tensor=viewspace_point_tensor,
                        visibility_filter=visibility_filter,
                        wb_run=wb_run,
                    )
                    if changed:
                        print(f"\n[ITER {iteration}] Motion-grid topology updated. Active cells: {int(gaussians.motion_grid.active_mask.sum().item())}/{gaussians.motion_grid.num_cells}")
                else:
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none=True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)

            if dyn.dynamic_motion and stage != "static":
                gaussians.clear_time()

                if bool(getattr(dyn, "export_phase_debug_snapshots", True)) and not bool(getattr(dyn, "disable_phase_debug_snapshots", False)):
                    if bool(getattr(dyn, "joint_single_rigid_training", False)):
                        early_euler = max(1, int(getattr(dyn, "joint_debug_early_iter", 1000)))
                        total_euler = int(getattr(dyn, "euler_field_train_iters", euler_iter))
                        mid_euler = int(getattr(dyn, "joint_debug_mid_iter", -1))
                        if mid_euler < 0:
                            mid_euler = max(early_euler + 1, total_euler // 2)
                        if euler_iter >= early_euler and "joint_early" not in phase_debug_done:
                            export_phase_debug_snapshot(scene, gaussians, dyn, iteration, euler_iter, stage, "joint_early")
                            phase_debug_done.add("joint_early")
                        if euler_iter >= mid_euler and "joint_mid" not in phase_debug_done:
                            export_phase_debug_snapshot(scene, gaussians, dyn, iteration, euler_iter, stage, "joint_mid")
                            phase_debug_done.add("joint_mid")
                        if (euler_iter >= total_euler or iteration == opt.iterations) and "joint_final" not in phase_debug_done:
                            export_phase_debug_snapshot(scene, gaussians, dyn, iteration, euler_iter, stage, "joint_final")
                            phase_debug_done.add("joint_final")
                    else:
                        motion_from_dbg = int(getattr(dyn, "motion_understanding_from_iter", -1))
                        if motion_from_dbg < 0:
                            motion_from_dbg = int(getattr(dyn, "dynamic_discovery_iters", 1000))
                        end_discovery_euler = max(1, motion_from_dbg - 1)
                        end_motion_euler = int(getattr(dyn, "euler_field_train_iters", euler_iter))
                        if euler_iter == end_discovery_euler and "end_dynamic_labeling" not in phase_debug_done:
                            export_phase_debug_snapshot(scene, gaussians, dyn, iteration, euler_iter, stage, "end_dynamic_labeling")
                            phase_debug_done.add("end_dynamic_labeling")
                        if (euler_iter >= end_motion_euler or iteration == opt.iterations) and "end_rigid_motion" not in phase_debug_done:
                            export_phase_debug_snapshot(scene, gaussians, dyn, iteration, euler_iter, stage, "end_rigid_motion")
                            phase_debug_done.add("end_rigid_motion")

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if dyn.dynamic_motion and gaussians.dynamic_enabled:
        tracks_path = None
        if getattr(dyn, "export_gaussian_tracks", True):
            if hasattr(gaussians, "export_time_tracks"):
                tracks_path = export_gaussian_tracks(scene, gaussians, dyn)
            else:
                print("[Dynamic 3DGS] WARNING: GaussianModel.export_time_tracks missing; skipping track export.")
        post_training_render_and_visualize(
            scene=scene,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            dataset=dataset,
            dyn=dyn,
            iteration=opt.iterations,
            tracks_path=tracks_path,
        )

    wandb_log_artifacts(wb_run, scene.model_path, dyn)
    if wb_run is not None:
        try:
            wb_run.finish()
        except Exception as exc:
            print(f"[W&B] finish failed: {exc}")


def post_training_render_and_visualize(scene, gaussians, pipe, background, dataset, dyn, iteration, tracks_path=None):
    """Optional post-training products: rendered RGB sequence and Plotly HTML tracks.

    This stays outside the training loop and uses utilities under utils/ so the
    same functions can also be called from standalone scripts/notebooks.
    """
    if not getattr(dyn, "dynamic_motion", False):
        return

    model_path = Path(scene.model_path)
    if tracks_path is None:
        tracks_path = model_path / getattr(dyn, "track_export_name", "gaussian_tracks.pt")
    else:
        tracks_path = Path(tracks_path)

    if getattr(dyn, "render_after_training", False):
        try:
            from utils.render_dynamic_motion import render_dynamic_sequence
            render_dynamic_sequence(
                scene=scene,
                gaussians=gaussians,
                pipeline=pipe,
                background=background,
                dyn=dyn,
                iteration=iteration,
                tracks_path=str(tracks_path),
                out_root=model_path / getattr(dyn, "render_output_name", "dynamic_renders"),
                cameras=getattr(dyn, "render_cameras", ""),
                frame_start=int(getattr(dyn, "render_frame_start", 0)),
                frame_end=int(getattr(dyn, "render_frame_end", -1)),
                frame_stride=int(getattr(dyn, "render_frame_stride", 1)),
                split=getattr(dyn, "render_split", "all"),
                use_tracks=bool(getattr(dyn, "render_use_tracks", True)),
                save_video=bool(getattr(dyn, "render_save_video", False)),
                train_test_exp=bool(getattr(dataset, "train_test_exp", False)),
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )
        except Exception as exc:
            print(f"[Dynamic 3DGS] Post-training render failed: {exc}")

    if getattr(dyn, "make_track_html", False):
        try:
            from utils.visualize_gaussian_tracks_html import write_track_html_from_file
            out_html = model_path / getattr(dyn, "track_html_name", "gaussian_tracks_visualization.html")
            cameras_json = model_path / "cameras.json"
            if not cameras_json.exists():
                cameras_json = ""
            write_track_html_from_file(
                tracks_path=str(tracks_path),
                out_html=str(out_html),
                cameras_json=str(cameras_json) if cameras_json else "",
                max_points=int(getattr(dyn, "track_html_max_points", 30000)),
                dynamic_threshold=float(getattr(dyn, "track_html_dynamic_threshold", 0.5)),
                frame_stride=int(getattr(dyn, "track_html_frame_stride", 1)),
                title="Dynamic Eulerian 3DGS Gaussian Tracks",
                label_mode=str(getattr(dyn, "track_html_label_mode", "auto")),
                speed_percentile=float(getattr(dyn, "track_html_speed_percentile", 80.0)),
                frustum_scale=float(getattr(dyn, "html_frustum_scale", 0.12)),
                camera_deduplicate=bool(getattr(dyn, "html_camera_deduplicate", True)),
                aspectmode=getattr(dyn, "html_aspectmode", "cube"),
                axis_scope=getattr(dyn, "html_axis_scope", "all"),
            )
        except Exception as exc:
            print(f"[Dynamic 3DGS] Track HTML visualization failed: {exc}")

    if getattr(dyn, "make_velocity_html", False):
        try:
            from utils.visualize_velocity_field_html import write_velocity_field_html_from_file
            out_html = model_path / getattr(dyn, "velocity_html_name", "velocity_field_visualization.html")
            cameras_json = model_path / "cameras.json"
            if not cameras_json.exists():
                cameras_json = ""
            write_velocity_field_html_from_file(
                tracks_path=str(tracks_path),
                out_html=str(out_html),
                cameras_json=str(cameras_json) if cameras_json else "",
                max_points=int(getattr(dyn, "velocity_html_max_points", 30000)),
                max_arrows=int(getattr(dyn, "velocity_html_max_arrows", 3000)),
                frame_stride=int(getattr(dyn, "velocity_html_frame_stride", 1)),
                speed_clip_percentile=float(getattr(dyn, "velocity_html_speed_clip_percentile", 98.0)),
                arrow_scale=float(getattr(dyn, "velocity_html_arrow_scale", 0.0)),
                title="Dynamic Eulerian 3DGS Optimized Velocity Field",
                frustum_scale=float(getattr(dyn, "html_frustum_scale", 0.12)),
                camera_deduplicate=bool(getattr(dyn, "html_camera_deduplicate", True)),
                aspectmode=getattr(dyn, "html_aspectmode", "cube"),
                axis_scope=getattr(dyn, "html_axis_scope", "all"),
            )
        except Exception as exc:
            print(f"[Dynamic 3DGS] Velocity-field HTML visualization failed: {exc}")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder.
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer.
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, train_test_exp, dyn=None, wb_run=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set.
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if dyn is not None and getattr(dyn, "dynamic_motion", False):
                        t = get_camera_timestamp(viewpoint, fps=dyn.fps, time_scale=dyn.time_scale)
                        scene.gaussians.set_time(t, detach_canonical=True, steps=dyn.motion_integration_steps)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    if dyn is not None and getattr(dyn, "dynamic_motion", False):
                        scene.gaussians.clear_time()
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                wandb_log_validation(wb_run, iteration, config['name'], float(l1_test), float(psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser.
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=True)
    parser.add_argument('--enable_viewer', dest='disable_viewer', action='store_false')
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    add_dynamic_args(parser)
    args = parser.parse_args(sys.argv[1:])

    # Single-root scene input.
    # If --root_scene is provided, always derive the standard 3DGS source path as ROOT/colmap.
    # This means the user does not need to pass -s for generic dynamic scenes.
    if args.root_scene:
        root_scene = Path(args.root_scene).expanduser().resolve()
        args.root_scene = str(root_scene)
        args.source_path = str(root_scene / "colmap")
    elif not getattr(args, "source_path", ""):
        parser.error("Either pass --root_scene ROOT or the standard 3DGS -s/--source_path COLMAP_PATH.")

    args = apply_dynamic_defaults(args)

    if args.dynamic_motion:
        args.iterations = int(args.static_train_iters) + int(args.euler_field_train_iters)

    args.save_iterations.append(args.iterations)
    if args.dynamic_motion and args.static_train_iters not in args.save_iterations:
        args.save_iterations.append(args.static_train_iters)
    if args.dynamic_motion and args.iterations not in args.checkpoint_iterations:
        args.checkpoint_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG).
    safe_state(args.quiet)

    dataset_args = lp.extract(args)
    dataset_args.fps = args.fps
    dataset_args.time_scale = args.time_scale

    for _name in [
        "root_scene",
        "dynamic_sequence_root", "dynamic_rgb_dir", "dynamic_depth_npy_dir",
        "dynamic_depth_png_dir", "dynamic_use_depth", "dynamic_use_depth_npy",
        "dynamic_depth_is_metric", "dynamic_cameras", "dynamic_frame_start",
        "dynamic_frame_end", "dynamic_frame_stride", "dynamic_test_stride",
        "use_dinov3", "dinov3_root", "dinov3_feature_ext",
    ]:
        setattr(dataset_args, _name, getattr(args, _name))

    opt_args = op.extract(args)
    pipe_args = pp.extract(args)
    dyn_args = args

    # Start GUI server, configure and run training.
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(dataset_args, opt_args, pipe_args, args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, dyn_args)

    # All done.
    print("\nTraining complete.")
