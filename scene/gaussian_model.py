#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.motion_field import EulerianVelocityField
from scene.adaptive_motion_grid import AdaptiveMotionGrid
from scene.rigid_twist_field import TimeVaryingRigidTwist
from scene.joint_part_field import (
    JointPartField,
    initialize_joint_parts_from_static_dino,
    save_joint_initialization,
)

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._semantic_logits = torch.empty(0)
        self._dino_feature_field = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # Dynamic Eulerian-motion extension. These attributes are inert unless
        # init_dynamic_motion(...) is called by train.py. The CUDA renderer still
        # consumes the ordinary Gaussian tensors through the get_* properties.
        self.motion_field = None
        self.motion_grid = None
        self.motion_optimizer = None
        self.dynamic_optimizer = None
        self.rigid_twist_field = None
        self.joint_part_field = None
        self._joint_initialization_result = None
        self._joint_initialization_config = {}
        self.joint_rigid_handover = 0.0
        self.joint_confident_static_threshold = 0.20
        self.joint_confident_dynamic_threshold = 0.80
        self._last_joint_occupancy_info = {}
        self.rigid_center = torch.empty(0)
        self.joint_single_rigid_enabled = False
        self.joint_total_max_velocity = 0.0
        self._joint_cluster_id = torch.empty(0, dtype=torch.long)
        self._current_rigid_velocity = None
        self._current_residual_velocity = None
        self._current_omega = None
        self._current_translation = None
        self.dino_feature_optimizer = None
        self._dynamic_logit = torch.empty(0)
        self._dino_feat = torch.empty(0)
        self._cell_id = torch.empty(0, dtype=torch.long)
        self._current_xyz = None
        self._current_rotation = None
        self._current_velocity = None
        self._current_time = None
        self.dynamic_enabled = False
        self._dynamic_foreground_mask = torch.empty(0, dtype=torch.bool, device="cuda")
        self.dynamic_foreground_threshold = 0.5
        self.setup_functions()

    def capture_aux_state(self):
        """Checkpoint learnable static auxiliary fields.

        PLY saving already embeds these fields, but checkpoints should preserve
        them too.  Older checkpoints without this state still load normally.
        """
        return {
            "semantic_logits": self._semantic_logits.detach() if self._semantic_logits is not None and self._semantic_logits.numel() > 0 else None,
            "dino_feature_field": self._dino_feature_field.detach() if self._dino_feature_field is not None and self._dino_feature_field.numel() > 0 else None,
        }

    def load_aux_state(self, state):
        if state is None:
            return
        sem = state.get("semantic_logits", None) if isinstance(state, dict) else None
        if sem is not None and sem.numel() > 0:
            self._semantic_logits = nn.Parameter(sem.float().cuda().contiguous().requires_grad_(True))
        din = state.get("dino_feature_field", None) if isinstance(state, dict) else None
        if din is not None and din.numel() > 0:
            self._dino_feature_field = nn.Parameter(din.float().cuda().contiguous().requires_grad_(True))

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.capture_dynamic_state(),
            self.capture_aux_state(),
        )
    
    def restore(self, model_args, training_args):
        dynamic_state = None
        aux_state = None
        if len(model_args) == 14:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            dynamic_state,
            aux_state) = model_args
        elif len(model_args) == 13:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            dynamic_state) = model_args
        else:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        if aux_state is not None:
            self.load_aux_state(aux_state)
        if dynamic_state is not None:
            self.load_dynamic_state(dynamic_state, training_args)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self._current_rotation is not None:
            return self.rotation_activation(self._current_rotation)
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        if self._current_xyz is not None:
            return self._current_xyz
        return self._xyz

    @property
    def get_canonical_xyz(self):
        return self._xyz

    @property
    def get_current_velocity(self):
        return self._current_velocity

    @property
    def get_current_rigid_velocity(self):
        return self._current_rigid_velocity

    @property
    def get_current_residual_velocity(self):
        # Compatibility name: in joint mode this is the unrestricted proposal
        # field, not a final residual correction.
        return self._current_residual_velocity

    @property
    def get_current_proposal_velocity(self):
        return self._current_residual_velocity

    @property
    def get_current_twist(self):
        return self._current_omega, self._current_translation

    @property
    def get_joint_cluster_id(self):
        if self.joint_part_field is not None:
            return self.joint_part_field.hard_cluster_id_full()
        return self._joint_cluster_id

    @property
    def get_joint_initial_cluster_id(self):
        if self.joint_part_field is not None:
            return self.joint_part_field.initial_cluster_id_full()
        return self._joint_cluster_id

    @property
    def get_joint_part_assignments(self):
        if self.joint_part_field is None:
            return None
        return self.joint_part_field.full_assignments()

    @property
    def get_dynamic_probability(self):
        if self.joint_part_field is not None:
            return self.joint_part_field.dynamic_probability_full()
        if self._dynamic_logit.numel() == 0:
            return None
        return torch.sigmoid(self._dynamic_logit)

    @property
    def get_cell_id(self):
        if self.motion_grid is not None:
            return self.motion_grid.gaussian_cell_id
        return self._cell_id

    @property
    def get_dino_features(self):
        # Motion/adaptive-grid hint features. Prefer the learned feature field
        # when it exists, otherwise fall back to any loaded static hints.
        if self._dino_feature_field is not None and self._dino_feature_field.numel() > 0:
            return self._dino_feature_field
        return self._dino_feat

    @property
    def get_dino_feature_field(self):
        return self._dino_feature_field

    @property
    def get_semantic_logits(self):
        return self._semantic_logits

    @property
    def get_semantic_probability(self):
        if self._semantic_logits is None or self._semantic_logits.numel() == 0:
            return None
        return torch.sigmoid(self._semantic_logits)

    @property
    def get_dynamic_foreground_mask(self):
        if self._dynamic_foreground_mask is None or self._dynamic_foreground_mask.numel() != self._xyz.shape[0]:
            return torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
        return self._dynamic_foreground_mask.bool()

    def build_dynamic_foreground_mask(self, args):
        """Select Gaussians allowed to move.

        By default all Gaussians are dynamic candidates.  When --filter_sem is
        enabled, only learned semantic-foreground Gaussians are allowed to move.
        """
        n = int(self._xyz.shape[0])
        device = self._xyz.device
        use_semantic = bool(getattr(args, "filter_sem", False)) or bool(getattr(args, "use_semantic_dynamic_foreground", False))
        if bool(getattr(args, "disable_semantic_dynamic_foreground", False)):
            use_semantic = False
        thr = float(getattr(args, "semantic_dynamic_threshold", 0.5))
        min_pts = int(getattr(args, "semantic_dynamic_min_points", 32))
        self.dynamic_foreground_threshold = thr

        if not use_semantic:
            mask = torch.ones(n, dtype=torch.bool, device=device)
        else:
            sem_prob = self.get_semantic_probability
            if sem_prob is None or sem_prob.numel() != n:
                mask = torch.ones(n, dtype=torch.bool, device=device)
            else:
                prob = sem_prob.detach().reshape(-1)
                mask = prob > thr
                if int(mask.sum().item()) < max(1, min_pts):
                    k = min(n, max(1, min_pts))
                    top = torch.argsort(prob, descending=True)[:k]
                    mask = torch.zeros(n, dtype=torch.bool, device=device)
                    mask[top] = True

        self._dynamic_foreground_mask = mask.detach().bool()
        return self._dynamic_foreground_mask

    def mask_dynamic_gradients(self):
        """Prevent non-foreground Gaussians from learning legacy logits.

        JointPartField stores parameters only for foreground Gaussians, so it
        requires no gradient masking. This method remains for legacy mode.
        """
        if self.joint_part_field is not None:
            return
        if self._dynamic_logit is not None and self._dynamic_logit.grad is not None:
            mask = self.get_dynamic_foreground_mask.reshape(-1, 1).to(self._dynamic_logit.grad.device)
            self._dynamic_logit.grad.mul_(mask.float())

    def init_semantic_and_feature_fields(self, semantic=True, dino_dim=0, init_dino_random=True):
        """Initialize learnable semantic/DINO fields for the current Gaussians.

        This must be called before training_setup() if the fields should be part
        of the main Gaussian optimizer. It is intentionally safe to call after a
        PLY load; shape mismatches are rebuilt.
        """
        n = int(self._xyz.shape[0]) if self._xyz.numel() else 0
        if n <= 0:
            return
        device = self._xyz.device
        if semantic:
            if self._semantic_logits is None or self._semantic_logits.numel() == 0 or self._semantic_logits.shape[0] != n:
                self._semantic_logits = nn.Parameter(torch.zeros((n, 1), device=device, dtype=torch.float32).requires_grad_(True))
        dino_dim = int(dino_dim or 0)
        if dino_dim > 0:
            if self._dino_feature_field is None or self._dino_feature_field.numel() == 0 or tuple(self._dino_feature_field.shape) != (n, dino_dim):
                if init_dino_random:
                    feat = torch.randn((n, dino_dim), device=device, dtype=torch.float32) * 0.01
                    feat = torch.nn.functional.normalize(feat, dim=-1)
                else:
                    feat = torch.zeros((n, dino_dim), device=device, dtype=torch.float32)
                self._dino_feature_field = nn.Parameter(feat.requires_grad_(True))
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        rotation = self._current_rotation if self._current_rotation is not None else self._rotation
        return self.covariance_activation(self.get_scaling, scaling_modifier, rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._semantic_logits = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True))
        self._dino_feature_field = torch.empty((self.get_xyz.shape[0], 0), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self._semantic_logits is not None and self._semantic_logits.numel() > 0:
            l.append({'params': [self._semantic_logits], 'lr': float(getattr(training_args, "semantic_feature_lr", 1e-3)), "name": "semantic_logits"})
        if self._dino_feature_field is not None and self._dino_feature_field.numel() > 0:
            l.append({'params': [self._dino_feature_field], 'lr': float(getattr(training_args, "dino_feature_lr", 1e-3)), "name": "dino_feature"})

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


    # ---------------------------------------------------------------------
    # Dynamic Eulerian-motion extension
    # ---------------------------------------------------------------------
    def capture_dynamic_state(self):
        if not self.dynamic_enabled or self.motion_field is None or self.motion_grid is None:
            return None
        return {
            "dynamic_enabled": self.dynamic_enabled,
            "motion_field": self.motion_field.state_dict(),
            "motion_grid": self.motion_grid.topology_state_dict(),
            "rigid_twist_field": self.rigid_twist_field.state_dict() if self.rigid_twist_field is not None else None,
            "rigid_center": self.rigid_center.detach() if torch.is_tensor(self.rigid_center) and self.rigid_center.numel() == 3 else None,
            "joint_single_rigid_enabled": bool(self.joint_single_rigid_enabled),
            "joint_cluster_id": self.get_joint_cluster_id.detach() if self.get_joint_cluster_id is not None and self.get_joint_cluster_id.numel() else None,
            "joint_initial_cluster_id": self.get_joint_initial_cluster_id.detach() if self.get_joint_initial_cluster_id is not None and self.get_joint_initial_cluster_id.numel() else None,
            "joint_part_field_state": self.joint_part_field.state_dict() if self.joint_part_field is not None else None,
            "joint_part_field_config": ({
                "num_gaussians": int(self.joint_part_field.num_gaussians),
                "gaussian_ids": self.joint_part_field.gaussian_ids.detach(),
                "initial_cluster_id_fg": self.joint_part_field.initial_cluster_id_fg.detach(),
                "num_moving_parts": int(self.joint_part_field.num_moving_parts),
                "embedding_dim": int(self.joint_part_field.embedding_dim),
                "assignment_temperature": float(self.joint_part_field.assignment_temperature),
            } if self.joint_part_field is not None else None),
            "joint_rigid_handover": float(self.joint_rigid_handover),
            "joint_confident_static_threshold": float(self.joint_confident_static_threshold),
            "joint_confident_dynamic_threshold": float(self.joint_confident_dynamic_threshold),
            "dynamic_logit": self._dynamic_logit.detach() if self._dynamic_logit.numel() else None,
            "dynamic_foreground_mask": self._dynamic_foreground_mask.detach() if self._dynamic_foreground_mask is not None and self._dynamic_foreground_mask.numel() else None,
            "dynamic_foreground_threshold": float(getattr(self, "dynamic_foreground_threshold", 0.5)),
            "dino_feat": self._dino_feat.detach() if self._dino_feat.numel() else None,
        }

    def load_dynamic_state(self, state, training_args=None):
        if state is None:
            return
        grid_state = state.get("motion_grid", None)
        if grid_state is None:
            return
        feature_dim = int(grid_state.get("feature_dim", grid_state["features"].shape[-1]))
        hidden_dim = int(getattr(training_args, "motion_hidden_dim", 64)) if training_args is not None else 64
        time_freq = int(getattr(training_args, "motion_time_frequencies", 8)) if training_args is not None else 8
        num_layers = int(getattr(training_args, "motion_num_layers", 3)) if training_args is not None else 3
        max_velocity = float(getattr(training_args, "motion_max_velocity", 0.0)) if training_args is not None else 0.0
        xyz_scale = max(float(self.spatial_lr_scale), 1e-6)
        self.motion_grid = AdaptiveMotionGrid.from_topology_state_dict(grid_state)
        self.motion_field = EulerianVelocityField(feature_dim, hidden_dim, num_layers, time_freq, max_velocity, xyz_scale).cuda()
        self.motion_field.load_state_dict(state["motion_field"], strict=False)
        self.joint_single_rigid_enabled = bool(state.get("joint_single_rigid_enabled", getattr(training_args, "joint_single_rigid_training", False) if training_args is not None else False))
        self.joint_total_max_velocity = float(getattr(training_args, "joint_total_max_velocity", 3.0)) if training_args is not None else 3.0
        if self.joint_single_rigid_enabled:
            self.rigid_twist_field = TimeVaryingRigidTwist(
                hidden_dim=int(getattr(training_args, "joint_twist_hidden_dim", 64)) if training_args is not None else 64,
                num_layers=int(getattr(training_args, "joint_twist_num_layers", 3)) if training_args is not None else 3,
                time_frequencies=int(getattr(training_args, "joint_twist_time_frequencies", 8)) if training_args is not None else 8,
                max_omega=float(getattr(training_args, "joint_max_omega", 4.0)) if training_args is not None else 4.0,
                max_translation=float(getattr(training_args, "joint_max_translation_velocity", 2.0)) if training_args is not None else 2.0,
            ).cuda()
            if state.get("rigid_twist_field", None) is not None:
                self.rigid_twist_field.load_state_dict(state["rigid_twist_field"], strict=False)
            center = state.get("rigid_center", None)
            self.rigid_center = (center.float().cuda() if center is not None else self._xyz.detach().mean(dim=0)).detach()
            self.joint_rigid_handover = float(state.get("joint_rigid_handover", 0.0))
            self.joint_confident_static_threshold = float(state.get("joint_confident_static_threshold", getattr(training_args, "joint_confident_static_threshold", 0.20) if training_args is not None else 0.20))
            self.joint_confident_dynamic_threshold = float(state.get("joint_confident_dynamic_threshold", getattr(training_args, "joint_confident_dynamic_threshold", 0.80) if training_args is not None else 0.80))

            part_cfg = state.get("joint_part_field_config", None)
            part_state = state.get("joint_part_field_state", None)
            if part_cfg is not None and part_state is not None:
                saved_ids = part_cfg.get("gaussian_ids", part_cfg.get("foreground_ids", None))
                if saved_ids is None:
                    saved_ids = torch.arange(self._xyz.shape[0], dtype=torch.long)
                self.joint_part_field = JointPartField(
                    num_gaussians=int(part_cfg.get("num_gaussians", self._xyz.shape[0])),
                    gaussian_ids=saved_ids.long(),
                    initial_cluster_id_fg=part_cfg["initial_cluster_id_fg"].long(),
                    num_moving_parts=int(part_cfg.get("num_moving_parts", 1)),
                    embedding_dim=int(part_cfg.get("embedding_dim", 8)),
                    assignment_temperature=float(part_cfg.get("assignment_temperature", 1.0)),
                    init_noise=0.0,
                ).cuda()
                # Compatibility with the previous foreground-only buffer name.
                if "foreground_ids" in part_state and "gaussian_ids" not in part_state:
                    part_state = dict(part_state)
                    part_state["gaussian_ids"] = part_state.pop("foreground_ids")
                self.joint_part_field.load_state_dict(part_state, strict=True)
                self._joint_cluster_id = self.joint_part_field.initial_cluster_id_full().detach()
            else:
                cluster_id = state.get("joint_initial_cluster_id", state.get("joint_cluster_id", None))
                if cluster_id is not None:
                    self._joint_cluster_id = cluster_id.long().cuda().detach()
        if self.joint_part_field is None and state.get("dynamic_logit", None) is not None:
            self._dynamic_logit = nn.Parameter(state["dynamic_logit"].float().cuda().requires_grad_(True))
        if state.get("dynamic_foreground_mask", None) is not None:
            self._dynamic_foreground_mask = state["dynamic_foreground_mask"].bool().cuda().detach()
        elif self.joint_part_field is not None:
            # Exact joint branch assigns every starting Gaussian. Semantic
            # filtering is retained only as an auxiliary/debug field.
            self._dynamic_foreground_mask = torch.ones(self._xyz.shape[0], dtype=torch.bool, device="cuda")
        elif self._dynamic_logit is not None and self._dynamic_logit.numel() > 0:
            self._dynamic_foreground_mask = torch.ones(self._dynamic_logit.shape[0], dtype=torch.bool, device="cuda")
        self.dynamic_foreground_threshold = float(state.get("dynamic_foreground_threshold", 0.5))
        if state.get("dino_feat", None) is not None:
            self._dino_feat = state["dino_feat"].float().cuda().detach()
        if self.joint_single_rigid_enabled and self.joint_part_field is None and training_args is not None:
            self.initialize_joint_part_field(training_args)
        self._cell_id = self.motion_grid.gaussian_cell_id
        self.dynamic_enabled = True
        if training_args is not None:
            self.setup_dynamic_optimizers(training_args)

    def init_dynamic_motion(self, args, scene_extent=1.0, cameras=None):
        """Initialize velocity field, motion grid, dynamic logits, and DINO hints."""
        feature_dim = int(getattr(args, "motion_feature_dim", 32))
        hidden_dim = int(getattr(args, "motion_hidden_dim", 64))
        num_layers = int(getattr(args, "motion_num_layers", 3))
        time_freq = int(getattr(args, "motion_time_frequencies", 8))
        max_velocity = float(getattr(args, "motion_max_velocity", 0.0))
        voxel_size = float(getattr(args, "motion_voxel_size", -1.0))
        opacity_thr = float(getattr(args, "motion_opacity_threshold", 0.005))
        max_cells = int(getattr(args, "motion_max_initial_cells", 50000))
        xyz_scale = max(float(scene_extent), 1e-6)

        self.joint_single_rigid_enabled = bool(getattr(args, "joint_single_rigid_training", False))
        semantic_mask = self.build_dynamic_foreground_mask(args).detach().reshape(-1).bool()
        if self.joint_single_rigid_enabled:
            # The semantic mask isolates the foreground used by joint clustering,
            # static/moving assignment, and rigid-twist fitting.
            self._dynamic_foreground_mask = semantic_mask
            # The Eulerian proposal field remains available to every Gaussian.
            fg_mask = torch.ones_like(semantic_mask, dtype=torch.bool)
        else:
            fg_mask = semantic_mask
        min_fg = int(getattr(args, "semantic_dynamic_min_points", 32))
        # Run exact semantic-foreground XYZ+DINO initialization before allocating
        # the adaptive grid, motion MLP, rigid MLP, or expanded dynamic cameras.
        # The initializer itself is CPU-only, so the one-time N^2 operation cannot
        # consume or destabilize the training GPU.
        if self.joint_single_rigid_enabled:
            if (
                self._dino_feature_field is None
                or not torch.is_tensor(self._dino_feature_field)
                or self._dino_feature_field.ndim != 2
                or self._dino_feature_field.shape[0] != self._xyz.shape[0]
                or self._dino_feature_field.shape[1] <= 0
            ):
                raise RuntimeError(
                    "Joint one-rigid-part initialization requires the frozen static "
                    "DINO feature field for every Gaussian. Train/load the static "
                    "DINO field before enabling dynamic optimization."
                )
            self._dino_feature_field.requires_grad_(False)
            self._dynamic_logit = torch.empty(0, device=self._xyz.device)
            self.initialize_joint_part_field(args)

        self.motion_grid = AdaptiveMotionGrid.build_from_gaussians(
            self._xyz.detach(),
            opacity=self.get_opacity.detach(),
            feature_dim=feature_dim,
            voxel_size=voxel_size,
            opacity_threshold=opacity_thr,
            max_initial_cells=max_cells,
            foreground_mask=fg_mask,
            min_foreground_points=min_fg,
        ).cuda()
        self._cell_id = self.motion_grid.gaussian_cell_id
        self.motion_field = EulerianVelocityField(feature_dim, hidden_dim, num_layers,
                                                  time_freq, max_velocity, xyz_scale).cuda()
        self.joint_total_max_velocity = float(getattr(args, "joint_total_max_velocity", 3.0))
        self.joint_confident_static_threshold = float(getattr(args, "joint_confident_static_threshold", 0.20))
        self.joint_confident_dynamic_threshold = float(getattr(args, "joint_confident_dynamic_threshold", 0.80))
        if self.joint_single_rigid_enabled:
            self.rigid_twist_field = TimeVaryingRigidTwist(
                hidden_dim=int(getattr(args, "joint_twist_hidden_dim", 64)),
                num_layers=int(getattr(args, "joint_twist_num_layers", 3)),
                time_frequencies=int(getattr(args, "joint_twist_time_frequencies", 8)),
                max_omega=float(getattr(args, "joint_max_omega", 4.0)),
                max_translation=float(getattr(args, "joint_max_translation_velocity", 2.0)),
            ).cuda()
            self.rigid_center = self._xyz.detach()[self._dynamic_foreground_mask].mean(dim=0).detach()
        self.load_dino_features(getattr(args, "dino_features_path", ""), feature_dim=int(getattr(args, "dino_feature_dim", 0)))
        if not self.joint_single_rigid_enabled:
            init_logit = float(getattr(args, "dynamic_logit_init", -4.0))
            bg_logit = float(getattr(args, "semantic_dynamic_background_logit", -20.0))
            logits = torch.full((self._xyz.shape[0], 1), init_logit, device=self._xyz.device)
            logits[~fg_mask.reshape(-1), 0] = bg_logit
            self._dynamic_logit = nn.Parameter(logits)
        self.dynamic_enabled = True
        self.setup_dynamic_optimizers(args)

    def load_dino_features(self, path="", feature_dim=0):
        if path is not None and str(path) != "" and os.path.exists(path):
            data = torch.load(path, map_location="cuda")
            if isinstance(data, dict):
                for key in ["features", "dino", "dino_feat", "gaussian_features"]:
                    if key in data:
                        data = data[key]
                        break
            feat = torch.as_tensor(data, device="cuda", dtype=torch.float32)
            if feat.ndim != 2 or feat.shape[0] != self._xyz.shape[0]:
                raise ValueError("DINO feature tensor must have shape [num_gaussians, feature_dim].")
            self._dino_feat = feat.detach()
        elif int(feature_dim) > 0:
            self._dino_feat = torch.zeros(self._xyz.shape[0], int(feature_dim), device="cuda")
        else:
            self._dino_feat = torch.empty(self._xyz.shape[0], 0, device="cuda")

    def setup_dynamic_optimizers(self, args):
        if self.motion_field is None or self.motion_grid is None:
            return
        motion_mlp_lr = float(getattr(args, "motion_mlp_lr", 1e-3))
        motion_grid_lr = float(getattr(args, "motion_grid_lr", 1e-2))
        dynamic_lr = float(getattr(args, "dynamic_lr", 1e-3))
        motion_groups = [
            {"params": self.motion_field.parameters(), "lr": motion_mlp_lr, "name": "motion_mlp_residual"},
            {"params": [self.motion_grid.features], "lr": motion_grid_lr, "name": "motion_grid_features"},
        ]
        if self.joint_single_rigid_enabled and self.rigid_twist_field is not None:
            motion_groups.append({
                "params": self.rigid_twist_field.parameters(),
                "lr": float(getattr(args, "joint_twist_lr", 5e-4)),
                "name": "joint_rigid_twist",
            })
        self.motion_optimizer = torch.optim.Adam(motion_groups, eps=1e-15)
        if self.joint_single_rigid_enabled and self.joint_part_field is not None:
            self.dynamic_optimizer = torch.optim.Adam([
                {
                    "params": [self.joint_part_field.point_field],
                    "lr": float(getattr(args, "joint_part_field_lr", dynamic_lr)),
                    "name": "joint_point_part_field",
                },
                {
                    "params": [self.joint_part_field.prototypes],
                    "lr": float(getattr(args, "joint_part_prototype_lr", dynamic_lr)),
                    "name": "joint_part_prototypes",
                },
            ], eps=1e-15)
        else:
            self.dynamic_optimizer = torch.optim.Adam([
                {"params": [self._dynamic_logit], "lr": dynamic_lr, "name": "dynamic_logits"},
            ], eps=1e-15)

        # DINO descriptors are learned in the static stage and frozen in the
        # dynamic stage. Dynamic DINO supervision should move transported
        # Gaussians by training motion_field/motion_grid, not adapt z_i.
        if self._dino_feature_field is not None and self._dino_feature_field.numel() > 0:
            if self.joint_single_rigid_enabled:
                # Exact initialization and all later diagnostics refer to the
                # same statically trained per-Gaussian DINO field.
                self._dino_feature_field.requires_grad_(False)
            else:
                self._dino_feature_field.requires_grad_(not bool(getattr(args, "freeze_dino_feature_dynamic", True)))
        self.dino_feature_optimizer = None

    @torch.no_grad()
    def initialize_joint_part_field(self, args):
        """Initialize static + one moving part from the exact XYZ+DINO matrix.

        Every starting Gaussian contributes its canonical center and its frozen
        statically trained DINO descriptor. The full precomputed
        Euclidean+cosine matrix is retained for debugging and k-medoids uses
        this matrix directly.
        """
        dino = self._dino_feature_field
        semantic_mask = self.get_dynamic_foreground_mask.detach().reshape(-1).bool()
        if semantic_mask.numel() != self._xyz.shape[0]:
            raise RuntimeError("Semantic foreground mask length does not match Gaussian count")
        if int(semantic_mask.sum().item()) < 2:
            raise RuntimeError("Semantic foreground is too small for static/moving joint initialization")
        result = initialize_joint_parts_from_static_dino(
            canonical_xyz=self._xyz.detach().cpu(),
            static_dino_field=dino.detach().cpu(),
            foreground_mask=semantic_mask.detach().cpu(),
            num_moving_parts=int(getattr(args, "joint_num_moving_parts", 1)),
            xyz_weight=float(getattr(args, "joint_init_euclidean_weight", 1.0)),
            dino_weight=float(getattr(args, "joint_init_cosine_weight", 1.0)),
            xyz_scale=float(getattr(args, "joint_init_euclidean_scale", -1.0)),
            distance_block_size=int(getattr(args, "joint_init_distance_block_size", 512)),
            distance_device=str(getattr(args, "joint_init_distance_device", "cpu")),
            distance_storage=str(getattr(args, "joint_init_distance_storage", "float16")),
            kmedoids_iterations=int(getattr(args, "joint_init_kmedoids_iters", 12)),
        )
        self._joint_initialization_result = result
        self._joint_initialization_config = {
            "xyz_weight": float(getattr(args, "joint_init_euclidean_weight", 1.0)),
            "dino_weight": float(getattr(args, "joint_init_cosine_weight", 1.0)),
            "save_distance_matrix": bool(getattr(args, "joint_save_initial_distance_matrix", True)),
        }
        self.joint_part_field = JointPartField(
            num_gaussians=int(self._xyz.shape[0]),
            foreground_ids=result.foreground_ids,
            initial_cluster_id_fg=result.initial_cluster_id_fg,
            num_moving_parts=int(getattr(args, "joint_num_moving_parts", 1)),
            embedding_dim=int(getattr(args, "joint_part_field_dim", 8)),
            assignment_temperature=float(getattr(args, "joint_part_assignment_temperature", 1.0)),
            init_noise=float(getattr(args, "joint_part_field_init_noise", 0.02)),
        ).to(self._xyz.device)
        self._joint_cluster_id = self.joint_part_field.initial_cluster_id_full().detach()
        expected_ids = torch.where(semantic_mask)[0]
        actual_ids = self.joint_part_field.foreground_ids.to(expected_ids.device)
        if expected_ids.numel() != actual_ids.numel() or not torch.equal(expected_ids, actual_ids):
            raise RuntimeError(
                f"Semantic foreground mismatch: expected {expected_ids.numel()} IDs, "
                f"JointPartField contains {actual_ids.numel()} IDs"
            )
        print(
            f"[Joint init] semantic foreground only: {actual_ids.numel()} / "
            f"{self._xyz.shape[0]} Gaussians"
        )
        return self._joint_cluster_id

    @torch.no_grad()
    def initialize_joint_clusters(self, args):
        """Compatibility alias for the exact per-Gaussian initializer."""
        return self.initialize_joint_part_field(args)

    @torch.no_grad()
    def export_joint_initialization(self, output_dir):
        if self._joint_initialization_result is None:
            return
        cfg = self._joint_initialization_config
        save_joint_initialization(
            output_dir=str(output_dir),
            result=self._joint_initialization_result,
            xyz_weight=float(cfg.get("xyz_weight", 1.0)),
            dino_weight=float(cfg.get("dino_weight", 1.0)),
            save_distance_matrix=bool(cfg.get("save_distance_matrix", True)),
        )
        if self.joint_part_field is not None:
            with open(os.path.join(str(output_dir), "joint_part_field_initial_summary.json"), "w", encoding="utf-8") as f:
                json.dump(self.joint_part_field.summary(), f, indent=2)
        # The full N x N matrix has now been persisted. Release the temporary
        # in-memory copy before dynamic training continues.
        self._joint_initialization_result = None

    def set_joint_training_progress(self, euler_iter, args):
        """Continuous rigid handover; all joint losses remain active throughout."""
        steps = max(1, int(getattr(args, "joint_rigid_handover_iters", 3000)))
        start = int(getattr(args, "joint_rigid_handover_start", 0))
        value = (float(euler_iter) - float(start)) / float(steps)
        self.joint_rigid_handover = max(0.0, min(1.0, value))

    def query_joint_velocity(self, x, t, gaussian_cell_id=None, dynamic_prob=None):
        """Query proposal and rigid experts without suppressing uncertain motion.

        p=0   -> confident static, zero transport
        p=0.5 -> uncertain, full free proposal for rendering evidence
        p=1   -> confident moving, rigid expert (after continuous handover)

        Both proposal and rigid models are trained at every dynamic iteration;
        handover is a continuous trust ramp, not a separated training phase.
        """
        proposal = self.motion_field.query_velocity(
            x, t, grid=self.motion_grid, gaussian_cell_id=gaussian_cell_id
        )
        if not self.joint_single_rigid_enabled or self.rigid_twist_field is None:
            zero = torch.zeros_like(proposal)
            return proposal, zero, proposal, None, None

        rigid, omega, trans = self.rigid_twist_field.velocity(x, t, self.rigid_center)
        p = (
            torch.full((x.shape[0], 1), 0.5, device=x.device, dtype=x.dtype)
            if dynamic_prob is None
            else dynamic_prob.reshape(-1, 1).to(device=x.device, dtype=x.dtype).detach().clamp(0.0, 1.0)
        )
        # Do not suppress motion merely because a cluster was initialized on
        # the static side. Only assignments beyond the same confidence
        # thresholds used by the losses can hand transport to an expert.
        static_thr = max(float(self.joint_confident_static_threshold), 1e-4)
        dynamic_thr = min(float(self.joint_confident_dynamic_threshold), 1.0 - 1e-4)
        static_conf = ((static_thr - p) / static_thr).clamp(0.0, 1.0)
        moving_conf = ((p - dynamic_thr) / (1.0 - dynamic_thr)).clamp(0.0, 1.0)
        uncertain = (1.0 - static_conf - moving_conf).clamp(0.0, 1.0)
        h = float(max(0.0, min(1.0, self.joint_rigid_handover)))
        moving_velocity = (1.0 - h) * proposal + h * rigid
        total = uncertain * proposal + moving_conf * moving_velocity
        if self.joint_total_max_velocity > 0:
            norm = torch.linalg.norm(total, dim=-1, keepdim=True).clamp_min(1e-8)
            total = total * torch.clamp(self.joint_total_max_velocity / norm, max=1.0)
        return total, rigid, proposal, omega, trans

    def integrate_joint_positions(self, x0, t0, t1, steps=4, gaussian_cell_id=None, dynamic_prob=None, return_components=False):
        steps = max(1, int(steps))
        x = x0
        dt = (float(t1) - float(t0)) / float(steps)
        total = rigid = proposal = torch.zeros_like(x0)
        omega = trans = None
        for step in range(steps):
            ts = float(t0) + (step + 0.5) * dt
            total, rigid, proposal, omega, trans = self.query_joint_velocity(x, ts, gaussian_cell_id, dynamic_prob)
            x = x + dt * total
        if return_components:
            return x, total, rigid, proposal, omega, trans
        return x, total

    def rebuild_dynamic_optimizers(self, args):
        """Rebuild after grid topology changes while preserving Adam state.

        Only motion_grid.features changes identity/shape. The motion MLP, explicit
        twist field, per-Gaussian part field, and prototypes keep their optimizer
        moments across grid split/merge updates.
        """
        old_motion = self.motion_optimizer
        old_dynamic = self.dynamic_optimizer
        motion_state = {} if old_motion is None else {param: state for param, state in old_motion.state.items()}
        dynamic_state = {} if old_dynamic is None else {param: state for param, state in old_dynamic.state.items()}
        self.setup_dynamic_optimizers(args)
        if self.motion_optimizer is not None:
            for group in self.motion_optimizer.param_groups:
                for param in group["params"]:
                    if param in motion_state:
                        self.motion_optimizer.state[param] = motion_state[param]
        if self.dynamic_optimizer is not None:
            for group in self.dynamic_optimizer.param_groups:
                for param in group["params"]:
                    if param in dynamic_state:
                        self.dynamic_optimizer.state[param] = dynamic_state[param]

    def freeze_canonical(self, freeze_geometry=True, freeze_appearance=True):
        if freeze_geometry:
            self._xyz.requires_grad_(False)
            self._scaling.requires_grad_(False)
            self._rotation.requires_grad_(False)
        if freeze_appearance:
            self._features_dc.requires_grad_(False)
            self._features_rest.requires_grad_(False)
            self._opacity.requires_grad_(False)

    def unfreeze_canonical_appearance(self):
        self._features_dc.requires_grad_(True)
        self._features_rest.requires_grad_(True)
        self._opacity.requires_grad_(True)

    def set_time(self, t, detach_canonical=True, steps=4):
        """Cache advected tensors before calling the original renderer.

        Semantic foreground uses the static/moving joint model. Semantic
        background keeps the unrestricted Eulerian proposal velocity and never
        enters the rigid joint fit.
        """
        if not self.dynamic_enabled or self.motion_field is None:
            self.clear_time()
            return

        x0 = self._xyz.detach() if detach_canonical else self._xyz
        fg = self.get_dynamic_foreground_mask.reshape(-1).bool()
        xyz_t = x0.clone()
        vel_t = torch.zeros_like(x0)

        if self.joint_single_rigid_enabled:
            rigid_t = torch.zeros_like(x0)
            proposal_t = torch.zeros_like(x0)

            if fg.any():
                fg_cell_id = self.motion_grid.gaussian_cell_id[fg] if self.motion_grid is not None else None
                p_fg = self.get_dynamic_probability[fg] if self.get_dynamic_probability is not None else None
                xyz_fg, vel_fg, rigid_fg, proposal_fg, omega, trans = self.integrate_joint_positions(
                    x0[fg], 0.0, float(t), int(steps), fg_cell_id, p_fg, return_components=True
                )
                xyz_t[fg] = xyz_fg
                vel_t[fg] = vel_fg
                rigid_t[fg] = rigid_fg
                proposal_t[fg] = proposal_fg
                self._current_omega = omega
                self._current_translation = trans
            else:
                self._current_omega = None
                self._current_translation = None

            bg = ~fg
            if bg.any():
                bg_cell_id = self.motion_grid.gaussian_cell_id[bg] if self.motion_grid is not None else None
                xyz_bg, proposal_bg = self.motion_field.integrate(
                    x0=x0[bg], t0=0.0, t1=float(t), steps=int(steps),
                    grid=self.motion_grid, gaussian_cell_id=bg_cell_id,
                )
                xyz_t[bg] = xyz_bg
                vel_t[bg] = proposal_bg
                proposal_t[bg] = proposal_bg

            self._current_rigid_velocity = rigid_t
            self._current_residual_velocity = proposal_t
        else:
            if fg.any():
                cell_id = self.motion_grid.gaussian_cell_id[fg] if self.motion_grid is not None else None
                xyz_fg, vel_fg = self.motion_field.integrate(
                    x0=x0[fg], t0=0.0, t1=float(t), steps=int(steps),
                    grid=self.motion_grid, gaussian_cell_id=cell_id,
                )
                xyz_t[fg] = xyz_fg
                vel_t[fg] = vel_fg

        self._current_xyz = xyz_t
        self._current_velocity = vel_t
        self._current_rotation = self._rotation.detach() if detach_canonical else self._rotation
        self._current_time = float(t)

    def integrate_positions(self, x_start, t0, t1, steps=4, gaussian_cell_id=None):
        if self.motion_field is None:
            return x_start, torch.zeros_like(x_start)
        if gaussian_cell_id is None:
            gaussian_cell_id = self.motion_grid.gaussian_cell_id if self.motion_grid is not None else None
            if gaussian_cell_id is not None and gaussian_cell_id.numel() != x_start.shape[0]:
                fg = self.get_dynamic_foreground_mask
                if fg.numel() == gaussian_cell_id.numel() and int(fg.sum().item()) == x_start.shape[0]:
                    gaussian_cell_id = gaussian_cell_id[fg]
        if self.joint_single_rigid_enabled:
            p = self.get_dynamic_probability
            if p is not None and p.shape[0] != x_start.shape[0]:
                fg = self.get_dynamic_foreground_mask
                p = p[fg] if int(fg.sum().item()) == x_start.shape[0] else None
            return self.integrate_joint_positions(x_start, float(t0), float(t1), int(steps), gaussian_cell_id, p)
        return self.motion_field.integrate(x_start, float(t0), float(t1), int(steps),
                                           grid=self.motion_grid, gaussian_cell_id=gaussian_cell_id)

    def clear_time(self):
        self._current_xyz = None
        self._current_rotation = None
        self._current_velocity = None
        self._current_rigid_velocity = None
        self._current_residual_velocity = None
        self._current_omega = None
        self._current_translation = None
        self._current_time = None

    @torch.no_grad()
    def export_time_tracks(self, times, frame_ids=None, steps=4, dtype=torch.float32, device="cpu"):
        """Export canonical-to-time Gaussian tracks for visualization/evaluation.

        Returns a dict with:
            positions:          [T,N,3]
            velocities:         [T,N,3]
            canonical_positions:[N,3]
            frame_ids:          [T]
            times:              [T]
            dynamic_prob:       [N,1] if available
            cell_id:            [N]   if available
            semantic_prob/logit and dino_feature_field if available

        This function does not mutate the current renderer cache.  It directly
        integrates the Eulerian velocity field from t=0 to each requested time.
        """
        if times is None:
            times = []
        times = [float(t) for t in times]
        if frame_ids is None:
            frame_ids = list(range(len(times)))
        frame_ids = [int(f) for f in frame_ids]
        if len(frame_ids) != len(times):
            raise ValueError(f"frame_ids length {len(frame_ids)} != times length {len(times)}")

        out_device = torch.device(device)
        x0 = self._xyz.detach()
        n = int(x0.shape[0])
        pos_list = []
        vel_list = []
        rigid_vel_list = []
        residual_vel_list = []
        omega_list = []
        trans_list = []

        if self.dynamic_enabled and self.motion_field is not None and self.motion_grid is not None:
            fg = self.get_dynamic_foreground_mask
            for t in times:
                xt = x0.clone()
                vt = torch.zeros_like(x0)
                if fg.any():
                    cid = self.motion_grid.gaussian_cell_id[fg]
                    p_fg = self.get_dynamic_probability[fg] if self.get_dynamic_probability is not None else None
                    if self.joint_single_rigid_enabled:
                        xt_fg, vt_fg, rv_fg, res_fg, omega, trans = self.integrate_joint_positions(
                            x0[fg], 0.0, float(t), int(steps), cid, p_fg, return_components=True
                        )
                        rigid_full = torch.zeros_like(x0)
                        residual_full = torch.zeros_like(x0)
                        rigid_full[fg] = rv_fg
                        residual_full[fg] = res_fg
                        rigid_vel_list.append(rigid_full.detach().to(device=out_device, dtype=dtype))
                        residual_vel_list.append(residual_full.detach().to(device=out_device, dtype=dtype))
                        omega_list.append(omega.detach().to(device=out_device, dtype=torch.float32))
                        trans_list.append(trans.detach().to(device=out_device, dtype=torch.float32))
                    else:
                        xt_fg, vt_fg = self.motion_field.integrate(
                            x0=x0[fg], t0=0.0, t1=float(t), steps=int(steps),
                            grid=self.motion_grid, gaussian_cell_id=cid,
                        )
                    xt[fg] = xt_fg
                    vt[fg] = vt_fg
                pos_list.append(xt.detach().to(device=out_device, dtype=dtype))
                vel_list.append(vt.detach().to(device=out_device, dtype=dtype))
        else:
            zero_v = torch.zeros_like(x0)
            for _ in times:
                pos_list.append(x0.to(device=out_device, dtype=dtype))
                vel_list.append(zero_v.to(device=out_device, dtype=dtype))

        if pos_list:
            positions = torch.stack(pos_list, dim=0)
            velocities = torch.stack(vel_list, dim=0)
        else:
            positions = torch.empty((0, n, 3), device=out_device, dtype=dtype)
            velocities = torch.empty((0, n, 3), device=out_device, dtype=dtype)

        tracks = {
            "positions": positions,
            "velocities": velocities,
            "canonical_positions": x0.to(device=out_device, dtype=dtype),
            "frame_ids": torch.tensor(frame_ids, dtype=torch.long, device=out_device),
            "times": torch.tensor(times, dtype=torch.float32, device=out_device),
            "opacity": self.get_opacity.detach().to(device=out_device, dtype=torch.float32),
            "scaling": self.get_scaling.detach().to(device=out_device, dtype=torch.float32),
            "rotation": self.get_rotation.detach().to(device=out_device, dtype=torch.float32),
        }
        if rigid_vel_list:
            tracks["rigid_velocities"] = torch.stack(rigid_vel_list, dim=0)
            tracks["proposal_velocities"] = torch.stack(residual_vel_list, dim=0)
            tracks["residual_velocities"] = tracks["proposal_velocities"]  # compatibility
            tracks["rigid_omega"] = torch.stack(omega_list, dim=0)
            tracks["rigid_translation"] = torch.stack(trans_list, dim=0)
            tracks["rigid_center"] = self.rigid_center.detach().to(device=out_device, dtype=torch.float32)
        current_cluster = self.get_joint_cluster_id
        if current_cluster is not None and torch.is_tensor(current_cluster) and current_cluster.numel() == n:
            tracks["joint_cluster_id"] = current_cluster.detach().to(device=out_device, dtype=torch.long)
        initial_cluster = self.get_joint_initial_cluster_id
        if initial_cluster is not None and torch.is_tensor(initial_cluster) and initial_cluster.numel() == n:
            tracks["joint_initial_cluster_id"] = initial_cluster.detach().to(device=out_device, dtype=torch.long)
        if self.joint_part_field is not None:
            tracks["joint_part_assignments"] = self.joint_part_field.full_assignments().detach().to(device=out_device, dtype=torch.float32)
            tracks["joint_part_field"] = self.joint_part_field.normalized_point_field().detach().to(device=out_device, dtype=torch.float32)
            tracks["joint_part_field_gaussian_ids"] = self.joint_part_field.gaussian_ids.detach().to(device=out_device, dtype=torch.long)
            tracks["joint_part_field_foreground_ids"] = self.joint_part_field.gaussian_ids.detach().to(device=out_device, dtype=torch.long)  # compatibility
            tracks["joint_part_prototypes"] = self.joint_part_field.normalized_prototypes().detach().to(device=out_device, dtype=torch.float32)
            tracks["joint_rigid_handover"] = torch.tensor(float(self.joint_rigid_handover), device=out_device, dtype=torch.float32)
            tracks["joint_confident_static_threshold"] = torch.tensor(float(self.joint_confident_static_threshold), device=out_device, dtype=torch.float32)
            tracks["joint_confident_dynamic_threshold"] = torch.tensor(float(self.joint_confident_dynamic_threshold), device=out_device, dtype=torch.float32)

        dyn_prob = self.get_dynamic_probability
        if dyn_prob is not None:
            tracks["dynamic_prob"] = dyn_prob.detach().to(device=out_device, dtype=torch.float32)

        cell_id = self.get_cell_id
        if cell_id is not None and torch.is_tensor(cell_id) and cell_id.numel() == n:
            tracks["cell_id"] = cell_id.detach().to(device=out_device, dtype=torch.long)

        fg = self.get_dynamic_foreground_mask
        if fg is not None and fg.numel() == n:
            tracks["dynamic_foreground_mask"] = fg.detach().to(device=out_device, dtype=torch.bool)
            tracks["dynamic_foreground_threshold"] = torch.tensor(float(getattr(self, "dynamic_foreground_threshold", 0.5)), device=out_device)

        if self._semantic_logits is not None and self._semantic_logits.numel() > 0:
            sem = self._semantic_logits.detach()
            tracks["semantic_logit"] = sem.to(device=out_device, dtype=torch.float32)
            tracks["semantic_prob"] = torch.sigmoid(sem).to(device=out_device, dtype=torch.float32)

        if self._dino_feature_field is not None and self._dino_feature_field.numel() > 0:
            tracks["dino_feature_field"] = self._dino_feature_field.detach().to(device=out_device, dtype=torch.float32)

        return tracks


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        # Learned auxiliary per-Gaussian fields.
        # These are embedded directly into point_cloud.ply, so each Gaussian
        # carries its semantic and DINO feature state.
        sem = getattr(self, "_semantic_logits", None)
        if sem is not None and sem.numel() > 0:
            l.append('semantic_logit')
            l.append('semantic_prob')

        din = getattr(self, "_dino_feature_field", None)
        if din is not None and din.numel() > 0 and din.ndim == 2:
            for i in range(din.shape[1]):
                l.append('dino_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        aux_arrays = []

        # Embed semantic field directly in every Gaussian/PLY vertex.
        sem = getattr(self, "_semantic_logits", None)
        if sem is not None and sem.numel() > 0:
            sem_cpu = sem.detach().float().cpu()
            if sem_cpu.ndim == 2 and sem_cpu.shape[1] == 1:
                sem_cpu = sem_cpu[:, 0]
            if sem_cpu.ndim != 1 or sem_cpu.shape[0] != xyz.shape[0]:
                raise ValueError(
                    f"Invalid _semantic_logits shape for save_ply: "
                    f"{tuple(sem_cpu.shape)} vs N={xyz.shape[0]}"
                )
            sem_logit = sem_cpu.numpy()[:, None]
            sem_prob = torch.sigmoid(sem_cpu).numpy()[:, None]
            aux_arrays.extend([sem_logit, sem_prob])

        # Embed DINO feature field directly in every Gaussian/PLY vertex.
        din = getattr(self, "_dino_feature_field", None)
        if din is not None and din.numel() > 0:
            din_cpu = din.detach().float().cpu()
            if din_cpu.ndim != 2 or din_cpu.shape[0] != xyz.shape[0]:
                raise ValueError(
                    f"Invalid _dino_feature_field shape for save_ply: "
                    f"{tuple(din_cpu.shape)} vs N={xyz.shape[0]}"
                )
            aux_arrays.append(din_cpu.numpy())

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, *aux_arrays), axis=1)
        if attributes.shape[1] != len(dtype_full):
            raise RuntimeError(
                f"PLY attribute mismatch: attributes={attributes.shape[1]}, dtype={len(dtype_full)}. "
                f"Fields={[name for name, _ in dtype_full]}"
            )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        # Also keep compact sidecars for fast loading/debugging and backward compatibility.
        if sem is not None and sem.numel() > 0:
            torch.save(sem.detach().cpu(), os.path.join(os.path.dirname(path), "semantic_logits.pt"))
        if din is not None and din.numel() > 0:
            torch.save(din.detach().cpu(), os.path.join(os.path.dirname(path), "dino_feature_field.pt"))

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))

        # Load auxiliary semantic/DINO fields. Prefer embedded PLY properties;
        # fall back to the old sidecar tensors for backward compatibility.
        prop_names = [p.name for p in plydata.elements[0].properties]
        sidecar_dir = os.path.dirname(path)
        sem_path = os.path.join(sidecar_dir, "semantic_logits.pt")
        dino_path = os.path.join(sidecar_dir, "dino_feature_field.pt")

        if "semantic_logit" in prop_names:
            sem_np = np.asarray(plydata.elements[0]["semantic_logit"], dtype=np.float32)[:, None]
            self._semantic_logits = nn.Parameter(
                torch.tensor(sem_np, dtype=torch.float, device="cuda").contiguous().requires_grad_(True)
            )
        elif os.path.exists(sem_path):
            sem = torch.load(sem_path, map_location="cuda").float()
            if sem.ndim == 1:
                sem = sem[:, None]
            if sem.shape[0] == xyz.shape[0]:
                self._semantic_logits = nn.Parameter(sem.contiguous().requires_grad_(True))
            else:
                self._semantic_logits = nn.Parameter(torch.zeros((xyz.shape[0], 1), device="cuda").requires_grad_(True))
        else:
            self._semantic_logits = nn.Parameter(torch.zeros((xyz.shape[0], 1), device="cuda").requires_grad_(True))

        dino_names = [name for name in prop_names if name.startswith("dino_")]
        dino_names = sorted(dino_names, key=lambda x: int(x.split('_')[-1]))
        if len(dino_names) > 0:
            din_np = np.stack(
                [np.asarray(plydata.elements[0][name], dtype=np.float32) for name in dino_names],
                axis=1,
            )
            self._dino_feature_field = nn.Parameter(
                torch.tensor(din_np, dtype=torch.float, device="cuda").contiguous().requires_grad_(True)
            )
        elif os.path.exists(dino_path):
            din = torch.load(dino_path, map_location="cuda").float()
            if din.ndim == 2 and din.shape[0] == xyz.shape[0]:
                self._dino_feature_field = nn.Parameter(din.contiguous().requires_grad_(True))
            else:
                self._dino_feature_field = torch.empty((xyz.shape[0], 0), device="cuda")
        else:
            self._dino_feature_field = torch.empty((xyz.shape[0], 0), device="cuda")

        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "semantic_logits" in optimizable_tensors:
            self._semantic_logits = optimizable_tensors["semantic_logits"]
        elif self._semantic_logits is not None and self._semantic_logits.numel() > 0:
            self._semantic_logits = nn.Parameter(self._semantic_logits[valid_points_mask].requires_grad_(True))
        if "dino_feature" in optimizable_tensors:
            self._dino_feature_field = optimizable_tensors["dino_feature"]
        elif self._dino_feature_field is not None and self._dino_feature_field.numel() > 0:
            self._dino_feature_field = nn.Parameter(self._dino_feature_field[valid_points_mask].requires_grad_(True))

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] not in tensors_dict:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        sel = getattr(self, "_last_densify_selected_mask", None)
        rep = int(getattr(self, "_last_densify_repeat", 1))
        if sel is not None and self._semantic_logits is not None and self._semantic_logits.numel() > 0:
            d["semantic_logits"] = self._semantic_logits[sel].repeat(rep, 1)
        if sel is not None and self._dino_feature_field is not None and self._dino_feature_field.numel() > 0:
            d["dino_feature"] = self._dino_feature_field[sel].repeat(rep, 1)

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "semantic_logits" in optimizable_tensors:
            self._semantic_logits = optimizable_tensors["semantic_logits"]
        if "dino_feature" in optimizable_tensors:
            self._dino_feature_field = optimizable_tensors["dino_feature"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self._last_densify_selected_mask = selected_pts_mask
        self._last_densify_repeat = N
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self._last_densify_selected_mask = selected_pts_mask
        self._last_densify_repeat = 1
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
