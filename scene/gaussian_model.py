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
from contextlib import contextmanager
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
from scene.fixed_curl_grid import FixedCurlGrid
from scene.material_assignment_field import MaterialAssignmentField
from utils.dynamic_loss_utils import (
    aggregate_point_features_to_cells,
    build_occupied_cell_knn_pair_graph,
    aggregate_point_probabilities_to_cells,
    eulerian_velocity_gradient_rigidity_loss,
    integrate_world_angular_velocity_quaternion,
)
from utils.rigid_killing_motion import (
    RigidKillingMode,
    integrate_rigid_mode_exact,
    predict_rigid_cell_velocity,
)


RIGID_KILLING_REPRESENTATIONS = {
    "rigid_killing_1d",
    "rigid_killing_eulerian",
    "rigid_killing_lagrangian",
}

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
        self._sam3_feature_field = torch.empty(0)
        self._sam3_affinity_field = torch.empty(0)
        self._sam3_affinity_training_config = {}
        self._sam3_affinity_frame_idx = None
        self._gt_part_labels = torch.empty(0, dtype=torch.long)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # Fixed Eulerian state. The renderer reads transported tensors through
        # get_xyz/get_rotation.
        self.curl_grid = None
        self.material_assignment = None
        self._eulerian_strain_config = {
            "neighbors": 12,
            "radius_cells": 3.5,
            "min_neighbors": 3,
            "ridge": 1.0e-4,
            "same_part_floor": 0.05,
            "velocity_scale": 1.0,
            "velocity_spatial_robust_delta": 0.25,
            "static_gate_threshold": 0.85,
            "static_cell_purity": 0.95,
        }
        self.motion_representation = "eulerian_curl_strain"
        self.motion_optimizer = None
        self.membership_optimizer = None
        self.motion_time_mode = "seconds"
        self.rigid_reference_point = torch.empty(0)
        self.rigid_reference_velocity = torch.empty(0)
        self.rigid_angular_mode = torch.empty(0)
        self.rigid_interval_speeds = torch.empty(0)
        self.rigid_frame_times = torch.empty(0)
        self.rigid_dynamic_probability = torch.empty(0)
        self.rigid_effective_dynamic_mask = torch.empty(0, dtype=torch.bool)
        self.rigid_solver_diagnostics = {}
        self.kpart_joint_state = None
        self.sequence_assignment_state = None
        self.render_discovery_state = None
        self.dynamic_prior_metadata = {}
        self._current_cell_id = torch.empty(0, dtype=torch.long)
        self._current_xyz = None
        self._current_rotation = None
        self._current_velocity = None
        self._current_time = None
        self.dynamic_enabled = False
        self._dynamic_foreground_mask = torch.empty(0, dtype=torch.bool)
        self._motion_evidence = torch.empty(0)
        self._static_evidence = torch.empty(0)
        self._initial_dynamic_mask = torch.empty(0, dtype=torch.bool)
        self._foreground_source_indices = torch.empty(0, dtype=torch.long)
        self.dynamic_foreground_threshold = 0.5
        self.setup_functions()

    def capture_aux_state(self):
        """Checkpoint learnable static auxiliary fields.

        PLY saving already embeds these fields, but checkpoints should preserve
        them too.  Older checkpoints without this state still load normally.
        """
        return {
            "semantic_logits": self._semantic_logits.detach()
            if self._semantic_logits is not None and self._semantic_logits.numel() > 0
            else None,
            "dino_feature_field": self._dino_feature_field.detach()
            if self._dino_feature_field is not None
            and self._dino_feature_field.numel() > 0
            else None,
            "sam3_feature_field": self._sam3_feature_field.detach()
            if self._sam3_feature_field is not None
            and self._sam3_feature_field.numel() > 0
            else None,
            "sam3_affinity_field": self._sam3_affinity_field.detach()
            if self._sam3_affinity_field is not None
            and torch.is_tensor(self._sam3_affinity_field)
            and self._sam3_affinity_field.numel() > 0
            else None,
            "gt_part_labels": self._gt_part_labels.detach().cpu()
            if self._gt_part_labels is not None
            and torch.is_tensor(self._gt_part_labels)
            and self._gt_part_labels.numel() > 0
            else None,
        }

    def load_aux_state(self, state):
        if state is None:
            return
        sem = state.get("semantic_logits", None) if isinstance(state, dict) else None
        if sem is not None and sem.numel() > 0:
            self._semantic_logits = nn.Parameter(
                sem.float().cuda().contiguous().requires_grad_(True)
            )
        din = state.get("dino_feature_field", None) if isinstance(state, dict) else None
        if din is not None and din.numel() > 0:
            self._dino_feature_field = nn.Parameter(
                din.float().cuda().contiguous().requires_grad_(True)
            )
        sam = state.get("sam3_feature_field", None) if isinstance(state, dict) else None
        if sam is not None and sam.numel() > 0:
            self.set_sam3_feature_field(sam.float().cuda(), requires_grad=True)
        sam_aff = (
            state.get("sam3_affinity_field", None) if isinstance(state, dict) else None
        )
        if sam_aff is not None and sam_aff.numel() > 0:
            self.set_sam3_affinity_field(sam_aff.float().cuda(), requires_grad=True)
        gt = state.get("gt_part_labels", None) if isinstance(state, dict) else None
        if gt is not None and torch.as_tensor(gt).numel() > 0:
            self._gt_part_labels = torch.as_tensor(gt, dtype=torch.long, device="cuda").contiguous()

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
            (
                self.active_sh_degree,
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
                aux_state,
            ) = model_args
        elif len(model_args) == 13:
            (
                self.active_sh_degree,
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
            ) = model_args
        else:
            (
                self.active_sh_degree,
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
            ) = model_args
        # Auxiliary checkpoint tensors replace Parameter objects, so restore
        # them before constructing the optimizer parameter groups.
        if aux_state is not None:
            self.load_aux_state(aux_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
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
    def get_cell_id(self):
        return self._current_cell_id

    @property
    def get_dino_feature_field(self):
        return self._dino_feature_field

    @property
    def get_sam3_feature_field(self):
        return self._sam3_feature_field

    @property
    def get_sam3_features(self):
        return self._sam3_feature_field

    @property
    def get_sam3_affinity(self):
        return self._sam3_affinity_field

    def _assert_aux_row_alignment(self, name, value=None):
        tensor = getattr(self, name) if value is None else value
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        if tensor.ndim < 1 or int(tensor.shape[0]) != int(self._xyz.shape[0]):
            raise RuntimeError(
                f"{name} row alignment broken: {tuple(tensor.shape)} for "
                f"{int(self._xyz.shape[0])} Gaussians"
            )

    def init_sam3_feature_field(self, dim, *, init_random=True, requires_grad=True):
        dim = int(dim or 0)
        n = int(self._xyz.shape[0]) if self._xyz.numel() else 0
        if n <= 0:
            return
        if dim <= 0:
            self._sam3_feature_field = torch.empty((n, 0), device=self._xyz.device)
            return
        device = self._xyz.device
        current = getattr(self, "_sam3_feature_field", None)
        if (
            current is None
            or not torch.is_tensor(current)
            or current.numel() == 0
            or tuple(current.shape) != (n, dim)
        ):
            if init_random:
                feature = torch.randn((n, dim), device=device, dtype=torch.float32) * 0.01
                feature = torch.nn.functional.normalize(feature, dim=-1)
            else:
                feature = torch.zeros((n, dim), device=device, dtype=torch.float32)
            self._sam3_feature_field = nn.Parameter(
                feature.contiguous().requires_grad_(bool(requires_grad))
            )
        self._assert_aux_row_alignment("_sam3_feature_field")

    def set_sam3_feature_field(self, feature, *, requires_grad=True):
        value = torch.as_tensor(feature, device=self._xyz.device, dtype=torch.float32)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2:
            raise ValueError(f"SAM3 feature field must be [N,C], got {tuple(value.shape)}")
        if int(value.shape[0]) != int(self._xyz.shape[0]):
            raise ValueError(
                f"SAM3 feature rows {int(value.shape[0])} do not match "
                f"Gaussian rows {int(self._xyz.shape[0])}"
            )
        if value.shape[1] > 0:
            value = torch.nn.functional.normalize(value.float(), dim=-1)
            self._sam3_feature_field = nn.Parameter(
                value.contiguous().requires_grad_(bool(requires_grad))
            )
        else:
            self._sam3_feature_field = torch.empty((value.shape[0], 0), device=self._xyz.device)
        self._assert_aux_row_alignment("_sam3_feature_field")

    def save_sam3_feature_field(self, path):
        sam = getattr(self, "_sam3_feature_field", None)
        if sam is None or not torch.is_tensor(sam) or sam.numel() == 0:
            return
        torch.save(sam.detach().cpu(), path)

    def load_sam3_feature_field(self, path, *, requires_grad=True):
        sam = torch.load(path, map_location=self._xyz.device, weights_only=False)
        self.set_sam3_feature_field(sam, requires_grad=requires_grad)

    def init_sam3_affinity_field(self, dim, *, init_random=True, requires_grad=True):
        dim = int(dim or 0)
        n = int(self._xyz.shape[0]) if self._xyz.numel() else 0
        if n <= 0:
            return
        if dim <= 0:
            self._sam3_affinity_field = torch.empty((n, 0), device=self._xyz.device)
            return
        current = getattr(self, "_sam3_affinity_field", None)
        if (
            current is None
            or not torch.is_tensor(current)
            or current.numel() == 0
            or tuple(current.shape) != (n, dim)
        ):
            if init_random:
                feature = torch.randn((n, dim), device=self._xyz.device, dtype=torch.float32) * 0.01
                feature = torch.nn.functional.normalize(feature, dim=-1)
            else:
                feature = torch.zeros((n, dim), device=self._xyz.device, dtype=torch.float32)
            self._sam3_affinity_field = nn.Parameter(
                feature.contiguous().requires_grad_(bool(requires_grad))
            )
        self._assert_aux_row_alignment("_sam3_affinity_field")

    def set_sam3_affinity_field(self, feature, *, requires_grad=True):
        if isinstance(feature, dict):
            feature = feature.get("features", feature.get("sam3_affinity_field"))
        value = torch.as_tensor(feature, device=self._xyz.device, dtype=torch.float32)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2:
            raise ValueError(f"SAM3 affinity field must be [N,D], got {tuple(value.shape)}")
        if int(value.shape[0]) != int(self._xyz.shape[0]):
            raise ValueError(
                f"SAM3 affinity rows {int(value.shape[0])} do not match "
                f"Gaussian rows {int(self._xyz.shape[0])}"
            )
        if value.shape[1] > 0:
            self._sam3_affinity_field = nn.Parameter(
                value.float().contiguous().requires_grad_(bool(requires_grad))
            )
        else:
            self._sam3_affinity_field = torch.empty((value.shape[0], 0), device=self._xyz.device)
        self._assert_aux_row_alignment("_sam3_affinity_field")

    def save_sam3_affinity_field(self, path, *, training_config=None, frame_idx=None):
        sam = getattr(self, "_sam3_affinity_field", None)
        if sam is None or not torch.is_tensor(sam) or sam.numel() == 0:
            return
        if training_config is None:
            training_config = getattr(self, "_sam3_affinity_training_config", {})
        if frame_idx is None:
            frame_idx = getattr(self, "_sam3_affinity_frame_idx", None)
        payload = {
            "features": sam.detach().cpu(),
            "dim": int(sam.shape[1]),
            "normalization": "cosine/unit-vector-at-render",
            "training_config": dict(training_config or {}),
            "frame_idx": None if frame_idx is None else int(frame_idx),
        }
        torch.save(payload, path)

    def load_sam3_affinity_field(self, path, *, requires_grad=True):
        sam = torch.load(path, map_location=self._xyz.device, weights_only=False)
        self.set_sam3_affinity_field(sam, requires_grad=requires_grad)

    @property
    def get_semantic_probability(self):
        if self._semantic_logits is None or self._semantic_logits.numel() == 0:
            return None
        return torch.sigmoid(self._semantic_logits)

    @property
    def get_dynamic_foreground_mask(self):
        if (
            self._dynamic_foreground_mask is None
            or self._dynamic_foreground_mask.numel() != self._xyz.shape[0]
        ):
            raise RuntimeError("Semantic foreground mask has not been initialized")
        return self._dynamic_foreground_mask.bool()

    @property
    def get_motion_evidence(self):
        return self._motion_evidence

    @property
    def get_initial_dynamic_mask(self):
        return self._initial_dynamic_mask

    def build_dynamic_foreground_mask(self, args):
        """Build the strict semantic-foreground contract used by Phase 1."""
        n = int(self._xyz.shape[0])
        if bool(getattr(args, "dynamic_use_all_gaussians", False)):
            mask = torch.ones(n, device=self._xyz.device, dtype=torch.bool)
            self.dynamic_foreground_threshold = 0.0
            self._dynamic_foreground_mask = mask
            return self._dynamic_foreground_mask
        thr = float(getattr(args, "semantic_dynamic_threshold", 0.5))
        min_pts = int(getattr(args, "semantic_dynamic_min_points", 32))
        self.dynamic_foreground_threshold = thr
        sem_prob = self.get_semantic_probability
        if sem_prob is None or sem_prob.numel() != n:
            raise RuntimeError(
                "Dynamic joint fitting requires a trained semantic field; "
                "the all-Gaussian fallback was intentionally removed."
            )
        mask = sem_prob.detach().reshape(-1) > thr
        if int(mask.sum()) < min_pts:
            raise RuntimeError(
                f"Only {int(mask.sum())} semantic foreground Gaussians at threshold {thr}; "
                f"need at least {min_pts}. No top-k fallback is allowed."
            )
        self._dynamic_foreground_mask = mask.detach().bool()
        return self._dynamic_foreground_mask

    def init_semantic_and_feature_fields(
        self,
        semantic=True,
        dino_dim=0,
        init_dino_random=True,
        sam3_dim=0,
        init_sam3_random=True,
    ):
        """Initialize learnable semantic/DINO/SAM3 fields for the current Gaussians.

        This must be called before training_setup() if the fields should be part
        of the main Gaussian optimizer. It is intentionally safe to call after a
        PLY load; shape mismatches are rebuilt.
        """
        n = int(self._xyz.shape[0]) if self._xyz.numel() else 0
        if n <= 0:
            return
        device = self._xyz.device
        if semantic:
            if (
                self._semantic_logits is None
                or self._semantic_logits.numel() == 0
                or self._semantic_logits.shape[0] != n
            ):
                self._semantic_logits = nn.Parameter(
                    torch.zeros(
                        (n, 1), device=device, dtype=torch.float32
                    ).requires_grad_(True)
                )
        dino_dim = int(dino_dim or 0)
        if dino_dim <= 0:
            self._dino_feature_field = torch.empty((n, 0), device=device)
        else:
            if (
                self._dino_feature_field is None
                or self._dino_feature_field.numel() == 0
                or tuple(self._dino_feature_field.shape) != (n, dino_dim)
            ):
                if init_dino_random:
                    feat = (
                        torch.randn((n, dino_dim), device=device, dtype=torch.float32)
                        * 0.01
                    )
                    feat = torch.nn.functional.normalize(feat, dim=-1)
                else:
                    feat = torch.zeros(
                        (n, dino_dim), device=device, dtype=torch.float32
                    )
                self._dino_feature_field = nn.Parameter(feat.requires_grad_(True))
        self.init_sam3_feature_field(
            int(sam3_dim or 0),
            init_random=bool(init_sam3_random),
            requires_grad=True,
        )

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

    def get_covariance(self, scaling_modifier=1):
        rotation = (
            self._current_rotation
            if self._current_rotation is not None
            else self._rotation
        )
        return self.covariance_activation(self.get_scaling, scaling_modifier, rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(
        self, pcd: BasicPointCloud, cam_infos: int, spatial_lr_scale: float
    ):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._semantic_logits = nn.Parameter(
            torch.zeros((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True)
        )
        self._dino_feature_field = torch.empty(
            (self.get_xyz.shape[0], 0), device="cuda"
        )
        self._sam3_feature_field = torch.empty(
            (self.get_xyz.shape[0], 0), device="cuda"
        )
        self._sam3_affinity_field = torch.empty(
            (self.get_xyz.shape[0], 0), device="cuda"
        )
        part_indices = getattr(pcd, "part_indices", None)
        if part_indices is not None:
            part_np = np.asarray(part_indices, dtype=np.int64).reshape(-1)
            if part_np.shape[0] != self.get_xyz.shape[0]:
                raise ValueError(
                    f"Invalid BasicPointCloud part_indices length: "
                    f"{part_np.shape[0]} vs N={self.get_xyz.shape[0]}"
                )
            self._gt_part_labels = torch.tensor(
                part_np, dtype=torch.long, device="cuda"
            ).contiguous()
            unique, counts = np.unique(part_np, return_counts=True)
            count_text = ", ".join(
                f"{int(label)}:{int(count)}" for label, count in zip(unique, counts)
            )
            print(f"[GT parts] initialized Gaussian part labels: {count_text}")
        else:
            self._gt_part_labels = torch.empty(0, dtype=torch.long, device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {
            cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)
        }
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]
        if self._semantic_logits is not None and self._semantic_logits.numel() > 0:
            l.append(
                {
                    "params": [self._semantic_logits],
                    "lr": float(getattr(training_args, "semantic_feature_lr", 1e-3)),
                    "name": "semantic_logits",
                }
            )
        if (
            self._dino_feature_field is not None
            and self._dino_feature_field.numel() > 0
        ):
            l.append(
                {
                    "params": [self._dino_feature_field],
                    "lr": float(getattr(training_args, "dino_feature_lr", 1e-3)),
                    "name": "dino_feature",
                }
            )
        if (
            self._sam3_feature_field is not None
            and torch.is_tensor(self._sam3_feature_field)
            and self._sam3_feature_field.numel() > 0
        ):
            l.append(
                {
                    "params": [self._sam3_feature_field],
                    "lr": float(getattr(training_args, "sam3_feature_lr", 1e-3)),
                    "name": "sam3_feature",
                }
            )

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init,
            training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group["lr"] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Fixed Eulerian curl-strain dynamic pipeline
    # ------------------------------------------------------------------
    def capture_dynamic_state(self):
        if not self.dynamic_enabled:
            return None
        if self.motion_representation in RIGID_KILLING_REPRESENTATIONS:
            kpart_state = None
            if self.kpart_joint_state is not None:
                kpart_state = {
                    **self.kpart_joint_state,
                    "frame_times": self.kpart_joint_state["frame_times"].detach().cpu(),
                    "coordinates": self.kpart_joint_state["coordinates"].detach().cpu(),
                    "velocity": self.kpart_joint_state["velocity"].detach().cpu(),
                    "responsibilities": self.kpart_joint_state["responsibilities"].detach().cpu(),
                    "labels": self.kpart_joint_state["labels"].detach().cpu(),
                    "joints": [
                        {
                            **joint,
                            "axis": torch.as_tensor(joint["axis"]).detach().cpu(),
                            "pivot": None
                            if joint.get("pivot") is None
                            else torch.as_tensor(joint["pivot"]).detach().cpu(),
                            "coarse_q": torch.as_tensor(joint["coarse_q"]).detach().cpu(),
                        }
                        for joint in self.kpart_joint_state["joints"]
                    ],
                }
            return {
                "version": 25,
                "dynamic_phase": "phase1",
                "motion_time_mode": self.motion_time_mode,
                "motion_representation": self.motion_representation,
                "foreground_mask": self._dynamic_foreground_mask.detach().cpu(),
                "motion_evidence": self._motion_evidence.detach().cpu(),
                "static_evidence": self._static_evidence.detach().cpu(),
                "initial_dynamic_mask": self._initial_dynamic_mask.detach().cpu(),
                "foreground_source_indices": self._foreground_source_indices.detach().cpu(),
                "material_assignment": self.material_assignment.capture_state(),
                "rigid_reference_point": self.rigid_reference_point.detach().cpu(),
                "rigid_reference_velocity": self.rigid_reference_velocity.detach().cpu(),
                "rigid_angular_mode": self.rigid_angular_mode.detach().cpu(),
                "rigid_interval_speeds": self.rigid_interval_speeds.detach().cpu(),
                "rigid_frame_times": self.rigid_frame_times.detach().cpu(),
                "rigid_dynamic_probability": self.rigid_dynamic_probability.detach().cpu(),
                "rigid_effective_dynamic_mask": self.rigid_effective_dynamic_mask.detach().cpu(),
                "rigid_solver_diagnostics": self.rigid_solver_diagnostics,
                "sequence_assignment_state": self.sequence_assignment_state,
                "render_discovery_state": self.render_discovery_state,
                "dynamic_prior_metadata": dict(self.dynamic_prior_metadata),
                "kpart_joint_state": kpart_state,
            }
        if self.curl_grid is None:
            return None
        return {
            "version": 20,
            "dynamic_phase": "phase1",
            "motion_time_mode": self.motion_time_mode,
            "motion_representation": "eulerian_curl_strain",
            "eulerian_strain_config": dict(self._eulerian_strain_config),
            "foreground_mask": self._dynamic_foreground_mask.detach().cpu(),
            "motion_evidence": self._motion_evidence.detach().cpu(),
            "static_evidence": self._static_evidence.detach().cpu(),
            "initial_dynamic_mask": self._initial_dynamic_mask.detach().cpu(),
            "foreground_source_indices": self._foreground_source_indices.detach().cpu(),
            "curl_grid": self.curl_grid.capture_state(),
            "material_assignment": self.material_assignment.capture_state(),
            "motion_optimizer": None
            if self.motion_optimizer is None
            else self.motion_optimizer.state_dict(),
            "membership_optimizer": None
            if self.membership_optimizer is None
            else self.membership_optimizer.state_dict(),
            "render_discovery_state": self.render_discovery_state,
            "dynamic_prior_metadata": dict(self.dynamic_prior_metadata),
        }

    def load_dynamic_state(self, state, training_args=None):
        if state is None:
            return
        representation = state.get("motion_representation")
        version = int(state.get("version", 0))
        if representation in RIGID_KILLING_REPRESENTATIONS:
            if version not in (21, 22, 23, 24, 25):
                raise RuntimeError("Only rigid Killing checkpoint schemas v21-v25 are supported")
            device = self._xyz.device
            self.material_assignment = MaterialAssignmentField.from_state(
                state["material_assignment"], device=device
            )
            self._dynamic_foreground_mask = state["foreground_mask"].to(device).bool()
            self._foreground_source_indices = state["foreground_source_indices"].to(
                device
            ).long()
            self._motion_evidence = state["motion_evidence"].to(device).float()
            self._static_evidence = state["static_evidence"].to(device).float()
            self._initial_dynamic_mask = state["initial_dynamic_mask"].to(device).bool()
            self.rigid_reference_point = state["rigid_reference_point"].to(device)
            self.rigid_reference_velocity = state["rigid_reference_velocity"].to(device)
            self.rigid_angular_mode = state["rigid_angular_mode"].to(device)
            self.rigid_interval_speeds = state["rigid_interval_speeds"].to(device)
            self.rigid_frame_times = state["rigid_frame_times"].to(device)
            self.rigid_dynamic_probability = state["rigid_dynamic_probability"].to(
                device
            )
            self.rigid_effective_dynamic_mask = state[
                "rigid_effective_dynamic_mask"
            ].to(device).bool()
            self.rigid_solver_diagnostics = dict(
                state.get("rigid_solver_diagnostics", {})
            )
            self.sequence_assignment_state = state.get("sequence_assignment_state")
            self.render_discovery_state = state.get("render_discovery_state")
            self.dynamic_prior_metadata = dict(state.get("dynamic_prior_metadata", {}))
            saved_kpart = state.get("kpart_joint_state")
            if saved_kpart is None:
                self.kpart_joint_state = None
            else:
                self.kpart_joint_state = {
                    **saved_kpart,
                    "frame_times": saved_kpart["frame_times"].to(device),
                    "coordinates": saved_kpart["coordinates"].to(device),
                    "velocity": saved_kpart["velocity"].to(device),
                    "responsibilities": saved_kpart["responsibilities"].to(device),
                    "labels": saved_kpart["labels"].to(device).long(),
                    "joints": [
                        {
                            **joint,
                            "axis": torch.as_tensor(joint["axis"], device=device),
                            "pivot": None
                            if joint.get("pivot") is None
                            else torch.as_tensor(joint["pivot"], device=device),
                            "coarse_q": torch.as_tensor(joint["coarse_q"], device=device),
                        }
                        for joint in saved_kpart["joints"]
                    ],
                }
            self.motion_representation = representation
            self.motion_time_mode = str(state.get("motion_time_mode", "seconds"))
            self.dynamic_enabled = True
            self.motion_optimizer = None
            self.membership_optimizer = None
            return
        if version != 20:
            raise RuntimeError(
                "Only fixed Eulerian curl-strain checkpoint schema v20 is supported; "
                "restart older runs from their canonical PLY."
            )
        if representation != "eulerian_curl_strain":
            raise RuntimeError("Only eulerian_curl_strain checkpoints are supported")
        device = self._xyz.device
        self.curl_grid = FixedCurlGrid.from_state(state["curl_grid"], device=device)
        self.material_assignment = MaterialAssignmentField.from_state(
            state["material_assignment"], device=device
        )
        self._eulerian_strain_config = dict(state["eulerian_strain_config"])
        self._dynamic_foreground_mask = state["foreground_mask"].to(device).bool()
        self._foreground_source_indices = (
            state["foreground_source_indices"].to(device).long()
        )
        self._motion_evidence = state["motion_evidence"].to(device).float()
        self._static_evidence = state["static_evidence"].to(device).float()
        self._initial_dynamic_mask = state["initial_dynamic_mask"].to(device).bool()
        self.motion_representation = "eulerian_curl_strain"
        self.motion_time_mode = str(state.get("motion_time_mode", "seconds"))
        self.render_discovery_state = state.get("render_discovery_state")
        self.dynamic_prior_metadata = dict(state.get("dynamic_prior_metadata", {}))
        self.dynamic_enabled = True
        if training_args is not None:
            self.setup_dynamic_optimizers(training_args)
            if state.get("motion_optimizer") is not None:
                self.motion_optimizer.load_state_dict(state["motion_optimizer"])
            if state.get("membership_optimizer") is not None:
                self.membership_optimizer.load_state_dict(state["membership_optimizer"])

    @torch.no_grad()
    def compact_to_semantic_foreground(self, args):
        """Physically remove semantic-background Gaussians before Phase 1."""
        original_count = int(self._xyz.shape[0])
        mask = self.build_dynamic_foreground_mask(args).detach()
        source_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if source_indices.numel() == original_count:
            self._foreground_source_indices = source_indices.clone()
            self._dynamic_foreground_mask = torch.ones(
                original_count, device=self._xyz.device, dtype=torch.bool
            )
            return {
                "original_gaussians": original_count,
                "foreground_gaussians": original_count,
            }

        for name in (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
            "_semantic_logits",
            "_dino_feature_field",
            "_sam3_feature_field",
            "_sam3_affinity_field",
            "_gt_part_labels",
        ):
            value = getattr(self, name, None)
            if value is None or not torch.is_tensor(value) or value.numel() == 0:
                continue
            if value.shape[0] != original_count:
                continue
            compact = value.detach().index_select(0, source_indices).contiguous()
            if isinstance(value, nn.Parameter):
                compact = nn.Parameter(compact, requires_grad=value.requires_grad)
            setattr(self, name, compact)

        for name in ("max_radii2D", "xyz_gradient_accum", "denom"):
            value = getattr(self, name, None)
            if (
                torch.is_tensor(value)
                and value.numel() > 0
                and value.shape[0] == original_count
            ):
                setattr(
                    self,
                    name,
                    value.detach().index_select(0, source_indices).contiguous(),
                )

        self._foreground_source_indices = source_indices.clone()
        self._dynamic_foreground_mask = torch.ones(
            source_indices.numel(), device=self._xyz.device, dtype=torch.bool
        )
        self._motion_evidence = torch.empty(0, device=self._xyz.device)
        self._static_evidence = torch.empty(0, device=self._xyz.device)
        self._initial_dynamic_mask = torch.empty(
            0, device=self._xyz.device, dtype=torch.bool
        )
        self.clear_time()
        self.training_setup(args)
        return {
            "original_gaussians": original_count,
            "foreground_gaussians": int(source_indices.numel()),
            "removed_background_gaussians": original_count
            - int(source_indices.numel()),
        }

    def init_dynamic_motion(
        self, args, scene_extent=1.0, cameras=None, gradient_motion_state=None
    ):
        del scene_extent
        representation = str(args.phase1_motion_representation)
        if representation not in ("eulerian_curl_strain", *RIGID_KILLING_REPRESENTATIONS):
            raise ValueError(f"Unsupported Phase-1 representation: {representation}")
        if gradient_motion_state is None:
            raise RuntimeError(
                "Phase 1 requires full-scene gradient attribution"
            )

        time_values = (
            [float(getattr(camera, "timestamp", 0.0)) for camera in cameras]
            if cameras
            else [0.0, float(getattr(args, "time_scale", 1.0))]
        )
        compact_stats = self.compact_to_semantic_foreground(args)
        print(
            "[Dynamic foreground] physically retained "
            f"{compact_stats['foreground_gaussians']}/{compact_stats['original_gaussians']} Gaussians"
        )

        source_ids = self._foreground_source_indices.detach().cpu().long()
        raw_count = int(gradient_motion_state["canonical_xyz"].shape[0])
        if source_ids.numel() and int(source_ids.max()) >= raw_count:
            raise RuntimeError(
                "Foreground source index exceeds gradient-attribution state"
            )
        evidence_full = (
            gradient_motion_state.get(
                "motion_evidence_normalized", gradient_motion_state["motion_evidence"]
            )
            .detach()
            .reshape(-1)
            .cpu()
        )
        static_full = (
            gradient_motion_state.get(
                "static_consistency_score", 1.0 - evidence_full.clamp(0.0, 1.0)
            )
            .detach()
            .reshape(-1)
            .cpu()
        )
        unknown_full = gradient_motion_state.get("unknown_score", None)
        if unknown_full is None:
            prior_confidence = gradient_motion_state.get("prior_confidence", None)
            if prior_confidence is None:
                unknown_full = torch.zeros_like(evidence_full)
            else:
                unknown_full = (
                    1.0
                    - torch.as_tensor(prior_confidence)
                    .detach()
                    .reshape(-1)
                    .cpu()
                    .clamp(0.0, 1.0)
                )
        else:
            unknown_full = unknown_full.detach().reshape(-1).cpu()
        hard_full = (
            gradient_motion_state["initial_dynamic_mask"].detach().reshape(-1).cpu()
        )
        compact_evidence = evidence_full.index_select(0, source_ids).to(
            device=self._xyz.device, dtype=torch.float32
        )
        compact_static = static_full.index_select(0, source_ids).to(
            device=self._xyz.device, dtype=torch.float32
        )
        compact_unknown = unknown_full.index_select(0, source_ids).to(
            device=self._xyz.device, dtype=torch.float32
        )
        compact_hard = hard_full.index_select(0, source_ids).to(
            device=self._xyz.device, dtype=torch.bool
        )
        sequence_cores = gradient_motion_state.get("sequence_cores")
        if sequence_cores is not None:
            compact_sequence = {}
            for key, value in sequence_cores.items():
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == raw_count:
                    compact_sequence[key] = value.detach().cpu().index_select(0, source_ids)
                elif not torch.is_tensor(value):
                    compact_sequence[key] = value
            self.sequence_assignment_state = compact_sequence
        else:
            self.sequence_assignment_state = None
        self.dynamic_prior_metadata = dict(
            gradient_motion_state.get("dynamic_prior_metadata", {})
        )

        self.material_assignment = MaterialAssignmentField(
            compact_hard,
            num_moving_parts=int(args.num_moving_parts),
            initial_dynamic_score=compact_evidence,
            initial_static_score=compact_static,
            initial_unknown_score=compact_unknown,
            dynamic_probability=float(args.assignment_initial_dynamic_probability),
            static_dynamic_probability=float(
                args.assignment_initial_static_dynamic_probability
            ),
            evidence_temperature=float(args.assignment_evidence_temperature),
            evidence_probability_floor=float(args.eulerian_assignment_init_floor),
            evidence_probability_ceiling=float(args.eulerian_assignment_init_ceiling),
        )
        self._static_evidence = compact_static.detach()
        self.set_motion_evidence(compact_evidence, compact_hard)
        unique_times = sorted(set([0.0] + time_values))
        if representation in RIGID_KILLING_REPRESENTATIONS:
            initial_probability = (
                1.0 - self.material_assignment.soft_probabilities()[:, 0]
            ).detach()
            strongest = int(torch.argmax(compact_evidence))
            self.rigid_reference_point = self._xyz.detach()[strongest].clone()
            self.rigid_reference_velocity = torch.zeros(
                3, device=self._xyz.device, dtype=self._xyz.dtype
            )
            self.rigid_angular_mode = torch.zeros_like(self.rigid_reference_velocity)
            self.rigid_frame_times = torch.as_tensor(
                unique_times, device=self._xyz.device, dtype=self._xyz.dtype
            )
            self.rigid_interval_speeds = torch.zeros(
                max(0, len(unique_times) - 1),
                device=self._xyz.device,
                dtype=self._xyz.dtype,
            )
            self.rigid_dynamic_probability = initial_probability
            self.rigid_effective_dynamic_mask = compact_hard.detach().clone()
            self.rigid_solver_diagnostics = {"valid": False, "status": "not_fitted"}
            self.kpart_joint_state = None
            self.render_discovery_state = None
            self.curl_grid = None
            self.motion_representation = representation
            self.motion_time_mode = str(args.motion_time_mode)
            self.dynamic_enabled = True
            self.motion_optimizer = None
            self.membership_optimizer = None
            print(
                f"[{representation}] initialized without a curl field or motion optimizer; "
                f"points={self._xyz.shape[0]} intervals={self.rigid_interval_speeds.numel()}"
            )
            return

        self.curl_grid = FixedCurlGrid(
            self._xyz.detach(),
            unique_times,
            target_cells=int(args.curl_grid_target_cells),
            padding_fraction=float(args.curl_grid_padding),
            min_cells_per_axis=int(args.curl_grid_min_cells_per_axis),
            temporal_control_knots=int(args.curl_temporal_control_knots),
        )
        self._eulerian_strain_config = {
            "neighbors": int(args.curl_material_neighbors),
            "radius_cells": float(args.curl_material_neighbor_radius_cells),
            "min_neighbors": int(args.eulerian_velocity_gradient_min_neighbors),
            "ridge": float(args.eulerian_velocity_gradient_ridge),
            "same_part_floor": float(args.assignment_same_part_floor_end),
            "velocity_scale": float(args.eulerian_velocity_scale),
            "velocity_spatial_robust_delta": float(
                args.velocity_spatial_coherence_robust_delta
            ),
            "static_gate_threshold": float(args.eulerian_static_gate_threshold),
            "static_cell_purity": float(args.curl_static_cell_purity),
        }
        self.motion_representation = "eulerian_curl_strain"
        self.motion_time_mode = str(args.motion_time_mode)
        self.dynamic_enabled = True
        self.setup_dynamic_optimizers(args)
        print(
            "[Eulerian Curl + velocity-gradient rigidity] "
            f"cells={self.curl_grid.num_cells} shape={self.curl_grid.grid_shape} "
            f"A_controls={self.curl_grid.num_control_knots} "
            f"A_params={self.curl_grid.parameter_count} "
            f"assignment_params={self.material_assignment.logits.numel()}"
        )

    def set_motion_evidence(self, motion_evidence, initial_dynamic_mask):
        evidence = torch.as_tensor(
            motion_evidence, device=self._xyz.device, dtype=torch.float32
        ).reshape(-1)
        hard = torch.as_tensor(
            initial_dynamic_mask, device=self._xyz.device, dtype=torch.bool
        ).reshape(-1)
        if evidence.numel() != self._xyz.shape[0] or hard.numel() != self._xyz.shape[0]:
            raise ValueError("Motion evidence and labels must match the Gaussian count")
        if (hard & ~self.get_dynamic_foreground_mask).any():
            raise RuntimeError(
                "Gradient attribution marked semantic background as dynamic"
            )
        self._motion_evidence = evidence.detach()
        self._initial_dynamic_mask = hard.detach()

    def set_render_discovery_state(self, evidence, args):
        """Install render-denoise static/dynamic evidence on compact Gaussians."""
        if evidence is None:
            self.render_discovery_state = None
            return
        if not isinstance(evidence, dict):
            from dataclasses import asdict, is_dataclass

            if is_dataclass(evidence):
                evidence = asdict(evidence)
            else:
                raise TypeError("render discovery evidence must be a dict or dataclass")

        def detach_cpu(value):
            if torch.is_tensor(value):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {str(key): detach_cpu(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [detach_cpu(item) for item in value]
            return value

        def first(*names, default=None):
            for name in names:
                if name in evidence:
                    return evidence[name]
            if default is not None:
                return default
            raise KeyError(f"render discovery evidence missing {names[0]}")

        count = int(self._xyz.shape[0])
        dynamic = torch.as_tensor(
            first("dynamic_score", "per_gaussian_dynamic_score"),
            device=self._xyz.device,
            dtype=torch.float32,
        ).reshape(-1)
        static = torch.as_tensor(
            first("static_score", "per_gaussian_static_score", default=1.0 - dynamic.detach().cpu()),
            device=self._xyz.device,
            dtype=torch.float32,
        ).reshape(-1)
        unknown = torch.as_tensor(
            first("unknown_score", "per_gaussian_unknown_score", default=torch.zeros(count)),
            device=self._xyz.device,
            dtype=torch.float32,
        ).reshape(-1)
        confidence = torch.as_tensor(
            first("confidence", "per_gaussian_confidence", default=torch.ones(count)),
            device=self._xyz.device,
            dtype=torch.float32,
        ).reshape(-1)
        state = torch.as_tensor(
            first("state", "per_gaussian_state", default=torch.full((count,), 2)),
            device=self._xyz.device,
        ).long().reshape(-1)
        probability = torch.as_tensor(
            first(
                "final_dynamic_probability",
                "per_gaussian_final_dynamic_probability",
                default=dynamic.detach().cpu(),
            ),
            device=self._xyz.device,
            dtype=torch.float32,
        ).reshape(-1)
        tensors = {
            "dynamic_score": dynamic,
            "static_score": static,
            "unknown_score": unknown,
            "confidence": confidence,
            "state": state,
            "final_dynamic_probability": probability,
        }
        bad = [name for name, value in tensors.items() if value.numel() != count]
        if bad:
            raise ValueError(
                "render discovery evidence must match compact Gaussian count: "
                + ", ".join(bad)
            )

        self.render_discovery_state = detach_cpu(evidence)
        self.material_assignment = None
        from scene.render_guided_motion_probe import (
            DYNAMIC,
            material_assignment_from_render_evidence,
        )

        compact = {name: value.detach() for name, value in tensors.items()}
        self.material_assignment = material_assignment_from_render_evidence(
            self,
            compact,
            args,
        )
        hard_dynamic = state == DYNAMIC
        self._motion_evidence = probability.detach()
        self._static_evidence = static.detach()
        self._initial_dynamic_mask = hard_dynamic.detach()
        self.rigid_dynamic_probability = probability.detach()
        self.rigid_effective_dynamic_mask = hard_dynamic.detach()

    def freeze_canonical(self, freeze_geometry=True, freeze_appearance=True):
        if freeze_geometry:
            self._xyz.requires_grad_(False)
            self._scaling.requires_grad_(False)
            self._rotation.requires_grad_(False)
            self._opacity.requires_grad_(False)
        if freeze_appearance:
            self._features_dc.requires_grad_(False)
            self._features_rest.requires_grad_(False)
            if hasattr(self, "_exposure") and self._exposure is not None:
                self._exposure.requires_grad_(False)
            if self._semantic_logits is not None and self._semantic_logits.numel() > 0:
                self._semantic_logits.requires_grad_(False)
            if (
                self._dino_feature_field is not None
                and self._dino_feature_field.numel() > 0
            ):
                self._dino_feature_field.requires_grad_(False)
            if (
                self._sam3_feature_field is not None
                and torch.is_tensor(self._sam3_feature_field)
                and self._sam3_feature_field.numel() > 0
            ):
                self._sam3_feature_field.requires_grad_(False)

    def setup_dynamic_optimizers(self, args):
        if self.motion_representation in RIGID_KILLING_REPRESENTATIONS:
            self.motion_optimizer = None
            self.membership_optimizer = None
            return
        if self.curl_grid is None or self.material_assignment is None:
            raise RuntimeError("Initialize the Eulerian field before its optimizers")
        self.motion_optimizer = torch.optim.Adam(
            self.curl_grid.potential_parameters(), lr=float(args.motion_lr)
        )
        self.membership_optimizer = torch.optim.Adam(
            [self.material_assignment.logits], lr=float(args.assignment_lr)
        )

    def set_time(self, t, detach_canonical=True):
        if not self.dynamic_enabled:
            return
        xyz, rotation, velocity, cell_ids = self.dynamic_state_at(
            t, detach_canonical=detach_canonical
        )
        self._current_xyz = xyz
        self._current_rotation = rotation
        self._current_velocity = velocity
        self._current_cell_id = cell_ids
        self._current_time = float(torch.as_tensor(t).detach().cpu())

    def dynamic_state_at(self, t, detach_canonical=True):
        x0 = self._xyz.detach() if detach_canonical else self._xyz
        q0 = self._rotation.detach() if detach_canonical else self._rotation
        if not self.dynamic_enabled:
            return (
                x0,
                q0,
                torch.zeros_like(x0),
                torch.full((x0.shape[0],), -1, device=x0.device, dtype=torch.long),
            )
        if self.motion_representation in RIGID_KILLING_REPRESENTATIONS:
            return self._rigid_killing_state_at(x0, q0, t)
        return self._eulerian_strain_state_at(x0, q0, t)

    def set_rigid_killing_state(
        self,
        mode: RigidKillingMode,
        interval_speeds: torch.Tensor,
        frame_times: torch.Tensor,
        dynamic_probability: torch.Tensor,
        effective_dynamic_mask: torch.Tensor,
        diagnostics: dict,
    ):
        """Install a validated closed-form rigid solution for exact replay."""
        if not mode.valid or mode.mode_type not in ("rotational", "translational"):
            raise RuntimeError("A valid normalized rigid mode is required")
        device, dtype = self._xyz.device, self._xyz.dtype
        speeds = torch.as_tensor(interval_speeds, device=device, dtype=dtype).reshape(-1)
        times = torch.as_tensor(frame_times, device=device, dtype=dtype).reshape(-1)
        probability = torch.as_tensor(
            dynamic_probability, device=device, dtype=dtype
        ).reshape(-1)
        mask = torch.as_tensor(effective_dynamic_mask, device=device).bool().reshape(-1)
        if times.numel() != speeds.numel() + 1:
            raise ValueError("frame_times must contain one more value than interval speeds")
        if probability.numel() != self._xyz.shape[0] or mask.numel() != self._xyz.shape[0]:
            raise ValueError("Rigid assignment must match the Gaussian count")
        self.rigid_reference_point = mode.reference_point.to(device=device, dtype=dtype)
        self.rigid_reference_velocity = mode.reference_velocity.to(
            device=device, dtype=dtype
        )
        self.rigid_angular_mode = mode.angular_mode.to(device=device, dtype=dtype)
        self.rigid_interval_speeds = speeds
        self.rigid_frame_times = times
        self.rigid_dynamic_probability = probability.clamp(0.0, 1.0)
        self.rigid_effective_dynamic_mask = mask
        self.rigid_solver_diagnostics = dict(diagnostics)
        self.kpart_joint_state = None
        if self.motion_representation not in RIGID_KILLING_REPRESENTATIONS:
            self.motion_representation = "rigid_killing_1d"
        self.dynamic_enabled = True
        self.motion_optimizer = None
        self.membership_optimizer = None
        with torch.no_grad():
            probabilities = torch.zeros_like(self.material_assignment.logits)
            probabilities[:, 0] = 1.0 - self.rigid_dynamic_probability
            probabilities[:, 1:] = self.rigid_dynamic_probability[:, None] / float(
                self.material_assignment.num_moving_parts
            )
            self.material_assignment.logits.copy_(
                torch.log(probabilities.clamp_min(1.0e-8))
            )

    def set_kpart_joint_state(
        self,
        joints,
        coordinates,
        velocity,
        frame_times,
        responsibilities,
        labels,
        diagnostics=None,
    ):
        """Install frozen K-part assignments and absolute joint coordinates."""
        device, dtype = self._xyz.device, self._xyz.dtype
        q = torch.as_tensor(coordinates, device=device, dtype=dtype)
        qdot = torch.as_tensor(velocity, device=device, dtype=dtype)
        times = torch.as_tensor(frame_times, device=device, dtype=dtype).reshape(-1)
        probability = torch.as_tensor(responsibilities, device=device, dtype=dtype)
        part_labels = torch.as_tensor(labels, device=device).long().reshape(-1)
        if q.ndim != 2 or q.shape != qdot.shape or q.shape[0] != len(joints):
            raise ValueError("K-part coordinates/velocity must be [K,T]")
        if q.shape[1] != times.numel() or probability.shape != self.material_assignment.logits.shape:
            raise ValueError("K-part time or responsibility dimensions are inconsistent")
        if part_labels.numel() != self._xyz.shape[0]:
            raise ValueError("K-part labels must match the Gaussian count")
        if not torch.allclose(q[:, 0], torch.zeros_like(q[:, 0]), atol=1.0e-7, rtol=0.0):
            raise ValueError("Absolute joint coordinates must use q[:,0]=0")
        stored_joints = []
        for joint in joints:
            stored_joints.append({
                **joint,
                "axis": torch.as_tensor(joint["axis"], device=device, dtype=dtype),
                "pivot": None
                if joint.get("pivot") is None
                else torch.as_tensor(joint["pivot"], device=device, dtype=dtype),
                "coarse_q": torch.as_tensor(joint["coarse_q"], device=device, dtype=dtype),
            })
        self.kpart_joint_state = {
            "joints": stored_joints,
            "coordinates": q,
            "velocity": qdot,
            "frame_times": times,
            "responsibilities": probability,
            "labels": part_labels,
            "diagnostics": {} if diagnostics is None else dict(diagnostics),
        }
        self.rigid_frame_times = times
        self.rigid_dynamic_probability = probability[:, 1:].sum(dim=-1)
        self.rigid_effective_dynamic_mask = part_labels > 0
        self.dynamic_enabled = True
        self.motion_optimizer = None
        self.membership_optimizer = None
        with torch.no_grad():
            self.material_assignment.logits.copy_(
                torch.log(probability.clamp_min(1.0e-8))
            )

    @torch.no_grad()
    def _rigid_killing_state_at(self, x0, q0, t_end):
        if self.kpart_joint_state is not None:
            return self._kpart_joint_state_at(x0, q0, t_end)
        if (
            self.rigid_reference_point.numel() != 3
            or self.rigid_interval_speeds.numel() == 0
            or self.rigid_frame_times.numel() != self.rigid_interval_speeds.numel() + 1
        ):
            raise RuntimeError("Rigid Killing state is incomplete")
        target = float(torch.as_tensor(t_end).detach().cpu())
        times = self.rigid_frame_times.detach().cpu().double()
        speeds = self.rigid_interval_speeds.detach().cpu().double()
        total_increment = 0.0
        current_speed = 0.0
        for index in range(speeds.numel()):
            start, end = float(times[index]), float(times[index + 1])
            overlap = max(0.0, min(target, end) - start)
            if overlap > 0.0:
                total_increment += float(speeds[index]) * overlap
                current_speed = float(speeds[index])
            if target <= end:
                break
        mode = RigidKillingMode(
            reference_point=self.rigid_reference_point,
            reference_velocity=self.rigid_reference_velocity,
            angular_mode=self.rigid_angular_mode,
            mode_type=str(
                self.rigid_solver_diagnostics.get(
                    "mode_type",
                    "rotational"
                    if float(torch.linalg.norm(self.rigid_angular_mode)) > 0.0
                    else "translational",
                )
            ),
            normalization_scale=float(
                self.rigid_solver_diagnostics.get("normalization_scale", 1.0)
            ),
            valid=True,
            diagnostics=self.rigid_solver_diagnostics,
        )
        result = integrate_rigid_mode_exact(
            x0,
            mode,
            total_increment,
            orientations=q0,
            effective_dynamic_mask=self.rigid_effective_dynamic_mask,
        )
        xyz = result["positions"]
        rotation = result["orientations"]
        velocity = predict_rigid_cell_velocity(mode, xyz, current_speed)
        velocity = velocity * self.rigid_effective_dynamic_mask[:, None].to(velocity)
        cell_ids = torch.full(
            (xyz.shape[0],), -1, device=xyz.device, dtype=torch.long
        )
        return xyz, rotation, velocity, cell_ids

    @torch.no_grad()
    def _kpart_joint_state_at(self, x0, q0, t_end):
        """Replay exact joint transforms from absolute frame coordinates."""
        from utils.joint_model_fitting import joint_to_rigid_mode

        state = self.kpart_joint_state
        times = state["frame_times"]
        target = torch.as_tensor(t_end, device=times.device, dtype=times.dtype).reshape(())
        if target <= times[0]:
            lower = upper = 0
            alpha = times.new_zeros(())
        elif target >= times[-1]:
            lower = upper = int(times.numel()) - 1
            alpha = times.new_zeros(())
        else:
            upper = int(torch.searchsorted(times, target, right=False))
            lower = upper - 1
            alpha = (target - times[lower]) / (times[upper] - times[lower])
        coordinate = (1.0 - alpha) * state["coordinates"][:, lower] + alpha * state["coordinates"][:, upper]
        speed = (1.0 - alpha) * state["velocity"][:, lower] + alpha * state["velocity"][:, upper]
        xyz, rotation, point_velocity = x0.clone(), q0.clone(), torch.zeros_like(x0)
        labels = state["labels"]
        for part_index, joint in enumerate(state["joints"], start=1):
            mode = joint_to_rigid_mode(joint).to(device=x0.device, dtype=x0.dtype)
            result = integrate_rigid_mode_exact(
                x0, mode, coordinate[part_index - 1], orientations=q0
            )
            mask = labels == part_index
            xyz[mask] = result["positions"][mask]
            rotation[mask] = result["orientations"][mask]
            exact_velocity = predict_rigid_cell_velocity(
                mode, result["positions"], speed[part_index - 1]
            )
            point_velocity[mask] = exact_velocity[mask]
        return xyz, rotation, point_velocity, labels.clone()

    @torch.no_grad()
    def _eulerian_strain_state_at(self, x0, q0, t_end):
        if self.curl_grid is None or self.material_assignment is None:
            raise RuntimeError("Eulerian strain field/material assignment is missing")
        target = float(torch.as_tensor(t_end).detach().cpu())
        x = x0.detach().clone()
        q = q0.detach().clone()
        velocity = torch.zeros_like(x)
        cells = self.curl_grid.cell_ids(x)
        if target <= 0.0:
            return x, q, velocity, cells

        cfg = self._eulerian_strain_config
        centers = self.curl_grid.cell_centers(device=x.device, dtype=x.dtype)
        point_prob = self.material_assignment.soft_probabilities(0.25).to(x.dtype)
        previous = float(self.curl_grid.time_values[0])
        for endpoint in self.curl_grid.time_values.detach().cpu().tolist()[1:]:
            endpoint = float(endpoint)
            if previous >= target - 1.0e-12:
                break
            current_end = min(endpoint, target)
            dt = current_end - previous
            support = self.curl_grid.material_support(x)
            ids, occupied = support["cell_ids"], support["occupied"]
            max_radius = (
                float(cfg["radius_cells"]) * float(self.curl_grid.cell_size.max().cpu())
                if float(cfg["radius_cells"]) > 0
                else 0.0
            )
            graph = build_occupied_cell_knn_pair_graph(
                centers,
                occupied,
                neighbors_per_cell=int(cfg["neighbors"]),
                max_radius=max_radius,
                max_pairs=262144,
            )
            cell_prob, _ = aggregate_point_probabilities_to_cells(
                point_prob, ids, self.curl_grid.num_cells
            )
            point_static = (point_prob[:, 0] >= float(cfg["static_gate_threshold"])).to(
                x.dtype
            )[:, None]
            static_fraction, counts = aggregate_point_features_to_cells(
                point_static, ids, self.curl_grid.num_cells, normalize=False
            )
            hard_static = (
                occupied
                & (cell_prob[:, 0] >= float(cfg["static_gate_threshold"]))
                & (counts > 0)
                & (static_fraction[:, 0] >= float(cfg["static_cell_purity"]))
            )
            midpoint = 0.5 * (previous + current_end)
            raw_velocity = self.curl_grid.cell_velocities(midpoint)
            cell_velocity = (~hard_static)[:, None].to(
                raw_velocity.dtype
            ) * raw_velocity
            grad_state = eulerian_velocity_gradient_rigidity_loss(
                centers,
                cell_velocity,
                occupied,
                cell_prob,
                graph,
                same_part_floor=float(cfg["same_part_floor"]),
                velocity_scale=float(cfg["velocity_scale"]),
                min_neighbors=int(cfg["min_neighbors"]),
                relative_ridge=float(cfg["ridge"]),
                static_speed_weight=0.0,
                velocity_spatial_robust_delta=float(
                    cfg["velocity_spatial_robust_delta"]
                ),
            )
            velocity = self.curl_grid.gather_cell_velocity(
                ids, cell_velocity_table=cell_velocity
            )
            omega = self.curl_grid.gather_cell_velocity(
                ids, cell_velocity_table=grad_state["omega"]
            )
            q = integrate_world_angular_velocity_quaternion(q, omega, dt)
            x = x + dt * velocity
            cells = self.curl_grid.cell_ids(x)
            previous = current_end
            if current_end >= target - 1.0e-12:
                break
        return x, q, velocity, cells

    def set_free_working_state(
        self,
        xyz,
        t,
        velocity_override=None,
        cell_ids_override=None,
        rotation_override=None,
    ):
        if not self.dynamic_enabled:
            raise RuntimeError("Dynamic motion is not initialized")
        if velocity_override is None:
            velocity, cells = self.curl_grid.velocity(xyz, t)
        else:
            velocity = velocity_override
            cells = (
                torch.full((xyz.shape[0],), -1, device=xyz.device, dtype=torch.long)
                if cell_ids_override is None
                else cell_ids_override.detach().long()
            )
        self._current_xyz = xyz
        self._current_rotation = (
            self._rotation.detach() if rotation_override is None else rotation_override
        )
        self._current_velocity = velocity
        self._current_cell_id = cells
        self._current_time = float(torch.as_tensor(t).detach().cpu())
        if xyz.requires_grad:
            xyz.retain_grad()
        return xyz

    @contextmanager
    def render_probe_state(self, *, xyz, t=0.0, rotation=None):
        """Temporarily render overridden geometry without mutating canonical tensors."""
        previous = (
            self._current_xyz,
            self._current_rotation,
            self._current_velocity,
            self._current_cell_id,
            self._current_time,
        )
        self.set_free_working_state(
            xyz,
            float(t),
            velocity_override=torch.zeros_like(xyz),
            rotation_override=self._rotation.detach() if rotation is None else rotation,
        )
        try:
            yield
        finally:
            (
                self._current_xyz,
                self._current_rotation,
                self._current_velocity,
                self._current_cell_id,
                self._current_time,
            ) = previous

    def clear_time(self):
        self._current_xyz = None
        self._current_rotation = None
        self._current_velocity = None
        self._current_cell_id = torch.empty(
            0, dtype=torch.long, device=self._xyz.device
        )
        self._current_time = None

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))

        gt = getattr(self, "_gt_part_labels", None)
        if (
            gt is not None
            and torch.is_tensor(gt)
            and gt.numel() > 0
            and gt.shape[0] == self._xyz.shape[0]
        ):
            l.append("gt_part")
            l.append("gt_dynamic")

        # Learned auxiliary per-Gaussian fields.
        # These are embedded directly into point_cloud.ply, so each Gaussian
        # carries its semantic/DINO/SAM3 feature state.
        sem = getattr(self, "_semantic_logits", None)
        if sem is not None and sem.numel() > 0:
            l.append("semantic_logit")
            l.append("semantic_prob")

        din = getattr(self, "_dino_feature_field", None)
        if din is not None and din.numel() > 0 and din.ndim == 2:
            for i in range(din.shape[1]):
                l.append("dino_{}".format(i))
        sam3 = getattr(self, "_sam3_feature_field", None)
        if sam3 is not None and torch.is_tensor(sam3) and sam3.numel() > 0 and sam3.ndim == 2:
            for i in range(sam3.shape[1]):
                l.append("sam3_feature_{}".format(i))
        sam3_affinity = getattr(self, "_sam3_affinity_field", None)
        if (
            sam3_affinity is not None
            and torch.is_tensor(sam3_affinity)
            and sam3_affinity.numel() > 0
            and sam3_affinity.ndim == 2
        ):
            for i in range(sam3_affinity.shape[1]):
                l.append("sam3_affinity_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        aux_arrays = []

        gt = getattr(self, "_gt_part_labels", None)
        if (
            gt is not None
            and torch.is_tensor(gt)
            and gt.numel() > 0
        ):
            gt_cpu = gt.detach().long().cpu()
            if gt_cpu.ndim != 1 or gt_cpu.shape[0] != xyz.shape[0]:
                raise ValueError(
                    f"Invalid _gt_part_labels shape for save_ply: "
                    f"{tuple(gt_cpu.shape)} vs N={xyz.shape[0]}"
                )
            gt_part = gt_cpu.numpy().astype(np.float32)[:, None]
            gt_dynamic = (gt_cpu.numpy() > 0).astype(np.float32)[:, None]
            aux_arrays.extend([gt_part, gt_dynamic])

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

        # Embed dense SAM3 feature field directly in every Gaussian/PLY vertex.
        sam3 = getattr(self, "_sam3_feature_field", None)
        if sam3 is not None and torch.is_tensor(sam3) and sam3.numel() > 0:
            sam3_cpu = sam3.detach().float().cpu()
            if sam3_cpu.ndim != 2 or sam3_cpu.shape[0] != xyz.shape[0]:
                raise ValueError(
                    f"Invalid _sam3_feature_field shape for save_ply: "
                    f"{tuple(sam3_cpu.shape)} vs N={xyz.shape[0]}"
                )
            aux_arrays.append(sam3_cpu.numpy())

        # Embed SAM3 mask-affinity field too; the sidecar still stores metadata.
        sam3_affinity = getattr(self, "_sam3_affinity_field", None)
        if (
            sam3_affinity is not None
            and torch.is_tensor(sam3_affinity)
            and sam3_affinity.numel() > 0
        ):
            sam3_affinity_cpu = sam3_affinity.detach().float().cpu()
            if sam3_affinity_cpu.ndim != 2 or sam3_affinity_cpu.shape[0] != xyz.shape[0]:
                raise ValueError(
                    f"Invalid _sam3_affinity_field shape for save_ply: "
                    f"{tuple(sam3_affinity_cpu.shape)} vs N={xyz.shape[0]}"
                )
            aux_arrays.append(sam3_affinity_cpu.numpy())

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation, *aux_arrays),
            axis=1,
        )
        if attributes.shape[1] != len(dtype_full):
            raise RuntimeError(
                f"PLY attribute mismatch: attributes={attributes.shape[1]}, dtype={len(dtype_full)}. "
                f"Fields={[name for name, _ in dtype_full]}"
            )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

        # Also keep compact sidecars for fast loading/debugging and backward compatibility.
        if sem is not None and sem.numel() > 0:
            torch.save(
                sem.detach().cpu(),
                os.path.join(os.path.dirname(path), "semantic_logits.pt"),
            )
        if din is not None and din.numel() > 0:
            torch.save(
                din.detach().cpu(),
                os.path.join(os.path.dirname(path), "dino_feature_field.pt"),
            )
        if gt is not None and torch.is_tensor(gt) and gt.numel() > 0:
            gt_cpu = gt.detach().long().cpu()
            torch.save(
                {
                    "gt_part_labels": gt_cpu,
                    "gt_moving_mask": gt_cpu > 0,
                    "source": "gaussian_model_inherited_sparse_parts",
                },
                os.path.join(os.path.dirname(path), "gt_parts.pt"),
            )
        self.save_sam3_feature_field(
            os.path.join(os.path.dirname(path), "sam3_feature_field.pt")
        )
        self.save_sam3_affinity_field(
            os.path.join(os.path.dirname(path), "sam3_affinity_field.pt")
        )

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp=False, load_gt_parts=True):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(
                os.path.dirname(path), os.pardir, os.pardir, "exposure.json"
            )
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {
                    image_name: torch.FloatTensor(exposures[image_name])
                    .requires_grad_(False)
                    .cuda()
                    for image_name in exposures
                }
                print("Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )

        # Load auxiliary semantic/DINO/SAM3 fields. Prefer embedded PLY properties;
        # fall back to the old sidecar tensors for backward compatibility.
        prop_names = [p.name for p in plydata.elements[0].properties]
        sidecar_dir = os.path.dirname(path)
        sem_path = os.path.join(sidecar_dir, "semantic_logits.pt")
        dino_path = os.path.join(sidecar_dir, "dino_feature_field.pt")
        sam3_path = os.path.join(sidecar_dir, "sam3_feature_field.pt")
        sam3_affinity_path = os.path.join(sidecar_dir, "sam3_affinity_field.pt")
        gt_parts_path = os.path.join(sidecar_dir, "gt_parts.pt")

        if not load_gt_parts:
            self._gt_part_labels = torch.empty(0, dtype=torch.long, device='cuda')
        elif "gt_part" in prop_names:
            gt_np = np.asarray(plydata.elements[0]["gt_part"], dtype=np.int64)
            self._gt_part_labels = torch.tensor(
                gt_np, dtype=torch.long, device="cuda"
            ).contiguous()
        elif os.path.exists(gt_parts_path):
            gt_obj = torch.load(gt_parts_path, map_location="cpu", weights_only=False)
            if isinstance(gt_obj, dict):
                gt_obj = gt_obj.get("gt_part_labels", None)
            gt_tensor = torch.as_tensor(gt_obj, dtype=torch.long) if gt_obj is not None else None
            if gt_tensor is not None and gt_tensor.ndim == 1 and gt_tensor.shape[0] == xyz.shape[0]:
                self._gt_part_labels = gt_tensor.to(device="cuda").contiguous()
            else:
                self._gt_part_labels = torch.empty(0, dtype=torch.long, device="cuda")
        else:
            self._gt_part_labels = torch.empty(0, dtype=torch.long, device="cuda")

        if "semantic_logit" in prop_names:
            sem_np = np.asarray(
                plydata.elements[0]["semantic_logit"], dtype=np.float32
            )[:, None]
            self._semantic_logits = nn.Parameter(
                torch.tensor(sem_np, dtype=torch.float, device="cuda")
                .contiguous()
                .requires_grad_(True)
            )
        elif os.path.exists(sem_path):
            sem = torch.load(sem_path, map_location="cuda").float()
            if sem.ndim == 1:
                sem = sem[:, None]
            if sem.shape[0] == xyz.shape[0]:
                self._semantic_logits = nn.Parameter(
                    sem.contiguous().requires_grad_(True)
                )
            else:
                self._semantic_logits = nn.Parameter(
                    torch.zeros((xyz.shape[0], 1), device="cuda").requires_grad_(True)
                )
        else:
            self._semantic_logits = nn.Parameter(
                torch.zeros((xyz.shape[0], 1), device="cuda").requires_grad_(True)
            )

        dino_names = [name for name in prop_names if name.startswith("dino_")]
        dino_names = sorted(dino_names, key=lambda x: int(x.split("_")[-1]))
        if len(dino_names) > 0:
            din_np = np.stack(
                [
                    np.asarray(plydata.elements[0][name], dtype=np.float32)
                    for name in dino_names
                ],
                axis=1,
            )
            self._dino_feature_field = nn.Parameter(
                torch.tensor(din_np, dtype=torch.float, device="cuda")
                .contiguous()
                .requires_grad_(True)
            )
        elif os.path.exists(dino_path):
            din = torch.load(dino_path, map_location="cuda").float()
            if din.ndim == 2 and din.shape[0] == xyz.shape[0]:
                self._dino_feature_field = nn.Parameter(
                    din.contiguous().requires_grad_(True)
                )
            else:
                self._dino_feature_field = torch.empty((xyz.shape[0], 0), device="cuda")
        else:
            self._dino_feature_field = torch.empty((xyz.shape[0], 0), device="cuda")

        sam3_names = [name for name in prop_names if name.startswith("sam3_feature_")]
        sam3_names = sorted(sam3_names, key=lambda x: int(x.split("_")[-1]))
        if len(sam3_names) > 0:
            sam_np = np.stack(
                [
                    np.asarray(plydata.elements[0][name], dtype=np.float32)
                    for name in sam3_names
                ],
                axis=1,
            )
            self._sam3_feature_field = nn.Parameter(
                torch.nn.functional.normalize(
                    torch.tensor(sam_np, dtype=torch.float, device="cuda"),
                    dim=-1,
                )
                .contiguous()
                .requires_grad_(True)
            )
        elif os.path.exists(sam3_path):
            sam = torch.load(sam3_path, map_location="cuda", weights_only=False).float()
            if sam.ndim == 1:
                sam = sam[:, None]
            if sam.ndim == 2 and sam.shape[0] == xyz.shape[0]:
                self._sam3_feature_field = nn.Parameter(
                    torch.nn.functional.normalize(sam, dim=-1)
                    .contiguous()
                    .requires_grad_(True)
                )
            else:
                self._sam3_feature_field = torch.empty((xyz.shape[0], 0), device="cuda")
        else:
            self._sam3_feature_field = torch.empty((xyz.shape[0], 0), device="cuda")

        sam3_affinity_names = [
            name for name in prop_names if name.startswith("sam3_affinity_")
        ]
        sam3_affinity_names = sorted(
            sam3_affinity_names, key=lambda x: int(x.split("_")[-1])
        )
        if len(sam3_affinity_names) > 0:
            sam_aff_np = np.stack(
                [
                    np.asarray(plydata.elements[0][name], dtype=np.float32)
                    for name in sam3_affinity_names
                ],
                axis=1,
            )
            self._sam3_affinity_field = nn.Parameter(
                torch.tensor(sam_aff_np, dtype=torch.float, device="cuda")
                .contiguous()
                .requires_grad_(True)
            )
        elif os.path.exists(sam3_affinity_path):
            sam_aff = torch.load(
                sam3_affinity_path, map_location="cuda", weights_only=False
            )
            if isinstance(sam_aff, dict):
                sam_aff = sam_aff.get("features", sam_aff.get("sam3_affinity_field"))
            sam_aff = torch.as_tensor(sam_aff, device="cuda").float()
            if sam_aff.ndim == 1:
                sam_aff = sam_aff[:, None]
            if sam_aff.ndim == 2 and sam_aff.shape[0] == xyz.shape[0]:
                self._sam3_affinity_field = nn.Parameter(
                    sam_aff.contiguous().requires_grad_(True)
                )
            else:
                self._sam3_affinity_field = torch.empty((xyz.shape[0], 0), device="cuda")
        else:
            self._sam3_affinity_field = torch.empty((xyz.shape[0], 0), device="cuda")

        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
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
            self._semantic_logits = nn.Parameter(
                self._semantic_logits[valid_points_mask].requires_grad_(True)
            )
        if "dino_feature" in optimizable_tensors:
            self._dino_feature_field = optimizable_tensors["dino_feature"]
        elif (
            self._dino_feature_field is not None
            and self._dino_feature_field.numel() > 0
        ):
            self._dino_feature_field = nn.Parameter(
                self._dino_feature_field[valid_points_mask].requires_grad_(True)
            )
        if "sam3_feature" in optimizable_tensors:
            self._sam3_feature_field = optimizable_tensors["sam3_feature"]
        elif (
            self._sam3_feature_field is not None
            and torch.is_tensor(self._sam3_feature_field)
            and self._sam3_feature_field.numel() > 0
        ):
            self._sam3_feature_field = nn.Parameter(
                self._sam3_feature_field[valid_points_mask].requires_grad_(True)
            )
        if "sam3_affinity" in optimizable_tensors:
            self._sam3_affinity_field = optimizable_tensors["sam3_affinity"]
        elif (
            self._sam3_affinity_field is not None
            and torch.is_tensor(self._sam3_affinity_field)
            and self._sam3_affinity_field.numel() > 0
        ):
            self._sam3_affinity_field = nn.Parameter(
                self._sam3_affinity_field[valid_points_mask].requires_grad_(True)
            )
        gt = getattr(self, "_gt_part_labels", None)
        if gt is not None and torch.is_tensor(gt) and gt.numel() > 0:
            if gt.shape[0] != valid_points_mask.shape[0]:
                raise RuntimeError(
                    f"GT part label/prune mask mismatch: labels={gt.shape[0]} "
                    f"mask={valid_points_mask.shape[0]}"
                )
            self._gt_part_labels = gt[valid_points_mask].contiguous()

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
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_tmp_radii,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        sel = getattr(self, "_last_densify_selected_mask", None)
        rep = int(getattr(self, "_last_densify_repeat", 1))
        if (
            sel is not None
            and self._semantic_logits is not None
            and self._semantic_logits.numel() > 0
        ):
            d["semantic_logits"] = self._semantic_logits[sel].repeat(rep, 1)
        if (
            sel is not None
            and self._dino_feature_field is not None
            and self._dino_feature_field.numel() > 0
        ):
            d["dino_feature"] = self._dino_feature_field[sel].repeat(rep, 1)
        if (
            sel is not None
            and self._sam3_feature_field is not None
            and torch.is_tensor(self._sam3_feature_field)
            and self._sam3_feature_field.numel() > 0
        ):
            d["sam3_feature"] = self._sam3_feature_field[sel].repeat(rep, 1)
        old_sam3_affinity = getattr(self, "_sam3_affinity_field", None)
        if (
            sel is not None
            and old_sam3_affinity is not None
            and torch.is_tensor(old_sam3_affinity)
            and old_sam3_affinity.numel() > 0
        ):
            d["sam3_affinity"] = old_sam3_affinity[sel].repeat(rep, 1)
        old_gt_parts = getattr(self, "_gt_part_labels", None)
        new_gt_parts = None
        if (
            sel is not None
            and old_gt_parts is not None
            and torch.is_tensor(old_gt_parts)
            and old_gt_parts.numel() > 0
        ):
            if old_gt_parts.shape[0] != sel.shape[0]:
                raise RuntimeError(
                    f"GT part label/densify mask mismatch: labels={old_gt_parts.shape[0]} "
                    f"mask={sel.shape[0]}"
                )
            if old_gt_parts.device != sel.device:
                old_gt_parts = old_gt_parts.to(device=sel.device)
                self._gt_part_labels = old_gt_parts
            new_gt_parts = old_gt_parts[sel].repeat(rep)

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
        if "sam3_feature" in optimizable_tensors:
            self._sam3_feature_field = optimizable_tensors["sam3_feature"]
        if "sam3_affinity" in optimizable_tensors:
            self._sam3_affinity_field = optimizable_tensors["sam3_affinity"]
        elif "sam3_affinity" in d and torch.is_tensor(old_sam3_affinity):
            self._sam3_affinity_field = nn.Parameter(
                torch.cat((old_sam3_affinity, d["sam3_affinity"]), dim=0)
                .contiguous()
                .requires_grad_(bool(old_sam3_affinity.requires_grad))
            )
        if new_gt_parts is not None:
            self._gt_part_labels = torch.cat(
                (old_gt_parts, new_gt_parts), dim=0
            ).contiguous()

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self._last_densify_selected_mask = selected_pts_mask
        self._last_densify_repeat = N
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self._last_densify_selected_mask = selected_pts_mask
        self._last_densify_repeat = 1
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )

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
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
