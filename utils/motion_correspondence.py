"""Rasterized optical-flow supervision for the Eulerian velocity field.

Adjacent-frame flow is filtered by forward/backward consistency, semantic
foreground, and frozen DINO similarity. The retained CUDA paths use this dense
observation in two ways: accumulate multiview normal equations for per-cell 3-D
velocity targets, and rasterize differentiable projected cell velocity for a
dense robust endpoint loss. Source-time splat ownership is fixed in both paths.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.loss_utils import (
    _dino_feature_path_for_camera,
    _dynamic_root_for_train,
    _load_mask_for_view,
    _read_dino_feature_tensor,
)


def _solve_spd_3x3_batch_cpu(matrix: torch.Tensor, rhs: torch.Tensor):
    """Solve many tiny SPD 3x3 systems without CUDA linalg workspace explosions.

    The accumulated systems used by the observed-flow inversion are tiny in data
    volume (G x 3 x 3), but CUDA's batched eigensolver can request multi-GB
    workspaces for thousands of matrices.  Move only these compact normal
    equations to CPU, solve/condition them in float64, and return results on the
    original device/dtype.  This function is always called under no_grad.
    """
    if matrix.ndim != 3 or matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must be [N,3,3], got {tuple(matrix.shape)}")
    if rhs.shape != (matrix.shape[0], 3):
        raise ValueError(f"rhs must be [N,3], got {tuple(rhs.shape)}")
    device, dtype = matrix.device, matrix.dtype
    matrix_cpu = matrix.detach().to(device="cpu", dtype=torch.float64)
    rhs_cpu = rhs.detach().to(device="cpu", dtype=torch.float64)
    eig_cpu = torch.linalg.eigvalsh(matrix_cpu)
    solution_cpu = torch.linalg.solve(matrix_cpu, rhs_cpu[..., None])[..., 0]
    conditioning_cpu = (eig_cpu[:, 0] / eig_cpu[:, -1].clamp_min(1.0e-12)).clamp(
        0.0, 1.0
    )
    solution = solution_cpu.to(device=device, dtype=dtype)
    conditioning = conditioning_cpu.to(device=device, dtype=dtype)
    return solution, conditioning


def accumulate_flow_stats_cpu_reference(
    pixel_jacobian: torch.Tensor,
    observed_flow: torch.Tensor,
    contribution_weight: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
    *,
    delta_time: float,
):
    """Direct float64 reference for the normal equations accumulated by CUDA.

    Inputs are flattened valid pixel/Gaussian contributions. ``observed_flow``
    is a displacement in pixels, matching ``PairObservation``; the solved target
    is pixel velocity, so it is divided by the real interval duration here.
    """
    jacobian = torch.as_tensor(pixel_jacobian).detach().cpu().double()
    flow = torch.as_tensor(observed_flow).detach().cpu().double()
    weight = torch.as_tensor(contribution_weight).detach().cpu().double().reshape(-1)
    groups = torch.as_tensor(group_ids).detach().cpu().long().reshape(-1)
    if jacobian.ndim != 3 or jacobian.shape[1:] != (2, 3):
        raise ValueError("pixel_jacobian must be [M,2,3]")
    if flow.shape != (jacobian.shape[0], 2):
        raise ValueError("observed_flow must be [M,2]")
    if weight.shape[0] != jacobian.shape[0] or groups.shape[0] != jacobian.shape[0]:
        raise ValueError("weights and group_ids must have M entries")
    count = max(0, int(num_groups))
    valid = (
        (groups >= 0)
        & (groups < count)
        & (weight > 0)
        & torch.isfinite(jacobian).all(dim=(1, 2))
        & torch.isfinite(flow).all(dim=-1)
    )
    jacobian, flow, weight, groups = (
        jacobian[valid],
        flow[valid] / max(abs(float(delta_time)), 1.0e-8),
        weight[valid],
        groups[valid],
    )
    normal = torch.zeros((count, 3, 3), dtype=torch.float64)
    rhs = torch.zeros((count, 3), dtype=torch.float64)
    y2 = torch.zeros(count, dtype=torch.float64)
    weight_sum = torch.zeros(count, dtype=torch.float64)
    observation_count = torch.zeros(count, dtype=torch.long)
    if groups.numel():
        normal.index_add_(
            0, groups, weight[:, None, None] * jacobian.transpose(1, 2) @ jacobian
        )
        rhs.index_add_(
            0,
            groups,
            weight[:, None]
            * torch.einsum("mji,mj->mi", jacobian, flow),
        )
        y2.index_add_(0, groups, weight * flow.square().sum(dim=-1))
        weight_sum.index_add_(0, groups, weight)
        observation_count.index_add_(0, groups, torch.ones_like(groups))
    return {
        "normal_matrix": normal,
        "rhs": rhs,
        "target_y2": y2,
        "weight_sum": weight_sum,
        "observation_count": observation_count,
    }


def moving_model_static_improvement_mask(
    moving_rms_px: torch.Tensor,
    static_rms_px: torch.Tensor,
    *,
    absolute_margin_px: float,
    relative_margin: float,
    require_improvement: bool,
):
    """Require a nonzero-velocity model to beat the explicit zero model."""
    moving = torch.as_tensor(moving_rms_px)
    static = torch.as_tensor(static_rms_px).to(moving)
    improvement = static - moving
    relative = improvement / static.clamp_min(1.0e-8)
    finite = torch.isfinite(moving) & torch.isfinite(static)
    if require_improvement:
        accepted = (
            finite
            & (improvement > float(absolute_margin_px))
            & (relative > float(relative_margin))
        )
    else:
        accepted = finite
    return accepted, improvement, relative


def velocity_reprojection_rms(
    observation_state: dict,
    predicted_velocity: torch.Tensor,
    *,
    delta_time: float,
) -> torch.Tensor:
    """Evaluate any cell velocity with the same normal equations as target fitting."""
    velocity = torch.as_tensor(predicted_velocity)
    normal = observation_state["normal_matrix"].to(velocity)
    rhs = observation_state["rhs"].to(velocity)
    y2 = observation_state["target_y2"].to(velocity)
    weight = observation_state["weight_sum"].to(velocity)
    quadratic = torch.einsum("ni,nij,nj->n", velocity, normal, velocity)
    linear = 2.0 * (velocity * rhs).sum(dim=-1)
    residual = (y2 - linear + quadratic).clamp_min(0.0) / weight.clamp_min(1.0e-8)
    return torch.sqrt(residual) * abs(float(delta_time))


def _camera_key(camera) -> str:
    return str(
        getattr(camera, "dynamic_camera_id", getattr(camera, "image_name", "camera"))
    )


def _frame_id(camera) -> int:
    return int(getattr(camera, "frame_id", 0))


def pair_fixed_cameras(source_cameras: Sequence, target_cameras: Sequence):
    """Pair synchronized views by physical camera id."""
    source = {_camera_key(camera): camera for camera in source_cameras}
    target = {_camera_key(camera): camera for camera in target_cameras}
    common = sorted(set(source) & set(target))
    if not common:
        raise RuntimeError("No common physical cameras in adjacent frame groups")
    return [(source[key], target[key]) for key in common]


def _as_gray_u8(camera) -> np.ndarray:
    image = camera.original_image.detach().float().cpu().clamp(0.0, 1.0)
    rgb = image[:3].permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((rgb * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(gray)


def _resize_flow(flow: np.ndarray, height: int, width: int) -> np.ndarray:
    old_h, old_w = flow.shape[:2]
    if (old_h, old_w) == (height, width):
        return flow.astype(np.float32, copy=False)
    resized = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR).astype(
        np.float32
    )
    resized[..., 0] *= float(width) / float(max(old_w, 1))
    resized[..., 1] *= float(height) / float(max(old_h, 1))
    return resized


def _flow_tensor_from_file(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        value = np.load(path)
    else:
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            for key in ("flow", "flows", "forward_flow"):
                if key in value:
                    value = value[key]
                    break
        value = torch.as_tensor(value).float().cpu().numpy()
    if value.ndim == 3 and value.shape[0] == 2:
        value = np.moveaxis(value, 0, -1)
    if value.ndim != 3 or value.shape[-1] != 2:
        raise RuntimeError(
            f"Optical flow must be [H,W,2] or [2,H,W], got {value.shape}"
        )
    return value.astype(np.float32)


def _pixels_to_grid(pixels: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = 2.0 * pixels[..., 0] / float(max(width - 1, 1)) - 1.0
    y = 2.0 * pixels[..., 1] / float(max(height - 1, 1)) - 1.0
    return torch.stack((x, y), dim=-1)


def _sample_map(feature: torch.Tensor, pixels: torch.Tensor, height: int, width: int):
    """Bilinearly sample [C,h,w] using pixel coordinates in an HxW image."""
    grid = _pixels_to_grid(pixels, height, width).reshape(1, 1, -1, 2)
    sampled = F.grid_sample(
        feature.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, :, 0].transpose(0, 1)
    return sampled


def project_world_points(points: torch.Tensor, camera):
    """Project world points with the exact row-vector matrices used by 3DGS."""
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    homogeneous = torch.cat((points, ones), dim=-1)
    projection = camera.full_proj_transform.to(device=points.device, dtype=points.dtype)
    clip = homogeneous @ projection
    w = clip[:, 3]
    ndc = clip[:, :3] / w[:, None].clamp_min(1.0e-8)
    width, height = int(camera.image_width), int(camera.image_height)
    pixels = torch.stack(
        (
            0.5 * (ndc[:, 0] + 1.0) * float(max(width - 1, 1)),
            0.5 * (ndc[:, 1] + 1.0) * float(max(height - 1, 1)),
        ),
        dim=-1,
    )
    view = camera.world_view_transform.to(device=points.device, dtype=points.dtype)
    camera_points = homogeneous @ view
    depth = camera_points[:, 2]
    valid = (
        torch.isfinite(ndc).all(dim=-1)
        & torch.isfinite(depth)
        & (w > 1.0e-8)
        & (depth > 1.0e-8)
        & (ndc[:, 0].abs() <= 1.0)
        & (ndc[:, 1].abs() <= 1.0)
    )
    return pixels, depth, valid


@dataclass
class PairObservation:
    source_height: int
    source_width: int
    flow: torch.Tensor  # [2,h,w], original-pixel units
    confidence: torch.Tensor  # [1,h,w]
    calibration_confidence: torch.Tensor | None = None  # before semantic masking

    def to(self, device):
        return PairObservation(
            self.source_height,
            self.source_width,
            self.flow.to(device=device, dtype=torch.float32, non_blocking=True),
            self.confidence.to(device=device, dtype=torch.float32, non_blocking=True),
            None if self.calibration_confidence is None else self.calibration_confidence.to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
        )


class MotionCorrespondenceCache:
    """Lazy CPU cache of adjacent flow/DINO observations."""

    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
        self.enabled = bool(getattr(args, "use_motion_correspondence", True))
        self.root = _dynamic_root_for_train(dataset, args)
        self._cpu: Dict[Tuple[str, int, int], PairObservation] = {}
        self._device: OrderedDict = OrderedDict()
        self.measurement_source_id = "pair_observation_rasterized_flow_v1"

    def _key(self, source_camera, target_camera):
        return (
            _camera_key(source_camera),
            _frame_id(source_camera),
            _frame_id(target_camera),
        )

    def _precomputed_flow_path(self, source_camera, target_camera):
        flow_root = self.root / str(getattr(self.args, "dynamic_flow_dir", "flow"))
        camera = _camera_key(source_camera)
        source, target = _frame_id(source_camera), _frame_id(target_camera)
        for extension in (".npy", ".pt", ".pth"):
            names = [f"frame_{source:06d}_to_{target:06d}{extension}"]
            # A source-only filename is conventionally an adjacent-frame flow.
            # Never reuse it for the frame0->far-future multiview material probe.
            if int(target) == int(source) + 1:
                names.append(f"frame_{source:06d}{extension}")
            for name in names:
                path = flow_root / camera / name
                if path.exists():
                    return path
        return None

    def _compute_flow_pair(self, source_camera, target_camera, height, width):
        source_gray = cv2.resize(
            _as_gray_u8(source_camera), (width, height), interpolation=cv2.INTER_AREA
        )
        target_gray = cv2.resize(
            _as_gray_u8(target_camera), (width, height), interpolation=cv2.INTER_AREA
        )

        path = self._precomputed_flow_path(source_camera, target_camera)
        if path is not None:
            forward = _resize_flow(_flow_tensor_from_file(path), height, width)
        else:
            try:
                estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                forward = estimator.calc(source_gray, target_gray, None)
            except Exception:
                forward = cv2.calcOpticalFlowFarneback(
                    source_gray,
                    target_gray,
                    None,
                    0.5,
                    4,
                    21,
                    4,
                    7,
                    1.5,
                    0,
                )

        reverse_path = self._precomputed_flow_path(target_camera, source_camera)
        if reverse_path is not None:
            backward = _resize_flow(_flow_tensor_from_file(reverse_path), height, width)
        else:
            try:
                estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                backward = estimator.calc(target_gray, source_gray, None)
            except Exception:
                backward = cv2.calcOpticalFlowFarneback(
                    target_gray,
                    source_gray,
                    None,
                    0.5,
                    4,
                    21,
                    4,
                    7,
                    1.5,
                    0,
                )
        return forward.astype(np.float32), backward.astype(np.float32)

    def _dino_confidence(self, source_camera, target_camera, flow, height, width):
        if not bool(getattr(self.args, "use_dino_feature_field", False)):
            return np.ones((height, width), dtype=np.float32)
        source_path = _dino_feature_path_for_camera(
            source_camera, self.dataset, self.args
        )
        target_path = _dino_feature_path_for_camera(
            target_camera, self.dataset, self.args
        )
        if source_path is None or target_path is None:
            return np.ones((height, width), dtype=np.float32)
        try:
            source = F.normalize(_read_dino_feature_tensor(source_path).float(), dim=0)
            target = F.normalize(_read_dino_feature_tensor(target_path).float(), dim=0)
            dim = min(
                int(getattr(self.args, "dino_feature_field_dim", source.shape[0])),
                source.shape[0],
                target.shape[0],
            )
            source = F.interpolate(
                source[:dim][None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
            target = F.interpolate(
                target[:dim][None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
            yy, xx = torch.meshgrid(
                torch.arange(height, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32),
                indexing="ij",
            )
            target_pixels = torch.stack((xx, yy), dim=-1) + torch.from_numpy(flow)
            warped_target = _sample_map(
                target, target_pixels.reshape(-1, 2), height, width
            )
            warped_target = warped_target.reshape(height, width, dim).permute(2, 0, 1)
            cosine = (
                F.normalize(source, dim=0) * F.normalize(warped_target, dim=0)
            ).sum(dim=0)
            minimum = float(getattr(self.args, "correspondence_dino_min_cosine", 0.20))
            confidence = ((cosine - minimum) / max(1.0 - minimum, 1.0e-6)).clamp(
                0.0, 1.0
            )
            return confidence.cpu().numpy().astype(np.float32)
        except Exception as exc:
            print(f"[Motion correspondence] DINO validation unavailable: {exc}")
            return np.ones((height, width), dtype=np.float32)

    def _build(self, source_camera, target_camera):
        source_h, source_w = (
            int(source_camera.image_height),
            int(source_camera.image_width),
        )
        max_side = max(32, int(getattr(self.args, "correspondence_flow_max_side", 256)))
        scale = min(1.0, float(max_side) / float(max(source_h, source_w)))
        height = max(16, int(round(source_h * scale)))
        width = max(16, int(round(source_w * scale)))
        scale_x, scale_y = float(width) / source_w, float(height) / source_h

        forward, backward = self._compute_flow_pair(
            source_camera, target_camera, height, width
        )
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        target_x, target_y = xx + forward[..., 0], yy + forward[..., 1]
        backward_x = cv2.remap(
            backward[..., 0],
            target_x,
            target_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        backward_y = cv2.remap(
            backward[..., 1],
            target_x,
            target_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        cycle = np.sqrt(
            (forward[..., 0] + backward_x) ** 2 + (forward[..., 1] + backward_y) ** 2
        )
        cycle_threshold = max(
            1.0e-3,
            float(getattr(self.args, "correspondence_cycle_threshold", 1.5))
            * 0.5
            * (scale_x + scale_y),
        )
        cycle_confidence = np.exp(-np.square(cycle / cycle_threshold)).astype(
            np.float32
        )
        inside = (
            (target_x >= 0.0)
            & (target_x <= width - 1)
            & (target_y >= 0.0)
            & (target_y <= height - 1)
        )
        cycle_confidence *= inside.astype(np.float32)
        dino_confidence = self._dino_confidence(
            source_camera, target_camera, forward, height, width
        )
        semantic_mask = np.ones((height, width), dtype=np.float32)

        try:
            semantic = _load_mask_for_view(
                source_camera, self.dataset, self.args, (height, width), device="cpu"
            )
            if semantic is not None:
                semantic_mask = semantic.detach().cpu().numpy().astype(np.float32)
        except Exception:
            pass
        confidence = cycle_confidence * dino_confidence * semantic_mask
        calibration_confidence = cycle_confidence * dino_confidence

        flow_original = forward.copy()
        flow_original[..., 0] /= max(scale_x, 1.0e-8)
        flow_original[..., 1] /= max(scale_y, 1.0e-8)
        return PairObservation(
            source_h,
            source_w,
            torch.from_numpy(np.moveaxis(flow_original, -1, 0)).half(),
            torch.from_numpy(confidence[None]).half(),
            torch.from_numpy(calibration_confidence[None]).half(),
        )

    def get(self, source_camera, target_camera, device=None):
        if not self.enabled:
            return None
        key = self._key(source_camera, target_camera)
        if key not in self._cpu:
            self._cpu[key] = self._build(source_camera, target_camera)
        if device is None:
            return self._cpu[key]
        device_key = (key, str(device))
        if device_key not in self._device:
            self._device[device_key] = self._cpu[key].to(device)
            while len(self._device) > 16:
                self._device.popitem(last=False)
        else:
            self._device.move_to_end(device_key)
        return self._device[device_key]

    @torch.no_grad()
    def _observed_velocity_targets_rasterized(
        self,
        query_xyz: torch.Tensor,
        source_cameras: Sequence,
        target_cameras: Sequence,
        time_start: float,
        time_end: float,
        *,
        group_ids: torch.Tensor,
        num_groups: int,
        gaussians,
        pipe,
        query_rotations: torch.Tensor | None = None,
        observation_transform=None,
    ):
        """Accumulate observed-flow equations through actual 3DGS visibility."""
        from gaussian_renderer import render_observed_flow_stats

        device, dtype = query_xyz.device, query_xyz.dtype
        G = int(num_groups or 0)
        if G <= 0:
            return {
                "velocity": query_xyz.new_zeros((0, 3)),
                "confidence": query_xyz.new_zeros((0,)),
                "valid": torch.zeros((0,), device=device, dtype=torch.bool),
                "base_valid": torch.zeros((0,), device=device, dtype=torch.bool),
                "observation_count": torch.zeros((0,), device=device, dtype=torch.long),
                "distinct_camera_count": torch.zeros(
                    (0,), device=device, dtype=torch.long
                ),
                "camera_support_mask": torch.zeros(
                    (0, len(source_cameras)), device=device, dtype=torch.bool
                ),
                "conditioning": query_xyz.new_zeros((0,)),
                "reprojection_rms_px_per_s": query_xyz.new_zeros((0,)),
                "reprojection_rms_px": query_xyz.new_zeros((0,)),
                "static_rms_px_mean": query_xyz.new_zeros((0,)),
                "static_rms_px_max": query_xyz.new_zeros((0,)),
                "moving_rms_px": query_xyz.new_zeros((0,)),
                "static_improvement_px": query_xyz.new_zeros((0,)),
                "static_improvement_relative": query_xyz.new_zeros((0,)),
                "rejected_static_like": torch.zeros(
                    (0,), device=device, dtype=torch.bool
                ),
                "weight_sum": query_xyz.new_zeros((0,)),
                "normal_matrix": query_xyz.new_zeros((0, 3, 3)),
                "rhs": query_xyz.new_zeros((0, 3)),
                "target_y2": query_xyz.new_zeros((0,)),
                "flow_measurement_source_id": self.measurement_source_id,
            }

        normal = torch.zeros((G, 3, 3), device=device, dtype=dtype)
        rhs = torch.zeros((G, 3), device=device, dtype=dtype)
        y2 = torch.zeros(G, device=device, dtype=dtype)
        weight_sum = torch.zeros(G, device=device, dtype=dtype)
        observation_count = torch.zeros(G, device=device, dtype=torch.long)
        distinct_camera_count = torch.zeros(G, device=device, dtype=torch.long)
        camera_static_rms_terms = []
        camera_support_terms = []
        min_conf = float(getattr(self.args, "correspondence_min_confidence", 0.05))
        dt = max(abs(float(time_end) - float(time_start)), 1.0e-8)

        for source, target in pair_fixed_cameras(source_cameras, target_cameras):
            obs = self.get(source, target, device)
            if obs is not None and observation_transform is not None:
                obs = observation_transform(source, target, obs)
            if obs is None:
                camera_static_rms_terms.append(
                    (
                        torch.zeros(G, device=device, dtype=dtype),
                        torch.zeros(G, device=device, dtype=dtype),
                    )
                )
                camera_support_terms.append(
                    torch.zeros(G, device=device, dtype=torch.bool)
                )
                continue
            stats = render_observed_flow_stats(
                source,
                gaussians,
                pipe,
                query_xyz.detach(),
                obs.flow,
                obs.confidence,
                group_ids=group_ids,
                num_groups=G,
                query_rotations=query_rotations,
                delta_time=dt,
                min_observation_weight=min_conf,
            )

            n6 = stats["normal6"].to(device=device, dtype=dtype)
            if n6.shape != (G, 6):
                raise RuntimeError(
                    "rasterized flow stats returned an unexpected normal shape"
                )
            normal[:, 0, 0] += n6[:, 0]
            normal[:, 0, 1] += n6[:, 1]
            normal[:, 1, 0] += n6[:, 1]
            normal[:, 0, 2] += n6[:, 2]
            normal[:, 2, 0] += n6[:, 2]
            normal[:, 1, 1] += n6[:, 3]
            normal[:, 1, 2] += n6[:, 4]
            normal[:, 2, 1] += n6[:, 4]
            normal[:, 2, 2] += n6[:, 5]

            rhs += stats["rhs"].to(device=device, dtype=dtype)
            cam_y2 = stats["y2"].to(device=device, dtype=dtype)
            cam_weight = stats["weight_sum"].to(device=device, dtype=dtype)
            y2 += cam_y2
            weight_sum += cam_weight
            observation_count += stats["observation_count"].to(
                device=device, dtype=torch.long
            )
            has_camera = cam_weight > min_conf
            distinct_camera_count += has_camera.to(torch.long)
            camera_support_terms.append(has_camera)
            camera_static_rms_terms.append((cam_y2, cam_weight))

        trace = torch.diagonal(normal, dim1=-2, dim2=-1).sum(dim=-1)
        relative_ridge = float(
            getattr(self.args, "observed_flow_velocity_ridge", 1.0e-4)
        )
        ridge = relative_ridge * (trace / 3.0).clamp_min(1.0e-8) + 1.0e-8
        eye = torch.eye(3, device=device, dtype=dtype)[None]
        regularized = normal + ridge[:, None, None] * eye
        velocity, conditioning = _solve_spd_3x3_batch_cpu(regularized, rhs)

        minimum_obs = max(
            2, int(getattr(self.args, "observed_flow_velocity_min_observations", 6))
        )
        minimum_cameras = max(
            1, int(getattr(self.args, "observed_flow_velocity_min_cameras", 3))
        )
        min_condition = float(
            getattr(self.args, "observed_flow_velocity_min_condition", 1.0e-4)
        )

        quad = torch.einsum("ni,nij,nj->n", velocity, normal, velocity)
        linear = 2.0 * (velocity * rhs).sum(dim=-1)
        residual2 = (y2 - linear + quad).clamp_min(0.0) / weight_sum.clamp_min(1.0e-8)
        rms_px_per_s = torch.sqrt(residual2)
        rms_px = rms_px_per_s * dt

        static_rms_mean = torch.sqrt(y2 / weight_sum.clamp_min(1.0e-8)) * dt
        if camera_static_rms_terms:
            per_camera = []
            for cam_y2, cam_w in camera_static_rms_terms:
                cam_rms = torch.sqrt(cam_y2 / cam_w.clamp_min(1.0e-8)) * dt
                cam_rms = torch.where(
                    cam_w > min_conf, cam_rms, torch.full_like(cam_rms, float("nan"))
                )
                per_camera.append(cam_rms)
            cam_stack = torch.stack(per_camera, dim=0)
            static_rms_max = (
                torch.nan_to_num(cam_stack, nan=-1.0).amax(dim=0).clamp_min(0.0)
            )
        else:
            static_rms_max = static_rms_mean

        max_reprojection_px = max(
            1.0e-4,
            float(
                getattr(self.args, "observed_flow_velocity_max_reprojection_px", 0.75)
            ),
        )
        base_valid = (
            (observation_count >= minimum_obs)
            & (distinct_camera_count >= minimum_cameras)
            & (weight_sum > min_conf)
            & (conditioning >= min_condition)
            & torch.isfinite(velocity).all(dim=-1)
            & torch.isfinite(rms_px)
            & (rms_px <= max_reprojection_px)
        )
        static_improvement_valid, static_improvement, static_improvement_relative = (
            moving_model_static_improvement_mask(
                rms_px,
                static_rms_mean,
                absolute_margin_px=float(
                    getattr(
                        self.args, "observed_flow_static_improvement_abs", 0.05
                    )
                ),
                relative_margin=float(
                    getattr(
                        self.args, "observed_flow_static_improvement_rel", 0.15
                    )
                ),
                require_improvement=bool(
                    getattr(
                        self.args, "observed_flow_require_static_improvement", False
                    )
                ),
            )
        )
        valid = base_valid & static_improvement_valid
        velocity = torch.where(valid[:, None], velocity, torch.zeros_like(velocity))
        camera_support = (
            distinct_camera_count.to(dtype) / float(minimum_cameras)
        ).clamp(0.0, 1.0)
        sample_support = (observation_count.to(dtype) / float(minimum_obs)).clamp(
            0.0, 1.0
        )
        reprojection_quality = torch.exp(-torch.square(rms_px / max_reprojection_px))
        confidence = (
            torch.sqrt(conditioning.clamp_min(0.0))
            * camera_support
            * sample_support
            * reprojection_quality
        )
        confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
        return {
            "velocity": velocity,
            "confidence": confidence,
            "valid": valid,
            "base_valid": base_valid,
            "observation_count": observation_count,
            "distinct_camera_count": distinct_camera_count,
            "camera_support_mask": torch.stack(camera_support_terms, dim=-1),
            "conditioning": conditioning,
            "reprojection_rms_px_per_s": rms_px_per_s,
            "reprojection_rms_px": rms_px,
            "static_rms_px_mean": static_rms_mean,
            "static_rms_px_max": static_rms_max,
            "moving_rms_px": rms_px,
            "static_improvement_px": static_improvement,
            "static_improvement_relative": static_improvement_relative,
            "rejected_static_like": base_valid & ~static_improvement_valid,
            "weight_sum": weight_sum,
            "normal_matrix": normal,
            "rhs": rhs,
            "target_y2": y2,
            "flow_measurement_source_id": self.measurement_source_id,
        }

    @torch.no_grad()
    def observed_velocity_targets(
        self,
        query_xyz: torch.Tensor,
        source_cameras: Sequence,
        target_cameras: Sequence,
        time_start: float,
        time_end: float,
        *,
        group_ids: torch.Tensor | None = None,
        num_groups: int | None = None,
        gaussians=None,
        pipe=None,
        query_rotations: torch.Tensor | None = None,
        observation_transform=None,
    ):
        """Invert rasterized multiview 2-D flow into grouped or per-Gaussian velocities."""
        if gaussians is None or pipe is None:
            raise ValueError(
                "Observed velocity targets require Gaussian rasterization context"
            )
        device = query_xyz.device
        count = int(query_xyz.shape[0])
        if group_ids is None:
            group_ids = torch.arange(count, device=device, dtype=torch.long)
            num_groups = count
        else:
            group_ids = group_ids.to(device=device, dtype=torch.long).reshape(-1)
            if group_ids.shape[0] != count:
                raise ValueError("group_ids must have one id per query point")
            if num_groups is None:
                valid_ids = group_ids[group_ids >= 0]
                num_groups = int(valid_ids.max().item() + 1) if valid_ids.numel() else 0
        return self._observed_velocity_targets_rasterized(
            query_xyz,
            source_cameras,
            target_cameras,
            time_start,
            time_end,
            group_ids=group_ids,
            num_groups=int(num_groups or 0),
            gaussians=gaussians,
            pipe=pipe,
            query_rotations=query_rotations,
            observation_transform=observation_transform,
        )


def _dense_flow_pixels(observation: PairObservation, device, dtype):
    flow_height, flow_width = observation.flow.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(flow_height, device=device, dtype=dtype),
        torch.arange(flow_width, device=device, dtype=dtype),
        indexing="ij",
    )
    source_pixels = torch.stack(
        (
            xx
            * float(max(observation.source_width - 1, 1))
            / float(max(flow_width - 1, 1)),
            yy
            * float(max(observation.source_height - 1, 1))
            / float(max(flow_height - 1, 1)),
        ),
        dim=-1,
    ).reshape(-1, 2)
    observed_flow = observation.flow.permute(1, 2, 0).reshape(-1, 2)
    return source_pixels, observed_flow, flow_height, flow_width


def rasterized_projected_velocity_flow_loss(
    source_camera,
    target_camera,
    gaussians,
    pipe,
    query_xyz: torch.Tensor,
    velocity_table: torch.Tensor,
    observation: PairObservation | None,
    args,
    *,
    delta_time: float,
    query_rotation: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    validation_only: bool = False,
    flow_measurement_source_id: str = "pair_observation_rasterized_flow_v1",
):
    """Differentiable optical-flow loss on projected velocity/position only.

    The source-frame splat ownership is fixed: query positions, rotations,
    scales and opacities are detached before rasterization.  The rendered value
    per Gaussian is the exact camera-aware displacement

        project_target(x_i + v_i dt) - project_source(x_i)

    plus a one-channel validity/opacity marker.  The feature-rasterizer backward
    therefore routes gradients only to ``velocity_table`` or to the parameters
    that produced it; visibility/ownership remains a fixed sampling operator.
    """
    zero = velocity_table.sum() * 0.0
    if observation is None:
        return zero, {
            "projected_velocity_flow": 0.0,
            "projected_velocity_flow_total": 0.0,
            "projected_velocity_flow_valid": 0,
            "projected_velocity_flow_weight_sum": 0.0,
            "flow_measurement_source_id": flow_measurement_source_id,
        }

    weight_lambda = float(args.lambda_motion_projected_velocity_flow)
    if weight_lambda <= 0.0 and not validation_only:
        return zero, {
            "projected_velocity_flow": 0.0,
            "projected_velocity_flow_total": 0.0,
            "projected_velocity_flow_valid": 0,
            "projected_velocity_flow_weight_sum": 0.0,
            "flow_measurement_source_id": flow_measurement_source_id,
        }

    if velocity_table.ndim != 2 or velocity_table.shape[-1] != 3:
        raise ValueError(
            f"velocity_table must be [M,3], got {tuple(velocity_table.shape)}"
        )
    if query_xyz.ndim != 2 or query_xyz.shape[-1] != 3:
        raise ValueError(f"query_xyz must be [N,3], got {tuple(query_xyz.shape)}")

    device, dtype = velocity_table.device, velocity_table.dtype
    if observation.flow.device != device:
        observation = observation.to(device)

    query = query_xyz.detach().to(device=device, dtype=dtype).contiguous()
    if group_ids is None:
        if velocity_table.shape[0] != query.shape[0]:
            raise ValueError(
                "per-Gaussian projected velocity loss requires velocity_table "
                f"N={velocity_table.shape[0]} to match query N={query.shape[0]}"
            )
        point_velocity = velocity_table
        point_valid = torch.ones((query.shape[0],), device=device, dtype=torch.bool)
    else:
        ids = group_ids.detach().to(device=device, dtype=torch.long)
        if ids.shape[0] != query.shape[0]:
            raise ValueError("group_ids must match query_xyz")
        point_valid = (ids >= 0) & (ids < velocity_table.shape[0])
        if not bool(point_valid.any()):
            return zero, {
                "projected_velocity_flow": 0.0,
                "projected_velocity_flow_total": 0.0,
                "projected_velocity_flow_valid": 0,
                "projected_velocity_flow_weight_sum": 0.0,
                "flow_measurement_source_id": flow_measurement_source_id,
            }
        safe_ids = ids.clamp(0, max(int(velocity_table.shape[0]) - 1, 0))
        point_velocity = velocity_table[safe_ids] * point_valid[:, None].to(dtype)

    source_pixels_gaussian, _, source_project_valid = project_world_points(
        query, source_camera
    )
    target_points = query + point_velocity * float(delta_time)
    target_pixels_gaussian, _, target_project_valid = project_world_points(
        target_points, target_camera
    )
    point_valid = (
        point_valid & source_project_valid.detach() & target_project_valid.detach()
    )
    point_flow = target_pixels_gaussian - source_pixels_gaussian.detach()
    valid_marker = point_valid[:, None].to(dtype)
    features = torch.cat((point_flow * valid_marker, valid_marker), dim=-1)

    from gaussian_renderer import render_query_feature_field

    package = render_query_feature_field(
        viewpoint_camera=source_camera,
        pc=gaussians,
        pipe=pipe,
        query_positions=query,
        features=features,
        query_rotations=query_rotation,
    )
    rendered = package["render"]
    if rendered.ndim != 3 or rendered.shape[0] != 3:
        raise RuntimeError("projected velocity flow rasterizer must return [3,H,W]")

    source_pixels, observed_flow, flow_height, flow_width = _dense_flow_pixels(
        observation, device, dtype
    )
    sampled = _sample_map(
        rendered,
        source_pixels,
        observation.source_height,
        observation.source_width,
    )
    opacity = sampled[:, 2].clamp_min(0.0)
    predicted_flow = sampled[:, :2] / opacity[:, None].clamp_min(1.0e-6)
    observed_flow = observed_flow.to(device=device, dtype=dtype)
    observed_weight = observation.confidence.reshape(-1).to(device=device, dtype=dtype)

    valid = (
        torch.isfinite(predicted_flow).all(dim=-1)
        & torch.isfinite(observed_flow).all(dim=-1)
        & (opacity > float(getattr(args, "correspondence_alpha_threshold", 0.05)))
        & (
            observed_weight
            > float(getattr(args, "correspondence_min_confidence", 0.05))
        )
    )
    weight = observed_weight * valid.to(dtype)
    denominator = weight.sum().clamp_min(1.0e-8)
    if not bool(valid.any()) or float(denominator.detach()) <= 1.0e-7:
        return zero, {
            "projected_velocity_flow": 0.0,
            "projected_velocity_flow_total": 0.0,
            "projected_velocity_flow_valid": int(valid.sum().detach()),
            "projected_velocity_flow_weight_sum": float(denominator.detach()),
            "flow_measurement_source_id": flow_measurement_source_id,
        }

    edge = float(max(observation.source_height, observation.source_width))
    residual = (predicted_flow - observed_flow) / max(edge, 1.0)
    robust = torch.sqrt(residual.square().sum(dim=-1) + 1.0e-8) - 1.0e-4
    raw = (weight * robust).sum() / denominator
    total = raw if validation_only else weight_lambda * raw
    residual_px = torch.linalg.norm(
        predicted_flow[valid] - observed_flow[valid], dim=-1
    )
    return total, {
        "projected_velocity_flow": float(raw.detach()),
        "projected_velocity_flow_total": float(total.detach()),
        "projected_velocity_flow_valid": int(valid.sum().detach()),
        "projected_velocity_flow_weight_sum": float(denominator.detach()),
        "projected_velocity_flow_opacity_mean": float(opacity[valid].detach().mean()),
        "projected_velocity_flow_pred_median_px": float(
            torch.linalg.norm(predicted_flow[valid], dim=-1).detach().median()
        ),
        "projected_velocity_flow_obs_median_px": float(
            torch.linalg.norm(observed_flow[valid], dim=-1).detach().median()
        ),
        "projected_velocity_flow_error_median_px": float(residual_px.detach().median()),
        "projected_velocity_flow_dense_pixels": int(flow_height * flow_width),
        "flow_measurement_source_id": flow_measurement_source_id,
    }
