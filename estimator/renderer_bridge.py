"""Bridge from the estimator to the CUDA observation operators.

CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md section 19.1 and section 22.

This is the only place the estimator touches the rasterizer. It provides the
conceptual API of section 19.1 -- build_linearization_cache, render_from_cache,
apply_H, apply_HT, build_rhs, build_block_fisher -- on top of the real CUDA
extension in ``submodules/diff-gaussian-rasterization``.

Two invariants this module exists to enforce:

  * NOTHING here calls ``loss.backward()`` or touches autograd (section 36).
    Every derivative comes from the analytic CUDA kernels.
  * A :class:`FrameLinearization` is immutable for the lifetime of a PCG solve
    (section 20.3). Sorting, tile membership and every Jacobian coefficient are
    captured at construction, so ``A(v)`` is a genuinely fixed linear operator
    and conjugacy holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from diff_gaussian_rasterization import (
    iekf_apply_H,
    iekf_apply_HT,
    iekf_build_fisher,
    iekf_build_linearization_cache,
    iekf_fisher_to_dense,
    iekf_render,
)

from .state import GaussianMotionState, matrix_to_quaternion_wxyz

POSE_DIM = 6
STATE_DIM = 12


@dataclass(frozen=True)
class CameraSpec:
    """One synchronized camera. Intrinsics/extrinsics are frozen per frame."""

    viewmatrix: torch.Tensor      # [4,4] column-major, repo convention
    projmatrix: torch.Tensor      # [4,4] full projection
    tanfovx: float
    tanfovy: float
    image_height: int
    image_width: int


@dataclass
class CameraLinearization:
    """The frozen cache plus the primal render for ONE camera."""

    cache: object
    camera: CameraSpec
    rgb: torch.Tensor             # [3,H,W]
    depth: torch.Tensor           # [H,W] normalized D = Z/(A+eps)
    alpha: torch.Tensor           # [H,W] accumulated A
    transmittance: torch.Tensor   # [H,W] final T
    # [H,W] per-pixel count of contributors the forward loop composited before
    # transmittance saturated. The VJP and Fisher passes stop at the same index
    # so H^T is the true transpose of H (section 35 T8's adjoint identity).
    n_contrib: torch.Tensor


class FrameLinearization:
    """All cameras of one video frame, linearized at the same state.

    Section 25: information from synchronized cameras ADDS, so the operators
    accumulate over cameras in state space and never concatenate a cross-camera
    pixel matrix.
    """

    def __init__(self, views: Sequence[CameraLinearization], num_gaussians: int):
        self.views = list(views)
        self.num_gaussians = int(num_gaussians)

    # -- section 19.1: RGBD render_from_cache -------------------------------
    def predicted(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(v.rgb, v.depth) for v in self.views]

    # -- section 19.1: apply_H ----------------------------------------------
    def apply_H(self, pose_delta: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """H v, one RGB-D tangent per camera."""
        out = []
        for v in self.views:
            drgb, ddepth = iekf_apply_H(v.cache, pose_delta)
            out.append((drgb, ddepth))
        return out

    # -- section 19.1: apply_HT ---------------------------------------------
    def apply_HT(self, pixel_adjoints: Sequence[tuple[torch.Tensor, torch.Tensor]]
                 ) -> torch.Tensor:
        """H^T z accumulated over cameras -> [P,6]."""
        total = None
        for v, (zc, zd) in zip(self.views, pixel_adjoints):
            part = iekf_apply_HT(v.cache, zc, zd, v.transmittance, v.alpha,
                                 v.depth, v.n_contrib)
            total = part if total is None else total + part
        if total is None:
            dev = self.views[0].rgb.device if self.views else "cpu"
            total = torch.zeros((self.num_gaussians, POSE_DIM), device=dev)
        return total

    # -- section 19.1: build_block_fisher -----------------------------------
    def build_block_fisher(self, precisions: Sequence[tuple[torch.Tensor, torch.Tensor]]
                           ) -> torch.Tensor:
        """Per-Gaussian 6x6 measurement Fisher, summed over cameras -> [P,6,6]."""
        total = None
        for v, (pc, pd) in zip(self.views, precisions):
            packed = iekf_build_fisher(v.cache, pc, pd, v.transmittance,
                                       v.alpha, v.depth, v.n_contrib)
            dense = iekf_fisher_to_dense(packed)
            total = dense if total is None else total + dense
        if total is None:
            dev = self.views[0].rgb.device if self.views else "cpu"
            total = torch.zeros((self.num_gaussians, POSE_DIM, POSE_DIM), device=dev)
        return total

    # -- accumulated alpha A = 1 - T_end: H_A, H_A^T, alpha Fisher -----------
    # The masked-RGB observation (estimator/masked_observation.py) needs the
    # derivative of the renderer's own accumulated alpha. It comes from the
    # visible-alpha kernels (_C_vis) in full-ray mode (g_i = 1 on every pixel),
    # which walk the same frozen contributor lists, alpha clamp and T_STOP
    # termination as the RGB-D operators above; the primal A is v.alpha and the
    # walk stops at v.n_contrib, so no second render or cache is built.
    def _full_ray(self, v):
        key = (v.camera.image_height, v.camera.image_width, v.alpha.device)
        cache = getattr(self, "_full_ray_mode", None)
        if cache is None or cache[0] != key:
            mode = torch.ones((key[0], key[1]), dtype=torch.int32, device=key[2])
            zobs = torch.zeros((key[0], key[1]), dtype=torch.float32, device=key[2])
            self._full_ray_mode = (key, mode, zobs)
        return self._full_ray_mode[1], self._full_ray_mode[2]

    def apply_H_alpha(self, pose_delta: torch.Tensor) -> list[torch.Tensor]:
        """dA [H,W] per camera for a [P,6] pose tangent."""
        from diff_gaussian_rasterization import iekf_vis_apply_H
        out = []
        for v in self.views:
            mode, zobs = self._full_ray(v)
            out.append(iekf_vis_apply_H(v.cache, mode, zobs, 0.0, pose_delta.float().contiguous()))
        return out

    def apply_HT_alpha(self, alpha_adjoints: Sequence[torch.Tensor]) -> torch.Tensor:
        """H_A^T z_A accumulated over cameras -> [P,6]."""
        from diff_gaussian_rasterization import iekf_vis_apply_HT
        if len(alpha_adjoints) != len(self.views):
            raise AssertionError(f"{len(alpha_adjoints)} alpha adjoints for {len(self.views)} views")
        total = None
        for v, za in zip(self.views, alpha_adjoints):
            mode, zobs = self._full_ray(v)
            part = iekf_vis_apply_HT(v.cache, mode, zobs, 0.0, za.float().contiguous(),
                                     v.transmittance, v.n_contrib)
            total = part if total is None else total + part
        if total is None:
            dev = self.views[0].rgb.device if self.views else "cpu"
            total = torch.zeros((self.num_gaussians, POSE_DIM), device=dev)
        return total

    def build_alpha_fisher(self, weights: Sequence[torch.Tensor]) -> torch.Tensor:
        """Per-Gaussian 6x6 sum_p w_p H_A,p,i^T H_A,p,i over cameras -> [P,6,6]."""
        from diff_gaussian_rasterization import iekf_vis_build_fisher
        if len(weights) != len(self.views):
            raise AssertionError(f"{len(weights)} alpha weights for {len(self.views)} views")
        total = None
        for v, w in zip(self.views, weights):
            mode, zobs = self._full_ray(v)
            dense = iekf_fisher_to_dense(iekf_vis_build_fisher(
                v.cache, mode, zobs, 0.0, w.float().contiguous(), v.transmittance, v.n_contrib))
            total = dense if total is None else total + dense
        if total is None:
            dev = self.views[0].rgb.device if self.views else "cpu"
            total = torch.zeros((self.num_gaussians, POSE_DIM, POSE_DIM), device=dev)
        return total


def build_linearization(
    state: GaussianMotionState,
    cameras: Sequence[CameraSpec],
    *,
    background: torch.Tensor | None = None,
    eps_depth: float = 1e-6,
    antialiasing: bool = False,
    debug: bool = False,
) -> FrameLinearization:
    """Section 19.1 build_linearization_cache + render_from_cache, all cameras.

    Appearance is frozen (section 3): colors, opacity and scales come straight
    from the state and are never updated by the estimator. The rotation is
    converted to the renderer's quaternion here, which is the only place the
    SO(3) state meets the rasterizer's parameterization.
    """
    quats = matrix_to_quaternion_wxyz(state.R).contiguous().float()
    means = state.mu.contiguous().float()
    scales = state.scales.contiguous().float()
    opacity = state.opacity.contiguous().float()
    colors = state.colors.contiguous().float()
    if background is None:
        background = torch.zeros(3, device=means.device, dtype=means.dtype)

    views: list[CameraLinearization] = []
    for cam in cameras:
        cache = iekf_build_linearization_cache(
            means3D=means, scales=scales, rotations=quats, opacity=opacity,
            colors=colors, background=background.float(),
            viewmatrix=cam.viewmatrix.contiguous().float(),
            projmatrix=cam.projmatrix.contiguous().float(),
            tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
            image_height=cam.image_height, image_width=cam.image_width,
            antialiasing=antialiasing, eps_depth=eps_depth, debug=debug)
        rgb, depth, alpha, trans, n_contrib = iekf_render(cache)
        views.append(CameraLinearization(cache=cache, camera=cam, rgb=rgb,
                                         depth=depth, alpha=alpha,
                                         transmittance=trans,
                                         n_contrib=n_contrib))
    return FrameLinearization(views, int(state.num_gaussians))


def embed_pose_to_full_state(pose: torch.Tensor) -> torch.Tensor:
    """[P,6] -> [P,12]. Section 11.3: a single RGB-D frame carries no direct
    velocity Jacobian, so the velocity block is exactly zero. Velocity is
    corrected through the temporal prior's cross-information, not by inventing
    a measurement for it."""
    out = pose.new_zeros((pose.shape[0], STATE_DIM))
    out[:, :POSE_DIM] = pose
    return out


def extract_pose(full: torch.Tensor) -> torch.Tensor:
    """[P,12] -> [P,6]."""
    return full[:, :POSE_DIM].contiguous()
