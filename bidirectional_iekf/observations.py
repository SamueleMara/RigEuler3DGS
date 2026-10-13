"""Adapter from the free 12-D state to the EXISTING CUDA RGB-D operator.

P3A / T5 of CODEX_COMPLETE_UNIFIED_PHASE1_IEKS_REMAINING_STEPS.md.

No new rasterizer, no new photometric model, no new precision. Everything
here routes to `estimator/renderer_bridge.py` and `estimator/measurement.py`
as they are. What the adapter adds is the 12-D <-> 6-D plumbing the unified
free state needs, and the factor-inventory discipline the full-window
posterior needs.

AUDITED CONTRACTS this is built on, verified in the current tree:

  FrameLinearization.apply_H(pose_delta [P,6]) -> [(drgb [3,H,W],
                                                    ddepth [H,W]) per camera]
  FrameLinearization.apply_HT(pixel_adjoints) -> [P,6]
  CameraLinearization.depth is NORMALISED D = Z/(A+eps), which is the PDF's
    equation (35) convention -- NOT the training renderer's `invdepth`. The
    two must not be conflated: the IEKF bridge already composites the PDF's
    quantity, so no conversion is applied here and none is needed.
  n_contrib exists so apply_HT stops at the same contributor index the
    forward pass did, which is what makes H^T the true transpose of H.

The velocity columns of the image Jacobian are EXACTLY zero. A single frame
carries no direct velocity measurement; velocity is corrected through the
temporal process cross-information. `embed_pose_to_full_state` already
encodes that, and T5 tests it rather than trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from estimator.measurement import build_precision, measurement_energy, rgbd_residual
from estimator.renderer_bridge import (
    POSE_DIM,
    build_linearization,
    embed_pose_to_full_state,
    extract_pose,
)
from estimator.state import GaussianMotionState

STATE_DIM = 12


@dataclass(frozen=True)
class FactorKey:
    """Identity of one image factor. The inventory is keyed by this alone.

    A nonlinear relinearisation recomputes the residual and Jacobian of an
    EXISTING key; it never adds a key. That is the mechanism by which an
    image cannot be counted twice, and `FactorInventory` enforces it.
    """

    original_frame: int
    camera: int


class FactorInventory:
    """One image factor per (original frame, camera). Duplicates are refused."""

    def __init__(self) -> None:
        self._keys: dict[FactorKey, int] = {}

    def register(self, key: FactorKey) -> None:
        if key in self._keys:
            raise ValueError(
                f"{key} is already in the factor inventory. A relinearisation "
                "updates an existing factor's residual and Jacobian; it does "
                "not add a second copy of the same image. Counting one RGB-D "
                "frame twice doubles its information and biases the posterior.")
        self._keys[key] = len(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def frames(self) -> set:
        return {k.original_frame for k in self._keys}

    def as_manifest(self) -> dict:
        return {
            "num_factors": len(self._keys),
            "keys": [{"original_frame": k.original_frame, "camera": k.camera}
                     for k in sorted(self._keys, key=lambda k: (k.original_frame, k.camera))],
            "unique_by": "(original_frame, camera)",
            "note": ("Relinearisation updates these factors in place. The "
                     "count must not change between outer iterations."),
        }


@dataclass
class FrameCache:
    """One frame's linearisation plus everything its factors need."""

    original_frame: int
    linearization: object
    residuals: list                       # per camera (r_rgb, r_depth)
    precisions: list                      # per camera (p_rgb, p_depth)
    num_gaussians: int
    cameras: list = field(default_factory=list)


class FreeRGBDObservation:
    """The free state's view of the existing RGB-D measurement.

    `observations[t]` is a list over cameras of (rgb [3,H,W], depth [H,W],
    valid [H,W]) exactly as the dataset loader produces them.
    """

    def __init__(self, cameras, observations, *, frame_ids=None,
                 sigma_rgb: float = 1.0, depth_sigma_model="constant",
                 robust: bool = False, nu: float = 5.0,
                 use_rgb: bool = True, use_depth: bool = True,
                 eps_depth: float = 1e-6):
        self.cameras = list(cameras)
        self.observations = list(observations)
        self.frame_ids = (list(frame_ids) if frame_ids is not None
                          else list(range(len(self.observations))))
        if len(self.frame_ids) != len(self.observations):
            raise ValueError(
                f"{len(self.observations)} frames of observations but "
                f"{len(self.frame_ids)} frame ids")
        self.sigma_rgb = sigma_rgb
        self.depth_sigma_model = depth_sigma_model
        self.robust = bool(robust)
        self.nu = float(nu)
        self.use_rgb = bool(use_rgb)
        self.use_depth = bool(use_depth)
        self.eps_depth = float(eps_depth)
        self.inventory = FactorInventory()
        self._inventory_built = False

    # -- linearisation -----------------------------------------------------
    def linearize_frame(self, state: GaussianMotionState, frame_index: int) -> FrameCache:
        """Render through the EXISTING bridge and build residuals/precisions.

        Called once per frame per outer iteration. The factor keys are
        registered the first time only; afterwards this updates values.
        """
        original = self.frame_ids[frame_index]
        lin = build_linearization(state, self.cameras, eps_depth=self.eps_depth)
        predicted = lin.predicted()
        per_cam = self.observations[frame_index]

        residuals, precisions, used = [], [], []
        for camera_index, ((rgb_hat, depth_hat), observed) in enumerate(
                zip(predicted, per_cam)):
            if observed is None:
                residuals.append(None)
                precisions.append(None)
                continue
            rgb_gt, depth_gt, valid = observed
            r_rgb = rgbd_residual(rgb_gt.to(rgb_hat), rgb_hat)
            r_depth = rgbd_residual(depth_gt.to(depth_hat), depth_hat,
                                    valid_mask=valid)
            # Exactly the call the existing IEKF makes, including the empty
            # sigma_rgb for the depth-only channel: the layout decides which
            # sensor model applies, and a stray RGB sigma against a
            # depth-only layout is refused by build_precision.
            if self.use_rgb:
                p_rgb = build_precision(
                    r_rgb, valid_mask=None, sigma_rgb=self.sigma_rgb,
                    robust=self.robust, nu=self.nu,
                    layout=("rgb", "rgb", "rgb"), channel_dim=0)
            else:
                p_rgb = torch.zeros_like(r_rgb)
            if self.use_depth:
                p_depth = build_precision(
                    r_depth.unsqueeze(0), valid_mask=valid.unsqueeze(0),
                    sigma_rgb=(), depth_sigma_model=self.depth_sigma_model,
                    robust=self.robust, nu=self.nu,
                    layout=("depth",), channel_dim=0).squeeze(0)
            else:
                p_depth = torch.zeros_like(r_depth)
            residuals.append((r_rgb, r_depth))
            precisions.append((p_rgb, p_depth))
            used.append(camera_index)

        return FrameCache(original_frame=int(original), linearization=lin,
                          residuals=residuals, precisions=precisions,
                          num_gaussians=int(state.mu.shape[0]), cameras=used)

    def build_inventory(self) -> dict:
        """Register one factor per (original frame, camera), exactly once.

        Deliberately NOT done inside `linearize_frame`: that is called again
        at every outer iteration and for every line-search trial, and an
        inventory that grew on each call would make relinearisation look like
        new evidence. The factor set is fixed here and only its values change
        afterwards.
        """
        if self._inventory_built:
            return self.inventory.as_manifest()
        for frame_index, original in enumerate(self.frame_ids):
            for camera_index, observed in enumerate(self.observations[frame_index]):
                if observed is not None:
                    self.inventory.register(FactorKey(int(original), camera_index))
        self._inventory_built = True
        return self.inventory.as_manifest()

    # -- matrix-free operators --------------------------------------------
    def jvp(self, cache: FrameCache, delta_x: torch.Tensor) -> list:
        """12-D tangent -> pixel variation, through the existing CUDA H.

        Only the pose block acts. The velocity columns are structurally zero,
        not numerically small: `extract_pose` drops them before the renderer
        ever sees the tangent.
        """
        if delta_x.shape[-1] != STATE_DIM:
            raise ValueError(f"expected a [P,{STATE_DIM}] tangent, got {tuple(delta_x.shape)}")
        pose_delta = extract_pose(delta_x)
        return cache.linearization.apply_H(pose_delta)

    def vjp(self, cache: FrameCache, pixel_adjoints) -> torch.Tensor:
        """Pixel adjoint -> 12-D tangent adjoint, velocity block zero."""
        pose_adjoint = cache.linearization.apply_HT(pixel_adjoints)
        return embed_pose_to_full_state(pose_adjoint)

    def precision_apply(self, cache: FrameCache, pixel_vectors) -> list:
        """Apply the existing effective precision, channel for channel."""
        out = []
        for (d_rgb, d_depth), precision in zip(pixel_vectors, cache.precisions):
            if precision is None:
                out.append((torch.zeros_like(d_rgb), torch.zeros_like(d_depth)))
                continue
            p_rgb, p_depth = precision
            out.append((p_rgb * d_rgb, p_depth * d_depth))
        return out

    def information_apply(self, cache: FrameCache, delta_x: torch.Tensor) -> torch.Tensor:
        """H^T R^-1 H u for one frame, all cameras, matrix-free."""
        return self.vjp(cache, self.precision_apply(cache, self.jvp(cache, delta_x)))

    def gradient(self, cache: FrameCache) -> torch.Tensor:
        """H^T R^-1 r for one frame, all cameras."""
        weighted = []
        for residual, precision in zip(cache.residuals, cache.precisions):
            if residual is None or precision is None:
                shape_source = next(r for r in cache.residuals if r is not None)
                weighted.append((torch.zeros_like(shape_source[0]),
                                 torch.zeros_like(shape_source[1])))
                continue
            (r_rgb, r_depth), (p_rgb, p_depth) = residual, precision
            weighted.append((p_rgb * r_rgb, p_depth * r_depth))
        return self.vjp(cache, weighted)

    # -- the true nonlinear objective -------------------------------------
    def nonlinear_cost(self, state: GaussianMotionState, frame_index: int) -> dict:
        """Rerender and evaluate the ACTUAL measurement objective.

        Used by the line search. This is a fresh render at the candidate
        state, not a quadratic model evaluated with frozen weights -- the
        distinction matters because accepting on the surrogate while
        reporting the posterior is how an increase gets called a decrease.
        """
        cache = self.linearize_frame(state, frame_index)
        rgb_energy = depth_energy = 0.0
        for residual, precision in zip(cache.residuals, cache.precisions):
            if residual is None:
                continue
            (r_rgb, r_depth), (p_rgb, p_depth) = residual, precision
            rgb_energy += float(measurement_energy(r_rgb, p_rgb))
            depth_energy += float(measurement_energy(r_depth, p_depth))
        return {"rgb": rgb_energy, "depth": depth_energy,
                "total": rgb_energy + depth_energy,
                "original_frame": cache.original_frame,
                "cameras": len(cache.cameras)}


def adjoint_defect(observation: FreeRGBDObservation, cache: FrameCache,
                   u: torch.Tensor, z) -> dict:
    """<H u, z> against <u, H^T z>, on the real CUDA operator."""
    Hu = observation.jvp(cache, u)
    lhs = sum(float((a * c).sum()) + float((b * d).sum())
              for (a, b), (c, d) in zip(Hu, z))
    HTz = observation.vjp(cache, z)
    rhs = float((u * HTz).sum())
    scale = max(abs(lhs), abs(rhs), 1e-30)
    return {"lhs": lhs, "rhs": rhs, "abs_error": abs(lhs - rhs),
            "rel_error": abs(lhs - rhs) / scale}
