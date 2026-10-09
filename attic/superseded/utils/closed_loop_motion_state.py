"""Explicit state for the closed-loop diffusion / flow-matching / Killing pipeline.

Every outer cycle of the loop consumes and updates this one object.  The state is
deliberately not hidden inside the FM model, the Killing solver or the denoiser:
the whole point of the pipeline is that those four modules disagree with each
other in a way that is visible, so what they disagree about has to be a first
class, inspectable thing.

Two invariants hold everywhere in this module and in everything that touches it.

**Foreground only.**  Every motion tensor is indexed by ``N_F``, the number of
canonical foreground Gaussians, never by the full cloud.  ``foreground_idx``
scatters back to the full model for rendering; nothing scatters the other way.

**Static is label 0 and is exactly zero motion.**  There is no fitted static
field.  ``responsibilities[:, 0]`` is the static share and the static hypothesis
contributes ``v = 0`` and ``T = I`` by construction, not by fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


SCHEMA_VERSION = 1
PIPELINE_NAME = "closed_loop_diffusion_fm_killing"

#: Label 0 is static in every tensor with a ``K+1`` axis.
STATIC_LABEL = 0


@dataclass
class ClosedLoopMotionState:
    """Everything the closed loop carries between outer cycles.

    Shapes, with ``N_F`` foreground Gaussians, ``T`` frames and ``K`` moving
    parts.  ``T - 1`` is the number of frame intervals.
    """

    # --- domain ---------------------------------------------------------
    canonical_fg: torch.Tensor          # [N_F,3] canonical foreground positions
    foreground_idx: torch.Tensor        # [N_F]   indices into the full cloud
    frame_ids: list[int]                # [T]     physical frame ids
    frame_times: list[float]            # [T]     physical seconds
    num_moving_parts: int               # K

    # --- assignment -----------------------------------------------------
    responsibilities: torch.Tensor      # [N_F,K+1] rows sum to one
    assignment_entropy: torch.Tensor    # [N_F]
    hard_labels: torch.Tensor           # [N_F]     argmax, for rendering only

    # --- flow matching --------------------------------------------------
    fm_base_displacement: torch.Tensor  # [N_F,3] immutable legacy endpoint transport
    fm_base_velocity: torch.Tensor      # [T-1,N_F,3]
    fm_base_probability: torch.Tensor   # [N_F]
    fm_base_confidence: torch.Tensor    # [T-1,N_F]
    fm_base_valid: torch.Tensor         # [N_F]
    fm_velocity: torch.Tensor           # [T-1,N_F,3]
    fm_confidence: torch.Tensor         # [T-1,N_F]
    fm_current_displacement: torch.Tensor  # [N_F,3]
    fm_current_confidence: torch.Tensor    # [T-1,N_F]

    # --- rendered physical transport feedback --------------------------
    physical_displacement: torch.Tensor    # [N_F,3]
    physical_velocity: torch.Tensor        # [T-1,N_F,3]
    fm_physical_3d_residual: torch.Tensor  # [N_F]

    # --- fused motion estimate ------------------------------------------
    consensus_velocity: torch.Tensor    # [T-1,N_F,3] FM + observation + physics

    # --- rasterized multiview observations ------------------------------
    observed_velocity: torch.Tensor     # [T-1,N_F,3]
    observed_confidence: torch.Tensor   # [T-1,N_F]
    observed_normal_matrix: torch.Tensor  # [T-1,N_F,3,3]
    observed_valid: torch.Tensor        # [T-1,N_F]

    # --- Killing physics ------------------------------------------------
    killing_modes: list                 # K RigidKillingMode
    killing_interval_speeds: torch.Tensor  # [K,T-1]
    killing_velocity: torch.Tensor      # [K,T-1,N_F,3]
    killing_positions: torch.Tensor     # [K,T,N_F,3]

    # --- residual diffusion ---------------------------------------------
    diffusion_residual_mean: torch.Tensor  # [T,N_F,3]
    diffusion_residual_cov: torch.Tensor   # [T,N_F,3,3]
    diffusion_confidence: torch.Tensor     # [T,N_F]

    # --- assignment evidence --------------------------------------------
    label_energies: torch.Tensor        # [N_F,K+1]
    energy_fm_3d: torch.Tensor          # [N_F,K+1]
    energy_fm_render: torch.Tensor      # [N_F,K+1]
    energy_obs_render: torch.Tensor     # [N_F,K+1]
    energy_target_render: torch.Tensor  # [N_F,K+1]

    # --- trusted core / uncertain shell ---------------------------------
    core_masks: torch.Tensor            # [K,N_F]
    shell_masks: torch.Tensor           # [K,N_F]
    static_core_mask: torch.Tensor      # [N_F]
    moving_streak: torch.Tensor         # [N_F,K] consecutive cycles above enter
    static_streak: torch.Tensor         # [N_F]

    outer_iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def num_foreground(self) -> int:
        return int(self.canonical_fg.shape[0])

    @property
    def num_frames(self) -> int:
        return len(self.frame_times)

    @property
    def num_intervals(self) -> int:
        return max(0, self.num_frames - 1)

    @property
    def num_labels(self) -> int:
        return int(self.num_moving_parts) + 1

    def validate(self) -> None:
        """Fail loudly on any shape or invariant violation.

        Called after every stage that writes state.  A silent shape drift here
        would surface many stages later as a meaningless energy.
        """
        n, t, k = self.num_foreground, self.num_frames, self.num_moving_parts
        intervals, labels = self.num_intervals, self.num_labels
        expected = {
            "canonical_fg": (n, 3),
            "foreground_idx": (n,),
            "responsibilities": (n, labels),
            "assignment_entropy": (n,),
            "hard_labels": (n,),
            "fm_base_displacement": (n, 3),
            "fm_base_velocity": (intervals, n, 3),
            "fm_base_probability": (n,),
            "fm_base_confidence": (intervals, n),
            "fm_base_valid": (n,),
            "fm_velocity": (intervals, n, 3),
            "fm_confidence": (intervals, n),
            "fm_current_displacement": (n, 3),
            "fm_current_confidence": (intervals, n),
            "physical_displacement": (n, 3),
            "physical_velocity": (intervals, n, 3),
            "fm_physical_3d_residual": (n,),
            "consensus_velocity": (intervals, n, 3),
            "observed_velocity": (intervals, n, 3),
            "observed_confidence": (intervals, n),
            "observed_normal_matrix": (intervals, n, 3, 3),
            "observed_valid": (intervals, n),
            "killing_interval_speeds": (k, intervals),
            "killing_velocity": (k, intervals, n, 3),
            "killing_positions": (k, t, n, 3),
            "diffusion_residual_mean": (t, n, 3),
            "diffusion_residual_cov": (t, n, 3, 3),
            "diffusion_confidence": (t, n),
            "label_energies": (n, labels),
            "energy_fm_3d": (n, labels),
            "energy_fm_render": (n, labels),
            "energy_obs_render": (n, labels),
            "energy_target_render": (n, labels),
            "core_masks": (k, n),
            "shell_masks": (k, n),
            "static_core_mask": (n,),
            "moving_streak": (n, k),
            "static_streak": (n,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"ClosedLoopMotionState.{name} must be {shape}, got {tuple(value.shape)}"
                )
        if len(self.killing_modes) != k:
            raise ValueError(f"expected {k} Killing modes, got {len(self.killing_modes)}")
        if len(self.frame_ids) != t:
            raise ValueError("frame_ids and frame_times must have the same length")
        row_sum = self.responsibilities.sum(-1)
        if not bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1.0e-4)):
            raise ValueError("responsibility rows must sum to one")
        if bool((self.responsibilities < 0).any()):
            raise ValueError("responsibilities must be non-negative")

    # ------------------------------------------------------------------
    def moving_probability(self) -> torch.Tensor:
        """Total probability of being moved by any part: ``1 - r_static``."""
        return self.responsibilities[:, STATIC_LABEL + 1 :].sum(-1)

    def scatter_to_full(self, values: torch.Tensor, total: int, fill=0.0) -> torch.Tensor:
        """Place a foreground-only tensor back into full-cloud indexing.

        The only direction that is allowed.  Background Gaussians take ``fill``
        and are never read back into any motion tensor.
        """
        shape = (int(total),) + tuple(values.shape[1:])
        full = torch.full(shape, float(fill), device=values.device, dtype=values.dtype)
        full[self.foreground_idx.to(values.device)] = values
        return full

    def label_counts(self) -> dict[str, int]:
        counts = torch.bincount(self.hard_labels.reshape(-1), minlength=self.num_labels)
        result = {"static": int(counts[STATIC_LABEL])}
        for part in range(1, self.num_labels):
            result[f"part_{part}"] = int(counts[part])
        return result

    # ------------------------------------------------------------------
    def to_payload(self) -> dict[str, Any]:
        """Serialisable snapshot; Killing modes go through their own dict form."""
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "pipeline": PIPELINE_NAME,
            "num_moving_parts": int(self.num_moving_parts),
            "frame_ids": list(self.frame_ids),
            "frame_times": list(self.frame_times),
            "outer_iteration": int(self.outer_iteration),
            "history": list(self.history),
            "killing_modes": [mode.as_dict() for mode in self.killing_modes],
        }
        skip = {"killing_modes", "frame_ids", "frame_times", "num_moving_parts",
                "outer_iteration", "history"}
        for item in fields(self):
            if item.name in skip:
                continue
            value = getattr(self, item.name)
            payload[item.name] = value.detach().cpu() if torch.is_tensor(value) else value
        return payload

    def save(self, path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(self.to_payload(), temporary)
        temporary.replace(destination)


def allocate_closed_loop_state(
    canonical_fg: torch.Tensor,
    foreground_idx: torch.Tensor,
    frame_ids,
    frame_times,
    num_moving_parts: int,
    *,
    responsibilities: torch.Tensor | None = None,
    static_prior: float = 0.7,
    device=None,
) -> ClosedLoopMotionState:
    """Build a zeroed state of the right shapes with soft initial assignments.

    ``static_prior`` is only used when ``responsibilities`` is not supplied.  The
    specification asks for a static-biased but fully mutable start: nothing here
    is frozen, and every entry stays strictly inside ``(0, 1)`` so no label is
    unreachable by a later softmax update.
    """
    points = torch.as_tensor(canonical_fg).detach()
    device = points.device if device is None else torch.device(device)
    points = points.to(device=device, dtype=torch.float32)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("canonical_fg must be [N_F,3]")
    n = int(points.shape[0])
    if n == 0:
        raise ValueError("the closed loop needs at least one foreground Gaussian")
    k = int(num_moving_parts)
    if k < 1:
        raise ValueError("num_moving_parts must be at least one")
    frame_ids = [int(value) for value in frame_ids]
    frame_times = [float(value) for value in frame_times]
    t = len(frame_times)
    if t < 2:
        raise ValueError("the closed loop needs at least two frames")
    intervals = t - 1
    labels = k + 1

    if responsibilities is None:
        static_prior = float(min(max(static_prior, 1.0e-3), 1.0 - 1.0e-3))
        responsibilities = torch.empty((n, labels), device=device, dtype=torch.float32)
        responsibilities[:, STATIC_LABEL] = static_prior
        responsibilities[:, STATIC_LABEL + 1 :] = (1.0 - static_prior) / k
    else:
        responsibilities = torch.as_tensor(responsibilities).to(
            device=device, dtype=torch.float32
        )
        if tuple(responsibilities.shape) != (n, labels):
            raise ValueError(f"responsibilities must be [{n},{labels}]")
        responsibilities = responsibilities.clamp_min(0.0)
        responsibilities = responsibilities / responsibilities.sum(-1, keepdim=True).clamp_min(1.0e-8)

    zeros = lambda *shape: torch.zeros(shape, device=device, dtype=torch.float32)  # noqa: E731
    eye = torch.eye(3, device=device, dtype=torch.float32)
    state = ClosedLoopMotionState(
        canonical_fg=points,
        foreground_idx=torch.as_tensor(foreground_idx).to(device=device, dtype=torch.long),
        frame_ids=frame_ids,
        frame_times=frame_times,
        num_moving_parts=k,
        responsibilities=responsibilities,
        assignment_entropy=compute_entropy(responsibilities),
        hard_labels=responsibilities.argmax(-1),
        fm_base_displacement=zeros(n, 3),
        fm_base_velocity=zeros(intervals, n, 3),
        fm_base_probability=zeros(n),
        fm_base_confidence=zeros(intervals, n),
        fm_base_valid=torch.zeros(n, device=device, dtype=torch.bool),
        fm_velocity=zeros(intervals, n, 3),
        fm_confidence=zeros(intervals, n),
        fm_current_displacement=zeros(n, 3),
        fm_current_confidence=zeros(intervals, n),
        physical_displacement=zeros(n, 3),
        physical_velocity=zeros(intervals, n, 3),
        fm_physical_3d_residual=zeros(n),
        consensus_velocity=zeros(intervals, n, 3),
        observed_velocity=zeros(intervals, n, 3),
        observed_confidence=zeros(intervals, n),
        observed_normal_matrix=eye.expand(intervals, n, 3, 3).clone(),
        observed_valid=torch.zeros((intervals, n), device=device, dtype=torch.bool),
        killing_modes=[],
        killing_interval_speeds=zeros(k, intervals),
        killing_velocity=zeros(k, intervals, n, 3),
        killing_positions=points[None, None].expand(k, t, n, 3).clone(),
        diffusion_residual_mean=zeros(t, n, 3),
        diffusion_residual_cov=zeros(t, n, 3, 3),
        diffusion_confidence=zeros(t, n),
        label_energies=zeros(n, labels),
        energy_fm_3d=zeros(n, labels),
        energy_fm_render=zeros(n, labels),
        energy_obs_render=zeros(n, labels),
        energy_target_render=zeros(n, labels),
        core_masks=torch.zeros((k, n), device=device, dtype=torch.bool),
        shell_masks=torch.zeros((k, n), device=device, dtype=torch.bool),
        static_core_mask=torch.zeros(n, device=device, dtype=torch.bool),
        moving_streak=torch.zeros((n, k), device=device, dtype=torch.long),
        static_streak=torch.zeros(n, device=device, dtype=torch.long),
    )
    # Modes are allocated as identity-speed zero fields until the first fit, so
    # every shape is valid from cycle zero and validate() can run immediately.
    from utils.rigid_killing_motion import RigidKillingMode

    state.killing_modes = [
        RigidKillingMode(
            reference_point=points.double().mean(dim=0).cpu(),
            reference_velocity=torch.zeros(3, dtype=torch.float64),
            angular_mode=torch.zeros(3, dtype=torch.float64),
            mode_type="translational",
            normalization_scale=0.0,
            valid=False,
            diagnostics={"initialised": "no_fit_yet"},
        )
        for _ in range(k)
    ]
    state.validate()
    return state


def compute_entropy(responsibilities: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Shannon entropy of each responsibility row, in nats."""
    value = torch.as_tensor(responsibilities)
    return -(value * torch.log(value.clamp_min(eps))).sum(-1)


def extract_foreground(gaussians, args) -> dict[str, torch.Tensor]:
    """Resolve the canonical foreground domain from a GaussianModel.

    The closed loop never touches background Gaussians, so this is the one place
    the domain is decided.

    ``get_dynamic_foreground_mask`` is a property that *raises* when the mask has
    not been built, so it cannot be probed with ``getattr(..., None)``: the
    property runs and the exception propagates.  The mask is built here from the
    semantic field the canonical model was trained with, which is what the
    caller actually has after loading a PLY.
    """
    xyz = gaussians.get_canonical_xyz.detach()
    total = int(xyz.shape[0])
    threshold = float(getattr(args, "semantic_dynamic_threshold", 0.5))
    minimum = int(getattr(args, "semantic_dynamic_min_points", 32))

    mask = None
    existing = getattr(gaussians, "_dynamic_foreground_mask", None)
    if torch.is_tensor(existing) and existing.numel() == total:
        mask = existing.bool()
    else:
        probability = gaussians.get_semantic_probability
        if probability is not None and probability.numel() == total:
            builder_args = SimpleNamespace(
                dynamic_use_all_gaussians=False,
                semantic_dynamic_threshold=threshold,
                semantic_dynamic_min_points=minimum,
            )
            mask = gaussians.build_dynamic_foreground_mask(builder_args).bool()
        elif bool(getattr(args, "allow_all_gaussian_foreground", False)):
            # Opt-in only.  Treating a whole scene as foreground silently turns
            # the loop's foreground-only invariant into a no-op and lets
            # background geometry into the Killing fit, so it has to be asked
            # for rather than fallen back to.
            mask = torch.ones(total, dtype=torch.bool, device=xyz.device)
            print(
                "[closed loop] no semantic field; treating all "
                f"{total} Gaussians as foreground because "
                "--allow-all-gaussian-foreground was set"
            )
        else:
            raise RuntimeError(
                "the canonical model has no trained semantic field, so the "
                "foreground cannot be resolved. Train it with "
                "--use_semantic_feature_field, or pass "
                "--allow-all-gaussian-foreground to run motion discovery over "
                "the entire cloud."
            )

    count = int(mask.sum())
    if count < minimum:
        raise RuntimeError(
            f"foreground has {count} Gaussians at semantic threshold {threshold}, "
            f"fewer than the required {minimum}; lower "
            f"--semantic-dynamic-threshold or check the semantic field"
        )
    index = torch.nonzero(mask, as_tuple=False).flatten()
    return {
        "foreground_mask": mask,
        "foreground_idx": index,
        "canonical_fg": xyz[index].contiguous(),
        "total_gaussians": total,
        "semantic_threshold": threshold,
    }
