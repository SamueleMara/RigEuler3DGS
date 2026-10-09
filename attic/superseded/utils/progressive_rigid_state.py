"""State for the progressive rigid discovery Stage-1 pipeline.

The key invariant is simple and deliberate: flow matching gives immutable
endpoint measurements, then Stage 1 only chooses among explicit hypotheses:
static, or one rigid part transform.  Uncertainty is a status bit, not a
fractional motion model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = 1
PIPELINE_NAME = "progressive_rigid_discovery"

STATIC_LABEL = 0
STATUS_UNOBSERVABLE = -2
STATUS_UNCERTAIN = -1
STATUS_TRUSTED_STATIC = 0
STATUS_TRUSTED_PART_BASE = 1

PROVENANCE_NONE = 0
PROVENANCE_FM_CORE = 1
PROVENANCE_RIGID_INIT = 2
PROVENANCE_COMPONENT_ACCEPT = 3
PROVENANCE_ROLLBACK = 4


def status_for_part(part_label: int) -> int:
    """Status code for a trusted moving part label in ``1..K``."""
    return STATUS_TRUSTED_PART_BASE + int(part_label) - 1


@dataclass
class ProgressiveRigidState:
    """Every tensor carried by the progressive rigid discovery loop.

    Shapes use ``N`` foreground Gaussians, ``M`` all Gaussians, and ``K`` moving
    parts.  Label 0 is always static; labels 1..K are rigid moving parts.
    """

    # --- domain ---------------------------------------------------------
    canonical_all: torch.Tensor              # [M,3]
    canonical_fg: torch.Tensor               # [N,3]
    foreground_idx: torch.Tensor             # [N]
    foreground_mask: torch.Tensor            # [M]
    frame_ids: list[int]
    frame_times: list[float]
    num_moving_parts: int

    # --- immutable legacy endpoint FM ----------------------------------
    fm_endpoint_xyz: torch.Tensor            # [N,3]
    fm_displacement: torch.Tensor            # [N,3]
    fm_probability: torch.Tensor             # [N]
    fm_confidence_base: torch.Tensor         # [N]
    fm_confidence_current: torch.Tensor      # [N]
    fm_valid: torch.Tensor                   # [N]
    fm_candidate_matches: torch.Tensor | None = None  # [N,C]
    fm_candidate_costs: torch.Tensor | None = None    # [N,C]
    candidate_target_xyz: torch.Tensor | None = None   # [N,M,3] actual frame-T PCD/Gaussians
    candidate_target_idx: torch.Tensor | None = None   # [N,M] rows into frame-T target set
    candidate_fm_prior: torch.Tensor | None = None     # [N,M] match probability assuming moving
    candidate_valid: torch.Tensor | None = None        # [N,M]
    candidate_source_id: torch.Tensor | None = None    # [N,M] 1=legacy, 2=endpoint_knn, 3=t0_knn
    neighbor_idx: torch.Tensor | None = None           # [N,Kn] t0 k-NN graph, self excluded
    neighbor_dist0: torch.Tensor | None = None         # [N,Kn] t0 pairwise distance to each neighbour
    fm_dynamic_confidence: torch.Tensor | None = None  # [N] P(move) confidence, not match confidence
    correspondence_confidence: torch.Tensor | None = None  # [K,N]
    correspondence_selected: torch.Tensor | None = None     # [K,N]
    correspondence_target_xyz: torch.Tensor | None = None   # [K,N,3]
    initial_fm_dynamic_core: torch.Tensor | None = None     # [K,N]
    outer_accepted_moving: torch.Tensor | None = None       # [K,N]
    resolved_once: torch.Tensor | None = None               # [N]

    # --- K+1 assignment -------------------------------------------------
    responsibilities: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [N,K+1]
    hard_labels: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))  # [N]
    assignment_confidence: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [N]
    status: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))  # [N]
    provenance: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))  # [N]

    # --- physical hypotheses --------------------------------------------
    part_transforms_T: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [K,4,4]
    part_twists_xi: torch.Tensor = field(default_factory=lambda: torch.empty(0))     # [K,6]
    part_fit_residual: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [K]
    part_support_weight: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [K,N]

    # --- current evidence -----------------------------------------------
    fm_rigid_residual: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [K+1,N]
    pcd_residual_per_gaussian: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [N]
    render_residual_per_gaussian: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [N]
    depth_residual_per_gaussian: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [N]
    observable: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.bool))  # [N]
    visibility_mass: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [N]

    # --- progressive components -----------------------------------------
    candidate_component_id: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))  # [N]
    component_scores: list[dict[str, Any]] = field(default_factory=list)
    accepted_components: list[dict[str, Any]] = field(default_factory=list)
    rejected_components: list[dict[str, Any]] = field(default_factory=list)
    deferred_components: list[dict[str, Any]] = field(default_factory=list)
    outer_iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def num_foreground(self) -> int:
        return int(self.canonical_fg.shape[0])

    @property
    def num_all(self) -> int:
        return int(self.canonical_all.shape[0])

    @property
    def num_labels(self) -> int:
        return int(self.num_moving_parts) + 1

    @property
    def device(self):
        return self.canonical_fg.device

    def validate(self) -> "ProgressiveRigidState":
        n = self.num_foreground
        m = self.num_all
        k = int(self.num_moving_parts)
        labels = k + 1
        expected = {
            "canonical_all": (m, 3),
            "canonical_fg": (n, 3),
            "foreground_idx": (n,),
            "foreground_mask": (m,),
            "fm_endpoint_xyz": (n, 3),
            "fm_displacement": (n, 3),
            "fm_probability": (n,),
            "fm_confidence_base": (n,),
            "fm_confidence_current": (n,),
            "fm_valid": (n,),
            "responsibilities": (n, labels),
            "hard_labels": (n,),
            "assignment_confidence": (n,),
            "status": (n,),
            "provenance": (n,),
            "part_transforms_T": (k, 4, 4),
            "part_twists_xi": (k, 6),
            "part_fit_residual": (k,),
            "part_support_weight": (k, n),
            "fm_rigid_residual": (labels, n),
            "pcd_residual_per_gaussian": (n,),
            "render_residual_per_gaussian": (n,),
            "depth_residual_per_gaussian": (n,),
            "observable": (n,),
            "visibility_mass": (n,),
            "candidate_component_id": (n,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"ProgressiveRigidState.{name} must be {shape}, got {tuple(value.shape)}"
                )
        if self.fm_candidate_matches is not None and self.fm_candidate_matches.shape[0] != n:
            raise ValueError("fm_candidate_matches first dimension must be N")
        if self.fm_candidate_costs is not None and self.fm_candidate_costs.shape != self.fm_candidate_matches.shape:
            raise ValueError("fm_candidate_costs must match fm_candidate_matches")
        optional = {
            "candidate_target_xyz": (n, None, 3),
            "candidate_target_idx": (n, None),
            "candidate_fm_prior": (n, None),
            "candidate_valid": (n, None),
            "candidate_source_id": (n, None),
            "neighbor_idx": (n, None),
            "neighbor_dist0": (n, None),
            "fm_dynamic_confidence": (n,),
            "correspondence_confidence": (k, n),
            "correspondence_selected": (k, n),
            "correspondence_target_xyz": (k, n, 3),
            "initial_fm_dynamic_core": (k, n),
            "outer_accepted_moving": (k, n),
            "resolved_once": (n,),
        }
        for name, shape in optional.items():
            value = getattr(self, name)
            if value is None:
                continue
            got = tuple(value.shape)
            if len(got) != len(shape) or any(s is not None and g != s for g, s in zip(got, shape)):
                raise ValueError(f"ProgressiveRigidState.{name} must be {shape}, got {got}")
        if self.candidate_target_xyz is not None:
            m_candidates = self.candidate_target_xyz.shape[1]
            for name in ("candidate_target_idx", "candidate_fm_prior", "candidate_valid", "candidate_source_id"):
                value = getattr(self, name)
                if value is None or tuple(value.shape) != (n, m_candidates):
                    raise ValueError(f"{name} must be [N,M] when candidate_target_xyz is present")
        row_sum = self.responsibilities.sum(-1)
        if not bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1.0e-4)):
            raise ValueError("responsibility rows must sum to one")
        if bool((self.responsibilities < -1.0e-6).any()):
            raise ValueError("responsibilities must be non-negative")
        if len(self.frame_ids) != len(self.frame_times):
            raise ValueError("frame_ids and frame_times must have identical length")
        return self

    def moving_probability(self) -> torch.Tensor:
        return self.responsibilities[:, 1:].sum(-1)

    def trusted_static_mask(self) -> torch.Tensor:
        return self.status == STATUS_TRUSTED_STATIC

    def trusted_part_mask(self, part_label: int) -> torch.Tensor:
        return self.status == status_for_part(int(part_label))

    def moving_support_mask(self, part_label: int) -> torch.Tensor:
        part = int(part_label) - 1
        if self.initial_fm_dynamic_core is None:
            core = self.trusted_part_mask(part_label)[None]
        else:
            core = self.initial_fm_dynamic_core[part : part + 1].to(device=self.device).bool()
        accepted = (
            torch.zeros_like(core)
            if self.outer_accepted_moving is None
            else self.outer_accepted_moving[part : part + 1].to(device=self.device).bool()
        )
        return (core | accepted)[0]

    def uncertain_mask(self) -> torch.Tensor:
        return self.status == STATUS_UNCERTAIN

    def unobservable_mask(self) -> torch.Tensor:
        return self.status == STATUS_UNOBSERVABLE

    def part_positions(self, part_label: int) -> torch.Tensor:
        from utils.rigid_endpoint_fit import transform_points

        return transform_points(self.part_transforms_T[int(part_label) - 1], self.canonical_fg)

    def hypothesized_positions(self, labels: torch.Tensor | None = None) -> torch.Tensor:
        labels = self.hard_labels if labels is None else torch.as_tensor(labels, device=self.device).long()
        out = self.canonical_fg.clone()
        for part in range(1, self.num_labels):
            mask = labels == part
            if bool(mask.any()):
                out[mask] = self.part_positions(part)[mask]
        return out

    def scatter_positions_to_full(self, foreground_positions: torch.Tensor) -> torch.Tensor:
        full = self.canonical_all.clone()
        full[self.foreground_idx.to(full.device)] = foreground_positions.to(full)
        return full

    def label_counts(self) -> dict[str, int]:
        result = {
            "trusted_static": int(self.trusted_static_mask().sum()),
            "uncertain": int(self.uncertain_mask().sum()),
            "unobservable": int(self.unobservable_mask().sum()),
        }
        for part in range(1, self.num_labels):
            result[f"trusted_part_{part}"] = int(self.trusted_part_mask(part).sum())
        return result

    def enforce_committed_labels(self) -> None:
        """Trusted statuses always override soft argmax labels."""
        self.hard_labels[self.trusted_static_mask()] = 0
        for part in range(1, self.num_labels):
            self.hard_labels[self.trusted_part_mask(part)] = part

    def clone(self) -> "ProgressiveRigidState":
        payload = self.to_payload()
        return ProgressiveRigidState.from_payload(payload, device=self.device)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "pipeline": PIPELINE_NAME,
            "frame_ids": list(self.frame_ids),
            "frame_times": list(self.frame_times),
            "num_moving_parts": int(self.num_moving_parts),
            "outer_iteration": int(self.outer_iteration),
            "component_scores": list(self.component_scores),
            "accepted_components": list(self.accepted_components),
            "rejected_components": list(self.rejected_components),
            "deferred_components": list(self.deferred_components),
            "history": list(self.history),
        }
        skip = {
            "frame_ids", "frame_times", "num_moving_parts", "outer_iteration",
            "component_scores", "accepted_components", "rejected_components",
            "deferred_components", "history",
        }
        for item in fields(self):
            if item.name in skip:
                continue
            value = getattr(self, item.name)
            payload[item.name] = value.detach().cpu() if torch.is_tensor(value) else value
        return payload

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.to_payload(), temporary)
        temporary.replace(path)
        return path

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, device=None) -> "ProgressiveRigidState":
        kwargs: dict[str, Any] = {}
        tensor_names = {item.name for item in fields(cls)}
        for name in tensor_names:
            if name not in payload:
                continue
            value = payload[name]
            if torch.is_tensor(value):
                kwargs[name] = value.to(device=device) if device is not None else value
            else:
                kwargs[name] = value
        kwargs["frame_ids"] = list(payload["frame_ids"])
        kwargs["frame_times"] = list(payload["frame_times"])
        kwargs["num_moving_parts"] = int(payload["num_moving_parts"])
        kwargs["outer_iteration"] = int(payload.get("outer_iteration", 0))
        kwargs["component_scores"] = list(payload.get("component_scores", []))
        kwargs["accepted_components"] = list(payload.get("accepted_components", []))
        kwargs["rejected_components"] = list(payload.get("rejected_components", []))
        kwargs["deferred_components"] = list(payload.get("deferred_components", []))
        kwargs["history"] = list(payload.get("history", []))
        return cls(**kwargs).validate()


def allocate_progressive_state(
    *,
    canonical_all: torch.Tensor,
    canonical_fg: torch.Tensor,
    foreground_idx: torch.Tensor,
    foreground_mask: torch.Tensor,
    frame_ids: list[int],
    frame_times: list[float],
    num_moving_parts: int,
    fm_endpoint_xyz: torch.Tensor,
    fm_displacement: torch.Tensor,
    fm_probability: torch.Tensor,
    fm_confidence: torch.Tensor,
    fm_valid: torch.Tensor,
    fm_candidate_matches: torch.Tensor | None = None,
    fm_candidate_costs: torch.Tensor | None = None,
    candidate_target_xyz: torch.Tensor | None = None,
    candidate_target_idx: torch.Tensor | None = None,
    candidate_fm_prior: torch.Tensor | None = None,
    candidate_valid: torch.Tensor | None = None,
    candidate_source_id: torch.Tensor | None = None,
) -> ProgressiveRigidState:
    device = canonical_fg.device
    dtype = canonical_fg.dtype
    n = int(canonical_fg.shape[0])
    k = int(num_moving_parts)
    labels = k + 1
    responsibilities = torch.full((n, labels), 1.0 / float(labels), device=device, dtype=dtype)
    hard_labels = torch.zeros(n, device=device, dtype=torch.long)
    transforms = torch.eye(4, device=device, dtype=dtype).expand(k, 4, 4).clone()
    state = ProgressiveRigidState(
        canonical_all=canonical_all.detach().to(device=device, dtype=dtype).clone(),
        canonical_fg=canonical_fg.detach().to(device=device, dtype=dtype).clone(),
        foreground_idx=foreground_idx.detach().to(device=device).long().reshape(-1),
        foreground_mask=foreground_mask.detach().to(device=device).bool().reshape(-1),
        frame_ids=list(frame_ids),
        frame_times=list(frame_times),
        num_moving_parts=k,
        fm_endpoint_xyz=fm_endpoint_xyz.detach().to(device=device, dtype=dtype).clone(),
        fm_displacement=fm_displacement.detach().to(device=device, dtype=dtype).clone(),
        fm_probability=fm_probability.detach().to(device=device, dtype=dtype).reshape(-1).clamp(0.0, 1.0),
        fm_confidence_base=fm_confidence.detach().to(device=device, dtype=dtype).reshape(-1).clamp(0.0, 1.0),
        fm_confidence_current=fm_confidence.detach().to(device=device, dtype=dtype).reshape(-1).clamp(0.0, 1.0),
        fm_valid=fm_valid.detach().to(device=device).bool().reshape(-1),
        fm_candidate_matches=None if fm_candidate_matches is None else fm_candidate_matches.detach().cpu().long(),
        fm_candidate_costs=None if fm_candidate_costs is None else fm_candidate_costs.detach().cpu().float(),
        candidate_target_xyz=None if candidate_target_xyz is None else candidate_target_xyz.detach().to(device=device, dtype=dtype).clone(),
        candidate_target_idx=None if candidate_target_idx is None else candidate_target_idx.detach().to(device=device).long().clone(),
        candidate_fm_prior=None if candidate_fm_prior is None else candidate_fm_prior.detach().to(device=device, dtype=dtype).clone(),
        candidate_valid=None if candidate_valid is None else candidate_valid.detach().to(device=device).bool().clone(),
        candidate_source_id=None if candidate_source_id is None else candidate_source_id.detach().to(device=device).long().clone(),
        fm_dynamic_confidence=fm_confidence.detach().to(device=device, dtype=dtype).reshape(-1).clamp(0.0, 1.0),
        correspondence_confidence=torch.zeros((k, n), device=device, dtype=dtype),
        correspondence_selected=torch.full((k, n), -1, device=device, dtype=torch.long),
        correspondence_target_xyz=torch.zeros((k, n, 3), device=device, dtype=dtype),
        initial_fm_dynamic_core=torch.zeros((k, n), device=device, dtype=torch.bool),
        outer_accepted_moving=torch.zeros((k, n), device=device, dtype=torch.bool),
        resolved_once=torch.zeros(n, device=device, dtype=torch.bool),
        responsibilities=responsibilities,
        hard_labels=hard_labels,
        assignment_confidence=torch.zeros(n, device=device, dtype=dtype),
        status=torch.full((n,), STATUS_UNCERTAIN, device=device, dtype=torch.long),
        provenance=torch.full((n,), PROVENANCE_NONE, device=device, dtype=torch.long),
        part_transforms_T=transforms,
        part_twists_xi=torch.zeros((k, 6), device=device, dtype=dtype),
        part_fit_residual=torch.full((k,), float("inf"), device=device, dtype=dtype),
        part_support_weight=torch.zeros((k, n), device=device, dtype=dtype),
        fm_rigid_residual=torch.full((labels, n), float("inf"), device=device, dtype=dtype),
        pcd_residual_per_gaussian=torch.zeros(n, device=device, dtype=dtype),
        render_residual_per_gaussian=torch.zeros(n, device=device, dtype=dtype),
        depth_residual_per_gaussian=torch.zeros(n, device=device, dtype=dtype),
        observable=torch.ones(n, device=device, dtype=torch.bool),
        visibility_mass=torch.ones(n, device=device, dtype=dtype),
        candidate_component_id=torch.full((n,), -1, device=device, dtype=torch.long),
    )
    return state.validate()


def load_progressive_state(path: str | Path, *, device=None) -> ProgressiveRigidState:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    return ProgressiveRigidState.from_payload(payload, device=device)
