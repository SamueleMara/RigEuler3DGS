"""The debug state recorder.

CODEX_ADD_VISUAL_AND_NUMERICAL_DEBUG_GEOMETRIC_PIPELINE.md sections 0, 2, 4.

Section 0 rules 2-4 are the reason this is a RECORDER and not a re-simulation:
it stores the exact optimized states the estimator produced at each stage, so
the animation and the numbers come from inference rather than from an
approximate model built for display. Section 2's three transport laws

    rigid      x_{i,t} = R_{k,t} x_i^0 + t_{k,t}
    prismatic  x_{i,t} = x_i^0 + q_{k,t} e_k
    revolute   x_{i,t} = a_k + R(e_k, q_{k,t}) (x_i^0 - a_k)

are evaluated here from the estimator's own transforms and joint parameters;
no separate animation model exists anywhere in this package.

When `enabled` is False every method returns immediately and nothing is
retained, so a production run pays nothing for the debug layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

# Section 4: stable part colours. Index 0 is the static set, 1..K the moving
# parts. The mapping is fixed here and persisted, so a part never changes
# colour between frames, stages or runs.
PART_COLORS = [
    "#8d9196",   # 0 static  -- deliberately grey and desaturated
    "#e4572e",   # 1
    "#2e86ab",   # 2
    "#17bb75",   # 3
    "#f3a712",   # 4
    "#9d4edd",   # 5
    "#d81e5b",   # 6
    "#00b4d8",   # 7
]


def part_color(k: int) -> str:
    """Stable colour for label k (0 = static)."""
    return PART_COLORS[int(k) % len(PART_COLORS)]


@dataclass
class GeometricDebug:
    """Everything the debug layer needs, collected as inference runs.

    Every field is populated by the production pipeline at the stage that owns
    it. Nothing here is computed twice: the transforms, labels, joint fits and
    trajectories are the estimator's own objects.
    """

    enabled: bool = False
    out_dir: object = None
    args: object = None

    # ---- stage 0: canonical ------------------------------------------------
    mu0: torch.Tensor | None = None
    R0: torch.Tensor | None = None
    scales: torch.Tensor | None = None
    colors: torch.Tensor | None = None
    opacity: torch.Tensor | None = None
    supernode_id: torch.Tensor | None = None
    num_supernodes: int = 0
    canonical_hash: str | None = None

    # ---- scene / run metadata ---------------------------------------------
    model: str = ""
    dataset: str = ""
    K: int = 0
    frame_ids: list = field(default_factory=list)
    camera_meta: list = field(default_factory=list)
    image_size: tuple = (0, 0)
    dt: float = 1.0
    started: float = field(default_factory=time.perf_counter)

    # ---- stage 1: bootstrap motion evidence --------------------------------
    positions: list = field(default_factory=list)       # [T] of [N,3]
    motion_info: dict = field(default_factory=dict)
    motion_statistic: torch.Tensor | None = None        # [C] section 6 T_c
    support_null_threshold: float = 0.0

    # ---- stage 2: rigid parts ----------------------------------------------
    transforms: object = None                           # RigidTransforms
    labels: torch.Tensor | None = None                  # [N]
    node_labels: torch.Tensor | None = None             # [C]
    rigid_history: list = field(default_factory=list)
    rigid_residual: torch.Tensor | None = None          # [N]
    rigid_converged: bool = False
    rigid_stop_reason: str = ""
    assignment_costs: torch.Tensor | None = None        # [C, K+1]
    seed_labels: torch.Tensor | None = None             # [C]

    # ---- stage 3: geometric joint fit --------------------------------------
    joints: list = field(default_factory=list)
    direct_joint_estimates: list = field(default_factory=list)
    direct_joint_heat: dict = field(default_factory=dict)

    # ---- stage 4: renderer refinement --------------------------------------
    scene: object = None                                # SceneModel
    q_traj: list = field(default_factory=list)
    refine_info: dict = field(default_factory=dict)
    energy_by_stage: dict = field(default_factory=dict)
    support_reopen: list = field(default_factory=list)

    # ---- stage 5: reduced filter -------------------------------------------
    reduced: dict = field(default_factory=dict)
    handover: dict = field(default_factory=dict)

    # ---- per-frame / per-camera measurements -------------------------------
    frame_rows: list = field(default_factory=list)
    camera_rows: list = field(default_factory=list)
    multiview: list = field(default_factory=list)
    observability: dict = field(default_factory=dict)

    status: str = "unknown"
    notes: list = field(default_factory=list)
    render_panels: list = field(default_factory=list)   # section 12

    # ======================================================================
    def record(self, **kw) -> None:
        """Store named fields. A no-op when debugging is off."""
        if not self.enabled:
            return
        for k, v in kw.items():
            setattr(self, k, v)

    def add_frame_row(self, row: dict) -> None:
        if self.enabled:
            self.frame_rows.append(row)

    def add_camera_row(self, row: dict) -> None:
        if self.enabled:
            self.camera_rows.append(row)

    def note(self, text: str) -> None:
        if self.enabled:
            self.notes.append(text)

    # ======================================================================
    @property
    def num_frames(self) -> int:
        return len(self.frame_ids) or len(self.positions)

    @property
    def num_gaussians(self) -> int:
        return 0 if self.mu0 is None else int(self.mu0.shape[0])

    def label_of(self, i=None):
        if self.labels is None:
            return None
        return self.labels if i is None else self.labels[i]

    # ---- section 2: the exact transports, from the estimator's own state ---
    def positions_at(self, stage: str, t: int) -> torch.Tensor | None:
        """Gaussian centres at frame `t` for one estimator stage.

        `stage` is one of canonical, bootstrap, rigid, joint, refined. Each is
        read from the state that stage produced -- section 0 rule 3 forbids
        regenerating an approximate motion for display.
        """
        if self.mu0 is None:
            return None
        if stage == "canonical":
            return self.mu0
        if stage == "bootstrap":
            return (self.positions[t] if t < len(self.positions) else self.mu0)
        if stage == "rigid":
            if self.transforms is None or self.labels is None:
                return None
            from estimator.rigid_part_estimator import apply_transforms
            mu, _ = apply_transforms(self.mu0, self.R0, self.labels,
                                     self.transforms, t)
            return mu
        if stage in ("joint", "refined"):
            if self.scene is None or self.labels is None or not self.q_traj:
                return None
            qt = [self.q_traj[k][t] for k in range(len(self.q_traj))]
            mu, _ = self.scene.transport_all(self.mu0, self.R0, self.labels, qt)
            return mu
        return None

    def orientations_at(self, stage: str, t: int) -> torch.Tensor | None:
        """Section 3: Gaussian orientation R^G_{i,t} at one stage."""
        if self.R0 is None:
            return None
        if stage == "canonical":
            return self.R0
        if stage == "rigid":
            if self.transforms is None or self.labels is None:
                return None
            from estimator.rigid_part_estimator import apply_transforms
            _, R = apply_transforms(self.mu0, self.R0, self.labels,
                                    self.transforms, t)
            return R
        if stage in ("joint", "refined"):
            if self.scene is None or not self.q_traj:
                return None
            qt = [self.q_traj[k][t] for k in range(len(self.q_traj))]
            _, R = self.scene.transport_all(self.mu0, self.R0, self.labels, qt)
            return R
        return None

    def com_trajectory(self, k: int, stage: str = "rigid") -> list:
        """Section 9: c_k(t) = mean over the part's members at each frame."""
        if self.labels is None:
            return []
        mask = self.labels == k
        if not bool(mask.any()):
            return []
        out = []
        for t in range(self.num_frames):
            p = self.positions_at(stage, t)
            if p is None:
                break
            out.append([float(x) for x in p[mask].mean(dim=0)])
        return out

    def stage_available(self, stage: str) -> bool:
        return self.positions_at(stage, 0) is not None
