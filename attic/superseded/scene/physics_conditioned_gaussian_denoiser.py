"""Renderer-supervised residual critic of the current physical hypothesis.

This denoiser does not predict absolute motion from canonical.  It starts from
the discrete physical proposal the closed loop currently believes -- each
Gaussian either exactly canonical or exactly transported by one part's rigid
transform -- perturbs it, and learns the correction the renderer asks for.  The
final residual

    r_i = x_final - x_physical_proposal

is therefore a statement *about the hypothesis*, not about the motion:

    correct label and correct motion   ->  r ~ 0
    false moving assignment            ->  r ~ -d_k     (undo the motion)
    missed moving point                ->  r ~ +d_k     (apply the motion)
    wrong part, should be j not k       ->  r ~ d_j - d_k

which is exactly a label-transition critic, and is what
``utils/closed_loop_assignment.py`` turns into ``K+1`` energies.

The feature layout is built around that physical state rather than inherited
from the older canonical-motion denoiser, whose 14-D vector had no notion of a
current hypothesis to be right or wrong about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@torch.no_grad()
def build_foreground_knn(points: torch.Tensor, k: int = 16, chunk_size: int = 2048):
    """Canonical KNN over the foreground only, plus the nearest-neighbour scale.

    The scale is the median nearest-neighbour distance and is what every length
    in the feature vector is divided by, so the network sees displacements in
    units of local point spacing and transfers across scenes of different size.
    """
    xyz = torch.as_tensor(points).detach().float()
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("points must be [N_F,3]")
    n = int(xyz.shape[0])
    if n < 2:
        raise ValueError("KNN needs at least two foreground points")
    k = max(1, min(int(k), n - 1))
    rows, nearest = [], []
    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        distance = torch.cdist(xyz[start:end], xyz)
        distance[torch.arange(end - start, device=xyz.device),
                 torch.arange(start, end, device=xyz.device)] = float("inf")
        values, indices = torch.topk(distance, k, largest=False)
        rows.append(indices)
        nearest.append(values[:, 0])
    finite = torch.cat(nearest)
    finite = finite[torch.isfinite(finite) & (finite > 0)]
    scale = float(finite.median()) if finite.numel() else 1.0
    return torch.cat(rows, 0).long(), max(scale, 1.0e-8)


@dataclass
class PhysicsFeatureSpec:
    """Named blocks of the denoiser input, so the width is derived not declared.

    Every block is optional at construction time because not every run has depth
    or DINO evidence, and a feature that is silently zero is worse than a
    feature that is absent: the checkpoint records the exact layout it was
    trained with, and a mismatch is an error rather than a quiet reinterpretation
    of the wrong columns.
    """

    num_moving_parts: int
    include_full_responsibility: bool = True
    include_render_evidence: bool = True
    include_observation_confidence: bool = True

    #: (name, width) in the order they are concatenated.
    def blocks(self) -> list[tuple[str, int]]:
        items: list[tuple[str, int]] = [
            # 15.1 artificial residual relative to the physical proposal
            ("noise_residual", 3),
            # 15.2 physical proposal relative to canonical
            ("physical_displacement", 3),
            # 15.3 / 15.4 / 15.5 the two motion proposals and their disagreement
            ("fm_velocity", 3),
            ("killing_velocity", 3),
            ("fm_minus_killing", 3),
            # 15.6 assignment context
            ("static_probability", 1),
            ("assigned_probability", 1),
            ("max_moving_probability", 1),
            ("assignment_entropy", 1),
        ]
        if self.include_full_responsibility:
            items.append(("responsibility_vector", int(self.num_moving_parts) + 1))
        if self.include_render_evidence:
            items += [
                ("rgb_residual", 1),
                ("depth_residual", 1),
                ("render_position_gradient", 3),
                ("raw_support", 1),
                ("visible_mass", 1),
                ("valid_visible_mass", 1),
                ("visibility_ratio", 1),
                ("valid_fraction", 1),
            ]
        if self.include_observation_confidence:
            items += [
                ("fm_confidence", 1),
                ("flow_confidence", 1),
                ("condition_score", 1),
                ("reprojection_rms", 1),
                ("valid_camera_fraction", 1),
            ]
        return items

    @property
    def input_dim(self) -> int:
        return sum(width for _, width in self.blocks())

    def offsets(self) -> dict[str, tuple[int, int]]:
        result, cursor = {}, 0
        for name, width in self.blocks():
            result[name] = (cursor, cursor + width)
            cursor += width
        return result

    def to_dict(self) -> dict:
        return {
            "num_moving_parts": int(self.num_moving_parts),
            "include_full_responsibility": bool(self.include_full_responsibility),
            "include_render_evidence": bool(self.include_render_evidence),
            "include_observation_confidence": bool(self.include_observation_confidence),
            "input_dim": int(self.input_dim),
            "blocks": self.blocks(),
        }


def assemble_features(
    spec: PhysicsFeatureSpec,
    *,
    current_positions: torch.Tensor,
    physical_positions: torch.Tensor,
    canonical_fg: torch.Tensor,
    fm_velocity: torch.Tensor,
    killing_velocity: torch.Tensor,
    responsibilities: torch.Tensor,
    assignment_entropy: torch.Tensor,
    hard_labels: torch.Tensor,
    neighbour_scale: float,
    render_evidence: dict | None = None,
    observation_confidence: dict | None = None,
) -> torch.Tensor:
    """Concatenate the named blocks into the denoiser input, in spec order.

    All lengths are divided by ``neighbour_scale`` -- the median foreground
    nearest-neighbour distance -- so the network reads displacements as
    multiples of local point spacing.  Missing evidence blocks are filled with
    zeros only when the spec says they are present but a particular run has none
    for that Gaussian; a block the spec excludes is not emitted at all.
    """
    device = current_positions.device
    dtype = current_positions.dtype
    n = int(current_positions.shape[0])
    scale = max(float(neighbour_scale), 1.0e-8)

    def column(source: dict | None, key: str, width: int) -> torch.Tensor:
        if source is None or key not in source or source[key] is None:
            return torch.zeros((n, width), device=device, dtype=dtype)
        value = torch.as_tensor(source[key], device=device, dtype=dtype).reshape(n, -1)
        if value.shape[-1] != width:
            raise ValueError(f"{key} must have width {width}, got {value.shape[-1]}")
        return value

    moving = responsibilities[:, 1:]
    assigned = responsibilities.gather(1, hard_labels.reshape(-1, 1).clamp_min(0))
    pieces: dict[str, torch.Tensor] = {
        "noise_residual": (current_positions - physical_positions) / scale,
        "physical_displacement": (physical_positions - canonical_fg) / scale,
        "fm_velocity": fm_velocity,
        "killing_velocity": killing_velocity,
        "fm_minus_killing": fm_velocity - killing_velocity,
        "static_probability": responsibilities[:, :1],
        "assigned_probability": assigned,
        "max_moving_probability": (
            moving.amax(-1, keepdim=True)
            if moving.shape[-1]
            else torch.zeros((n, 1), device=device, dtype=dtype)
        ),
        "assignment_entropy": assignment_entropy.reshape(n, 1),
        "responsibility_vector": responsibilities,
    }
    ordered = []
    for name, width in spec.blocks():
        if name in pieces:
            ordered.append(pieces[name].to(device=device, dtype=dtype).reshape(n, width))
        elif name in ("rgb_residual", "depth_residual", "render_position_gradient",
                      "raw_support", "visible_mass", "valid_visible_mass",
                      "visibility_ratio", "valid_fraction"):
            ordered.append(column(render_evidence, name, width))
        else:
            ordered.append(column(observation_confidence, name, width))
    features = torch.cat(ordered, dim=-1)
    if features.shape[-1] != spec.input_dim:
        raise ValueError(
            f"assembled {features.shape[-1]} features, spec declares {spec.input_dim}"
        )
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class DenoiserConfig:
    hidden_dim: int = 128
    message_layers: int = 3
    knn: int = 16
    max_step_ratio: float = 1.0
    spec: PhysicsFeatureSpec | None = field(default=None)

    def to_dict(self) -> dict:
        return {
            "hidden_dim": int(self.hidden_dim),
            "message_layers": int(self.message_layers),
            "knn": int(self.knn),
            "max_step_ratio": float(self.max_step_ratio),
            "spec": None if self.spec is None else self.spec.to_dict(),
        }


class PhysicsConditionedGaussianDenoiser(nn.Module):
    """Graph network predicting a bounded positional correction per Gaussian.

    Message passing runs on the *canonical* foreground KNN, which is fixed for
    the whole run: neighbours are a property of the object, not of whichever
    hypothesis is currently being tested, so the graph does not change when the
    assignment does.

    The output is squashed and scaled by the local neighbour spacing, so a single
    denoising step can never teleport a Gaussian across the object -- the critic
    corrects a hypothesis, it does not re-derive the motion.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, message_layers: int = 3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.message_layers = max(1, int(message_layers))
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        # Separate self and neighbour projections: a residual that is coherent
        # across a neighbourhood (a wrong axis) and one that is isolated (a
        # wrong label) have to be distinguishable, and mean-pooling alone into a
        # shared trunk blurs them together.
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            for _ in range(self.message_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(self.hidden_dim) for _ in range(self.message_layers)
        )
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, features: torch.Tensor, knn_idx: torch.Tensor, step_scale) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must be [N_F,{self.input_dim}], got {tuple(features.shape)}")
        if knn_idx.ndim != 2 or knn_idx.shape[0] != features.shape[0]:
            raise ValueError("knn_idx must be [N_F,K] and match the feature rows")
        hidden = self.encoder(features)
        neighbours = knn_idx.to(device=features.device, dtype=torch.long)
        for block, norm in zip(self.blocks, self.norms):
            gathered = hidden[neighbours]
            message = torch.cat((gathered.mean(1), gathered.amax(1)), dim=-1)
            hidden = norm(hidden + block(torch.cat((hidden, message), dim=-1)))
        scale = torch.as_tensor(step_scale, device=features.device, dtype=features.dtype)
        # Zero-initialised head plus tanh: the first step of an untrained critic
        # is exactly no correction, so training starts from the hypothesis
        # itself rather than from a random displacement of it.
        return torch.tanh(self.out(hidden)) * scale

    def checkpoint_payload(self, config: DenoiserConfig) -> dict:
        """Record the architecture the weights actually have.

        The config carries the things the module cannot know -- the feature
        spec, the KNN size, the step ratio -- but its width and depth are only
        a request.  Writing those instead of the module's own would produce a
        checkpoint that fails to load against its own weights whenever the two
        disagree, which is exactly what happens when a caller passes a default
        config to a model it built with different sizes.
        """
        return {
            "schema_version": 1,
            "pipeline": "closed_loop_diffusion_fm_killing",
            "denoiser_state_dict": self.state_dict(),
            "denoiser_config": {
                **config.to_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "message_layers": self.message_layers,
            },
        }


def load_denoiser(payload: dict) -> tuple[PhysicsConditionedGaussianDenoiser, DenoiserConfig]:
    """Rebuild a denoiser from its own checkpoint, refusing legacy formats.

    The specification is explicit that old denoiser checkpoints are not carried
    forward.  A payload without this pipeline's name is rejected rather than
    partially loaded, because the feature layout differs and the columns would
    be silently reinterpreted.
    """
    if payload.get("pipeline") != "closed_loop_diffusion_fm_killing":
        raise ValueError(
            "this denoiser only loads closed-loop checkpoints; legacy motion "
            "denoiser checkpoints have a different feature layout and are not "
            "convertible"
        )
    config = payload["denoiser_config"]
    spec_dict = config.get("spec")
    spec = (
        None
        if spec_dict is None
        else PhysicsFeatureSpec(
            num_moving_parts=spec_dict["num_moving_parts"],
            include_full_responsibility=spec_dict["include_full_responsibility"],
            include_render_evidence=spec_dict["include_render_evidence"],
            include_observation_confidence=spec_dict["include_observation_confidence"],
        )
    )
    model = PhysicsConditionedGaussianDenoiser(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        message_layers=int(config["message_layers"]),
    )
    model.load_state_dict(payload["denoiser_state_dict"])
    return model, DenoiserConfig(
        hidden_dim=int(config["hidden_dim"]),
        message_layers=int(config["message_layers"]),
        knn=int(config["knn"]),
        max_step_ratio=float(config["max_step_ratio"]),
        spec=spec,
    )
