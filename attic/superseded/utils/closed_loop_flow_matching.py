"""A temporal flow-matching velocity field for the closed loop.

The existing ``utils/endpoint_flow_matching`` transports canonical points to a
single endpoint frame, which is enough to build a static/dynamic prior but not
enough here: the closed loop compares FM against the Killing field *per
interval*, so it needs ``v(x, t)`` rather than one displacement.

This is a deliberately small field.  FM's job in this pipeline is to be a
high-recall dense transport proposer -- it does not decide membership and it
does not have to be right, it has to be informative enough that the consensus
and the assignment energies have something to disagree with.  Keeping it small
also keeps it honest: a network with enough capacity to memorise the Killing
field it is regularised toward would collude with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TemporalFlowConfig:
    hidden_dim: int = 128
    layers: int = 4
    time_frequencies: int = 4
    learning_rate: float = 1.0e-3
    iterations: int = 200
    batch_size: int = 4096
    seed: int = 20_260_910


class TemporalFlowField(nn.Module):
    """``(x, t) -> delta_v``, with a Fourier time embedding.

    The time embedding matters because the interval speeds of an articulated
    object are not smooth in general -- a door can stop and restart -- and a
    field linear in ``t`` would average that away into a constant velocity that
    matches no interval.
    """

    def __init__(self, config: TemporalFlowConfig):
        super().__init__()
        self.config = config
        frequencies = max(1, int(config.time_frequencies))
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(frequencies, dtype=torch.float32)
        )
        input_dim = 3 + 2 * frequencies
        layers: list[nn.Module] = [nn.Linear(input_dim, config.hidden_dim), nn.SiLU()]
        for _ in range(max(1, int(config.layers)) - 1):
            layers += [nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(config.hidden_dim, 3))
        self.network = nn.Sequential(*layers)
        # Start at exactly zero residual.  With legacy FM enabled the current
        # field is therefore ``fm_base + 0``, not a freshly zero velocity field.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, points: torch.Tensor, normalized_time: torch.Tensor) -> torch.Tensor:
        time = normalized_time.reshape(-1, 1) * self.frequencies[None]
        embedding = torch.cat((torch.sin(time), torch.cos(time)), dim=-1)
        if embedding.shape[0] == 1:
            embedding = embedding.expand(points.shape[0], -1)
        return self.network(torch.cat((points, embedding), dim=-1))

    @torch.no_grad()
    def evaluate_all_intervals(
        self,
        points: torch.Tensor,
        num_intervals: int,
        *,
        base_velocity: torch.Tensor | None = None,
        max_residual_step: float = 0.0,
    ) -> torch.Tensor:
        """``[T-1, N_F, 3]`` velocity at the midpoint of every interval."""
        out = []
        for interval in range(int(num_intervals)):
            fraction = (interval + 0.5) / max(1, int(num_intervals))
            time = torch.full((1,), fraction, device=points.device, dtype=points.dtype)
            residual = self(points, time)
            residual = _clamp_residual(residual, float(max_residual_step))
            if base_velocity is not None:
                residual = residual + base_velocity[interval].to(residual)
            out.append(residual)
        return torch.stack(out)


def _clamp_residual(residual: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return residual
    norm = residual.norm(dim=-1, keepdim=True)
    scale = (float(max_norm) / norm.clamp_min(1.0e-8)).clamp(max=1.0)
    return residual * scale


def train_temporal_flow_field(
    field: TemporalFlowField,
    *,
    points: torch.Tensor,
    target_velocity: torch.Tensor,
    target_confidence: torch.Tensor,
    killing_velocity: torch.Tensor | None = None,
    responsibilities: torch.Tensor | None = None,
    killing_valid=None,
    base_velocity: torch.Tensor | None = None,
    physical_velocity: torch.Tensor | None = None,
    physical_confidence: torch.Tensor | None = None,
    static_feedback_gate: torch.Tensor | None = None,
    max_residual_step: float = 0.0,
    feedback_weights=None,
    config: TemporalFlowConfig | None = None,
    log_every: int = 0,
) -> dict:
    """Fit the field to observed transport, regularised by the current physics.

    The base objective is confidence-weighted regression onto whatever transport
    evidence exists.  On top of it, spec 12's feedback pulls the field toward the
    current Killing motion where a point is confidently moving and toward zero
    where it is confidently static -- both detached, so FM learns from the
    physics without being able to reshape it.

    Keeping the base term active is what stops FM from becoming a copy of
    Killing: if it were removed the field would satisfy the feedback perfectly
    by reproducing the physics, and the FM-versus-Killing disagreement that the
    assignment energies read would be identically zero.
    """
    from utils.closed_loop_motion_losses import (
        FlowMatchingFeedbackWeights,
        flow_matching_feedback_loss,
    )

    config = config or field.config
    weights = feedback_weights or FlowMatchingFeedbackWeights()
    device = points.device
    intervals = int(target_velocity.shape[0])
    n = int(points.shape[0])
    optimizer = torch.optim.Adam(field.parameters(), lr=float(config.learning_rate))
    generator = torch.Generator().manual_seed(int(config.seed))
    history = []

    for step in range(max(1, int(config.iterations))):
        interval = int(torch.randint(0, intervals, (1,), generator=generator))
        size = min(int(config.batch_size), n)
        rows = torch.randperm(n, generator=generator)[:size].to(device)
        fraction = (interval + 0.5) / max(1, intervals)
        time = torch.full((1,), fraction, device=device, dtype=points.dtype)

        optimizer.zero_grad(set_to_none=True)
        residual = _clamp_residual(field(points[rows], time), float(max_residual_step))
        predicted = residual
        if base_velocity is not None:
            predicted = predicted + base_velocity[interval, rows].detach().to(predicted)
        confidence = target_confidence[interval, rows].detach().clamp_min(0.0)
        base = (
            confidence * (predicted - target_velocity[interval, rows].detach()).square().sum(-1)
        ).sum() / confidence.sum().clamp_min(1.0e-8)

        total = float(weights.base) * base
        feedback = None
        if killing_velocity is not None and responsibilities is not None:
            feedback = flow_matching_feedback_loss(
                predicted_velocity=predicted[None],
                killing_velocity=killing_velocity[:, interval, rows][:, None],
                responsibilities=responsibilities[rows],
                fm_confidence=confidence[None],
                killing_valid=killing_valid,
                physical_velocity=None if physical_velocity is None else physical_velocity[interval, rows][None],
                physical_confidence=None if physical_confidence is None else physical_confidence[interval, rows][None],
                static_feedback_gate=None if static_feedback_gate is None else static_feedback_gate[interval, rows][None],
                weights=weights,
            )
            total = total + feedback["total"]
        if not bool(torch.isfinite(total)):
            raise RuntimeError("flow-matching loss became non-finite")
        total.backward()
        optimizer.step()
        if log_every and (step % int(log_every) == 0 or step == int(config.iterations) - 1):
            history.append({
                "step": step,
                "total": float(total.detach()),
                "base": float(base.detach()),
                "killing": None if feedback is None else float(feedback["killing_term"].detach()),
                "physical_3d": None if feedback is None else float(feedback["physical_3d_term"].detach()),
                "static": None if feedback is None else float(feedback["static_term"].detach()),
            })
    return {"history": history}


@torch.no_grad()
def flow_confidence_from_observations(
    observed_confidence: torch.Tensor, observed_valid: torch.Tensor
) -> torch.Tensor:
    """FM confidence inherited from whatever the cameras actually supported.

    An FM velocity in a region no camera constrained is a guess, and letting it
    enter the consensus at full weight would let the network's prior override
    the measurement everywhere the measurement is missing.
    """
    return torch.where(
        observed_valid, observed_confidence.clamp_min(0.0), torch.zeros_like(observed_confidence)
    )
