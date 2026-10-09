"""Independent forward/backward endpoint 3D flow matching wrappers."""

from __future__ import annotations

import torch

from utils.bidirectional_fm import FlowTrainConfig, FlowResult, integrate_flow, train_flow_direction
from utils.fm_cycle_consistency import cycle_consistency


def train_forward_endpoint_fm(source_xyz: torch.Tensor, target_xyz: torch.Tensor, cfg: FlowTrainConfig, *, device: str | torch.device = "cuda") -> FlowResult:
    return train_flow_direction(source_xyz, target_xyz, cfg, direction_name="endpoint_forward", device=device)


def train_backward_endpoint_fm(source_xyz: torch.Tensor, target_xyz: torch.Tensor, cfg: FlowTrainConfig, *, device: str | torch.device = "cuda") -> FlowResult:
    return train_flow_direction(target_xyz, source_xyz, cfg, direction_name="endpoint_backward", device=device)


def integrate_forward_endpoint_fm(result: FlowResult, points: torch.Tensor | None = None, *, cfg: FlowTrainConfig, device: str | torch.device = "cuda") -> torch.Tensor:
    return integrate_flow(result.model, result.endpoint - result.displacement if points is None else points, cfg, device=device)


def integrate_backward_endpoint_fm(result: FlowResult, points: torch.Tensor | None = None, *, cfg: FlowTrainConfig, device: str | torch.device = "cuda") -> torch.Tensor:
    return integrate_flow(result.model, result.endpoint - result.displacement if points is None else points, cfg, device=device)


def compute_endpoint_cycle_consistency(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    forward: FlowResult,
    backward: FlowResult,
    forward_cfg: FlowTrainConfig,
    backward_cfg: FlowTrainConfig,
    *,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    return cycle_consistency(
        source_xyz=source_xyz,
        target_xyz=target_xyz,
        forward_endpoint=forward.endpoint,
        backward_endpoint=backward.endpoint,
        forward_model=forward.model,
        backward_model=backward.model,
        forward_cfg=forward_cfg,
        backward_cfg=backward_cfg,
        source_scale=forward.local_scale,
        target_scale=backward.local_scale,
        device=device,
    )
