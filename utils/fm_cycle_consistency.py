"""Cycle-confidence measurements for independently trained FM fields."""

from __future__ import annotations

from pathlib import Path

import torch

from utils.bidirectional_fm import FlowTrainConfig, integrate_flow


@torch.no_grad()
def cycle_consistency(
    *,
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    forward_endpoint: torch.Tensor,
    backward_endpoint: torch.Tensor,
    forward_model,
    backward_model,
    forward_cfg: FlowTrainConfig,
    backward_cfg: FlowTrainConfig,
    source_scale: torch.Tensor,
    target_scale: torch.Tensor,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    source = torch.as_tensor(source_xyz).detach().float().cpu()
    target = torch.as_tensor(target_xyz).detach().float().cpu()
    fwd_endpoint = torch.as_tensor(forward_endpoint).detach().float().cpu()
    bwd_endpoint = torch.as_tensor(backward_endpoint).detach().float().cpu()
    back_to_source = integrate_flow(backward_model, fwd_endpoint, backward_cfg, device=device)
    forward_to_target = integrate_flow(forward_model, bwd_endpoint, forward_cfg, device=device)
    forward_error = torch.linalg.norm(back_to_source - source, dim=-1)
    backward_error = torch.linalg.norm(forward_to_target - target, dim=-1)
    source_scale = torch.as_tensor(source_scale).detach().cpu().float().clamp_min(1.0e-8)
    target_scale = torch.as_tensor(target_scale).detach().cpu().float().clamp_min(1.0e-8)
    forward_conf = torch.exp(-0.5 * (forward_error / source_scale).square()).clamp(0.0, 1.0)
    backward_conf = torch.exp(-0.5 * (backward_error / target_scale).square()).clamp(0.0, 1.0)
    return {
        "forward_cycle_error": forward_error,
        "forward_cycle_confidence": forward_conf,
        "backward_cycle_error": backward_error,
        "backward_cycle_confidence": backward_conf,
    }


def save_cycle_result(result: dict[str, torch.Tensor], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        torch.save(torch.as_tensor(value).detach().cpu(), output / f"{name}.pt")
