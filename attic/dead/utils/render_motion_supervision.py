"""Foreground-only render supervision bundle for predicted motion states."""

from __future__ import annotations

from typing import Any

import torch

from utils.semantic_render_likelihood import (
    compute_foreground_depth_likelihood,
    compute_foreground_flow_likelihood,
    compute_foreground_rgb_likelihood,
)


def compute_foreground_render_motion_evidence(
    *,
    observed_rgb: torch.Tensor,
    rendered_rgb: torch.Tensor,
    semantic_foreground_mask: torch.Tensor,
    observed_depth: torch.Tensor | None = None,
    rendered_depth: torch.Tensor | None = None,
    observed_flow: torch.Tensor | None = None,
    rendered_flow: torch.Tensor | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "rgb": compute_foreground_rgb_likelihood(observed_rgb, rendered_rgb, semantic_foreground_mask)
    }
    if observed_depth is not None and rendered_depth is not None:
        evidence["depth"] = compute_foreground_depth_likelihood(observed_depth, rendered_depth, semantic_foreground_mask)
    if observed_flow is not None and rendered_flow is not None:
        evidence["flow"] = compute_foreground_flow_likelihood(observed_flow, rendered_flow, semantic_foreground_mask)
    evidence["total_log_likelihood"] = sum(v["log_likelihood"] for v in evidence.values() if isinstance(v, dict) and "log_likelihood" in v)
    return evidence
