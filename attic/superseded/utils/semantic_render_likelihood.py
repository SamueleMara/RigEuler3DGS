"""Foreground-only render likelihoods for closed-loop dynamic evidence."""

from __future__ import annotations

from typing import Callable, Any

import torch

EPS = 1.0e-8


def _require_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mask is None:
        raise ValueError("semantic_foreground_mask is required")
    m = torch.as_tensor(mask, device=target.device).bool()
    while m.ndim < target.ndim:
        m = m.unsqueeze(0)
    return m.expand_as(target)


def render_predicted_dynamic_state(render_fn: Callable[..., dict[str, Any]], *args, **kwargs) -> dict[str, Any]:
    return render_fn(*args, **kwargs)


def _student_loglik(residual: torch.Tensor, scale: float) -> torch.Tensor:
    s = max(float(scale), 1.0e-6)
    return -2.5 * torch.log1p(residual.square() / (4.0 * s * s))


def compute_foreground_rgb_likelihood(observed_rgb: torch.Tensor, rendered_rgb: torch.Tensor, semantic_foreground_mask: torch.Tensor, *, scale: float | None = None) -> dict[str, torch.Tensor | float]:
    obs = torch.as_tensor(observed_rgb).float()
    pred = torch.as_tensor(rendered_rgb, device=obs.device).float()
    mask = _require_mask(semantic_foreground_mask, obs)
    residual = (obs - pred)[mask]
    s = float((residual - residual.median()).abs().median() * 1.4826) if scale is None and residual.numel() else float(scale or 1.0)
    ll = _student_loglik(residual, s)
    return {"log_likelihood": ll.sum(), "mean_abs_residual": float(residual.abs().mean()) if residual.numel() else 0.0, "pixels": int(mask[0].sum()) if mask.ndim == 3 else int(mask.sum())}


def compute_foreground_depth_likelihood(observed_depth: torch.Tensor, rendered_depth: torch.Tensor, semantic_foreground_mask: torch.Tensor, *, scale: float | None = None) -> dict[str, torch.Tensor | float]:
    obs = torch.as_tensor(observed_depth).float()
    pred = torch.as_tensor(rendered_depth, device=obs.device).float()
    mask = _require_mask(semantic_foreground_mask, obs)
    valid = mask & torch.isfinite(obs) & torch.isfinite(pred) & (obs > 0)
    residual = (obs - pred)[valid]
    s = float((residual - residual.median()).abs().median() * 1.4826) if scale is None and residual.numel() else float(scale or 1.0)
    ll = _student_loglik(residual, s)
    return {"log_likelihood": ll.sum(), "mean_abs_residual": float(residual.abs().mean()) if residual.numel() else 0.0, "pixels": int(valid.sum())}


def compute_foreground_flow_likelihood(observed_flow: torch.Tensor, rendered_flow: torch.Tensor, semantic_foreground_mask: torch.Tensor, *, scale: float | None = None) -> dict[str, torch.Tensor | float]:
    obs = torch.as_tensor(observed_flow).float()
    pred = torch.as_tensor(rendered_flow, device=obs.device).float()
    mask = _require_mask(semantic_foreground_mask, obs)
    residual = torch.linalg.norm((obs - pred), dim=0 if obs.ndim == 3 else -1)
    residual = residual[mask[0] if mask.ndim == 3 else mask]
    s = float((residual - residual.median()).abs().median() * 1.4826) if scale is None and residual.numel() else float(scale or 1.0)
    ll = _student_loglik(residual, s)
    return {"log_likelihood": ll.sum(), "median_flow_error": float(residual.median()) if residual.numel() else 0.0, "pixels": int(residual.numel())}
