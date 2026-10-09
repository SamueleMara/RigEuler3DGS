"""Robust empirical static-flow null calibration."""

from __future__ import annotations

import torch


def collect_known_static_flow_samples(
    flow: torch.Tensor,
    confidence: torch.Tensor,
    known_static_mask: torch.Tensor,
    *,
    min_confidence: float = 0.05,
) -> torch.Tensor:
    value = torch.as_tensor(flow).float()
    if value.ndim != 3 or value.shape[0] != 2:
        raise ValueError("flow must be [2,H,W]")
    conf = torch.as_tensor(confidence).float().squeeze()
    mask = torch.as_tensor(known_static_mask).bool().squeeze()
    valid = mask & (conf >= float(min_confidence)) & torch.isfinite(value).all(dim=0)
    return value[:, valid].T.contiguous()


def fit_robust_flow_null(
    samples_or_flow: torch.Tensor,
    static_mask: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    *,
    mad_floor: float = 0.15,
    min_samples: int = 64,
) -> dict[str, torch.Tensor | int | bool]:
    """Fit componentwise median/MAD; return an isotropic robust scale too."""
    value = torch.as_tensor(samples_or_flow).float()
    if value.ndim == 3:
        if static_mask is None:
            raise ValueError("static_mask is required for image-shaped flow")
        if confidence is None:
            confidence = torch.ones_like(torch.as_tensor(static_mask), dtype=torch.float32)
        value = collect_known_static_flow_samples(value, confidence, static_mask)
    if value.ndim != 2 or value.shape[-1] != 2:
        raise ValueError("samples must have shape [N,2]")
    finite = value[torch.isfinite(value).all(dim=-1)]
    enough = finite.shape[0] >= int(min_samples)
    if finite.numel() == 0:
        median = torch.zeros(2, dtype=torch.float32, device=value.device)
        mad = torch.full((2,), float(mad_floor), dtype=torch.float32, device=value.device)
    else:
        median = finite.median(dim=0).values
        mad = (finite - median).abs().median(dim=0).values * 1.4826
        mad = mad.clamp_min(float(mad_floor))
    return {
        "median": median,
        "mad": mad,
        "scale": torch.sqrt(mad.square().mean()).clamp_min(float(mad_floor)),
        "sample_count": int(finite.shape[0]),
        "reliable": bool(enough),
    }


def evaluate_static_flow_residual(
    flow: torch.Tensor,
    calibration: dict,
) -> dict[str, torch.Tensor]:
    value = torch.as_tensor(flow).float()
    if value.ndim < 2 or value.shape[-3 if value.ndim >= 3 else -1] not in (2,):
        if value.ndim != 2 or value.shape[-1] != 2:
            raise ValueError("flow must be [2,H,W] or [N,2]")
    median = torch.as_tensor(calibration["median"], device=value.device, dtype=value.dtype)
    mad = torch.as_tensor(calibration["mad"], device=value.device, dtype=value.dtype).clamp_min(1.0e-8)
    if value.ndim == 3:
        corrected = value - median[:, None, None]
        normalized = torch.sqrt(((corrected / mad[:, None, None]) ** 2).sum(dim=0))
    else:
        corrected = value - median
        normalized = torch.sqrt(((corrected / mad) ** 2).sum(dim=-1))
    return {"corrected_flow": corrected, "normalized_residual": normalized}


def compute_camera_interval_reliability(
    calibration: dict,
    *,
    reference_scale: float = 1.0,
    min_samples: int = 64,
) -> float:
    count = int(calibration.get("sample_count", 0))
    scale = float(torch.as_tensor(calibration.get("scale", 1.0)))
    sample_factor = min(1.0, count / max(1, int(min_samples)))
    noise_factor = 1.0 / (1.0 + scale / max(float(reference_scale), 1.0e-8))
    return float(sample_factor * noise_factor)

