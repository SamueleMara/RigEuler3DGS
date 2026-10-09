"""Adapter from DynamicPrior to the existing dynamic/Killing initialization path."""

from __future__ import annotations

import torch

from utils.dynamic_prior import DynamicPrior


def build_killing_dynamic_weights(
    prior: DynamicPrior,
    observation_confidence: torch.Tensor,
) -> torch.Tensor:
    prior.validate()
    confidence = torch.as_tensor(observation_confidence).detach().cpu().float().reshape(-1)
    if confidence.numel() != prior.probability.numel():
        raise ValueError("observation_confidence must match prior length")
    return prior.probability * prior.confidence * confidence * prior.valid.float()


def dynamic_prior_to_motion_state(
    prior: DynamicPrior,
    canonical_xyz: torch.Tensor,
    *,
    total_gaussians: int | None = None,
    use_soft_weights: bool = True,
    hard_threshold: float = 0.5,
    hard_requires_confidence: bool = True,
) -> dict[str, torch.Tensor | dict]:
    """Build the full-scene state expected by GaussianModel.init_dynamic_motion.

    Low-confidence or invalid prior entries are represented as UNKNOWN through
    ``unknown_score``.  They are not silently converted to static points.
    """
    prior.validate()
    xyz = torch.as_tensor(canonical_xyz).detach().cpu().float()
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("canonical_xyz must be [N,3]")
    n = int(total_gaussians if total_gaussians is not None else xyz.shape[0])
    if n != int(xyz.shape[0]):
        raise ValueError("total_gaussians must match canonical_xyz length")
    ids = prior.gaussian_ids.long()
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= n):
        raise ValueError("DynamicPrior gaussian_ids are outside canonical_xyz")

    full_probability = torch.full((n,), 0.5, dtype=torch.float32)
    full_raw = torch.zeros(n, dtype=torch.float32)
    full_confidence = torch.zeros(n, dtype=torch.float32)
    full_valid = torch.zeros(n, dtype=torch.bool)

    probability = prior.probability if bool(use_soft_weights) else (prior.probability >= float(hard_threshold)).float()
    full_probability.index_copy_(0, ids, probability.float().clamp(0.0, 1.0))
    full_raw.index_copy_(0, ids, prior.raw_score.float())
    full_confidence.index_copy_(0, ids, (prior.confidence * prior.valid.float()).float().clamp(0.0, 1.0))
    full_valid.index_copy_(0, ids, prior.valid.bool())

    unknown = torch.ones(n, dtype=torch.float32)
    unknown.index_copy_(0, ids, (1.0 - full_confidence.index_select(0, ids)).clamp(0.0, 1.0))
    dynamic_score = full_probability.clamp(0.0, 1.0)
    static_score = (1.0 - full_probability).clamp(0.0, 1.0)
    hard_valid = full_valid
    if bool(hard_requires_confidence):
        hard_valid = hard_valid & (full_confidence > 0.05)
    hard = hard_valid & (full_probability >= float(hard_threshold))
    return {
        "canonical_xyz": xyz,
        "motion_evidence": full_raw,
        "motion_evidence_normalized": dynamic_score,
        "static_consistency_score": static_score,
        "unknown_score": unknown,
        "prior_confidence": full_confidence,
        "prior_valid": full_valid,
        "initial_dynamic_mask": hard,
        "dynamic_prior_metadata": {
            "method": prior.method,
            "use_soft_weights": bool(use_soft_weights),
            "hard_threshold": float(hard_threshold),
            "hard_requires_confidence": bool(hard_requires_confidence),
            "num_prior_gaussians": int(ids.numel()),
        },
    }
