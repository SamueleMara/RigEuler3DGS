"""Shared static/dynamic prior container and metrics.

The estimators in this file stop at per-Gaussian prior probabilities.  They do
not call the rigid Killing solver; that separation is intentional so denoising,
endpoint transport, and GT upper bounds can be compared fairly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class DynamicPrior:
    probability: torch.Tensor
    raw_score: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor
    method: str
    gaussian_ids: torch.Tensor
    metadata: dict[str, Any]

    def validate(self) -> "DynamicPrior":
        self.probability = torch.as_tensor(self.probability).detach().cpu().float().reshape(-1)
        self.raw_score = torch.as_tensor(self.raw_score).detach().cpu().float().reshape(-1)
        self.valid = torch.as_tensor(self.valid).detach().cpu().bool().reshape(-1)
        self.confidence = torch.as_tensor(self.confidence).detach().cpu().float().reshape(-1)
        self.gaussian_ids = torch.as_tensor(self.gaussian_ids).detach().cpu().long().reshape(-1)
        n = int(self.probability.numel())
        if self.raw_score.numel() != n or self.valid.numel() != n or self.confidence.numel() != n:
            raise ValueError("DynamicPrior probability/raw/valid/confidence lengths differ")
        if self.gaussian_ids.numel() != n:
            raise ValueError("DynamicPrior gaussian_ids length differs from probability length")
        if not bool(torch.isfinite(self.probability).all()):
            raise ValueError("DynamicPrior probability contains non-finite values")
        if not bool(torch.isfinite(self.raw_score).all()):
            raise ValueError("DynamicPrior raw_score contains non-finite values")
        if not bool(torch.isfinite(self.confidence).all()):
            raise ValueError("DynamicPrior confidence contains non-finite values")
        self.probability = self.probability.clamp(0.0, 1.0)
        self.confidence = self.confidence.clamp(0.0, 1.0)
        self.metadata = dict(self.metadata or {})
        self.metadata.setdefault("ground_truth_used_as_algorithm_input", False)
        self.method = str(self.method)
        return self

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1,
            "probability": self.probability,
            "raw_score": self.raw_score,
            "valid": self.valid,
            "confidence": self.confidence,
            "method": self.method,
            "gaussian_ids": self.gaussian_ids,
            "metadata": self.metadata,
        }


def save_dynamic_prior(prior: DynamicPrior, path: str | Path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prior.to_payload(), path)
    return path


def load_dynamic_prior(path: str | Path) -> DynamicPrior:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if isinstance(payload, DynamicPrior):
        return payload.validate()
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported dynamic prior payload type: {type(payload).__name__}")
    return DynamicPrior(
        probability=payload["probability"],
        raw_score=payload.get("raw_score", payload["probability"]),
        valid=payload.get("valid", torch.ones_like(torch.as_tensor(payload["probability"]), dtype=torch.bool)),
        confidence=payload.get("confidence", torch.ones_like(torch.as_tensor(payload["probability"]), dtype=torch.float32)),
        method=payload.get("method", "unknown"),
        gaussian_ids=payload["gaussian_ids"],
        metadata=payload.get("metadata", {}),
    ).validate()


def make_all_foreground_prior(gaussian_ids: torch.Tensor) -> DynamicPrior:
    ids = torch.as_tensor(gaussian_ids).detach().cpu().long().reshape(-1)
    one = torch.ones(ids.numel(), dtype=torch.float32)
    return DynamicPrior(
        probability=one.clone(),
        raw_score=one.clone(),
        valid=torch.ones(ids.numel(), dtype=torch.bool),
        confidence=one.clone(),
        method="all_fg",
        gaussian_ids=ids,
        metadata={"description": "all semantic-foreground Gaussians treated as dynamic"},
    ).validate()


def make_all_gaussians_prior(num_gaussians: int) -> DynamicPrior:
    ids = torch.arange(int(num_gaussians), dtype=torch.long)
    one = torch.ones(ids.numel(), dtype=torch.float32)
    return DynamicPrior(
        probability=one.clone(),
        raw_score=one.clone(),
        valid=torch.ones(ids.numel(), dtype=torch.bool),
        confidence=one.clone(),
        method="baseline",
        gaussian_ids=ids,
        metadata={
            "description": "baseline: every canonical Gaussian is movable; no learned static/dynamic prior",
            "ground_truth_used_as_algorithm_input": False,
        },
    ).validate()


def make_gt_prior(gaussian_ids: torch.Tensor, full_gt_dynamic_mask: torch.Tensor) -> DynamicPrior:
    ids = torch.as_tensor(gaussian_ids).detach().cpu().long().reshape(-1)
    gt = torch.as_tensor(full_gt_dynamic_mask).detach().cpu().bool().reshape(-1)
    if ids.numel() and int(ids.max()) >= int(gt.numel()):
        raise ValueError("GT mask is shorter than gaussian_ids")
    probability = gt.index_select(0, ids).float()
    return DynamicPrior(
        probability=probability,
        raw_score=probability.clone(),
        valid=torch.ones_like(probability, dtype=torch.bool),
        confidence=torch.ones_like(probability),
        method="gt",
        gaussian_ids=ids,
        metadata={
            "role": "evaluation_upper_bound_only",
            "ground_truth_used_as_algorithm_input": False,
        },
    ).validate()


def _safe_float(value: torch.Tensor | float) -> float:
    value_t = torch.as_tensor(value).detach().cpu().float()
    if value_t.numel() == 0 or not bool(torch.isfinite(value_t).all()):
        return float("nan")
    return float(value_t.reshape(-1)[0])


def _binary_auc(probability: torch.Tensor, truth: torch.Tensor) -> float:
    p = probability.detach().cpu().float().reshape(-1)
    y = truth.detach().cpu().bool().reshape(-1)
    pos = int(y.sum())
    neg = int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    sorted_p, order = torch.sort(p, descending=False)
    sorted_y = y[order]
    ranks = torch.empty(p.numel(), dtype=torch.float32)
    start = 0
    while start < p.numel():
        end = start + 1
        while end < p.numel() and float(sorted_p[end]) == float(sorted_p[start]):
            end += 1
        ranks[start:end] = 0.5 * float((start + 1) + end)
        start = end
    pos_rank_sum = ranks[sorted_y].sum()
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / max(1, pos * neg))


def _pr_auc(probability: torch.Tensor, truth: torch.Tensor) -> float:
    p = probability.detach().cpu().float().reshape(-1)
    y = truth.detach().cpu().bool().reshape(-1)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = torch.argsort(p, descending=True)
    y = y[order].float()
    tp = torch.cumsum(y, 0)
    fp = torch.cumsum(1.0 - y, 0)
    recall = tp / max(1, pos)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = torch.cat((torch.zeros(1), recall))
    precision = torch.cat((torch.ones(1), precision))
    return float(torch.trapz(precision, recall).clamp(0.0, 1.0))


def _ece(probability: torch.Tensor, truth: torch.Tensor, bins: int = 10) -> float:
    p = probability.detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
    y = truth.detach().cpu().float().reshape(-1)
    if p.numel() == 0:
        return float("nan")
    edges = torch.linspace(0.0, 1.0, int(bins) + 1)
    total = torch.tensor(0.0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if float(hi) < 1.0 else p <= hi)
        if not bool(mask.any()):
            continue
        total += mask.float().mean() * torch.abs(p[mask].mean() - y[mask].mean())
    return float(total)


def prior_metrics(
    prior: DynamicPrior,
    full_gt_dynamic_mask: torch.Tensor | None,
    *,
    threshold: float = 0.5,
    high_confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    prior.validate()
    metrics: dict[str, Any] = {
        "method": prior.method,
        "num_gaussians": int(prior.probability.numel()),
        "valid_fraction": _safe_float(prior.valid.float().mean()) if prior.valid.numel() else float("nan"),
        "mean_probability": _safe_float(prior.probability.mean()) if prior.probability.numel() else float("nan"),
        "dynamic_mass_fraction": _safe_float(prior.probability.mean()) if prior.probability.numel() else float("nan"),
        "mean_confidence": _safe_float(prior.confidence.mean()) if prior.confidence.numel() else float("nan"),
    }
    if full_gt_dynamic_mask is None:
        return metrics
    full_gt = torch.as_tensor(full_gt_dynamic_mask).detach().cpu().bool().reshape(-1)
    ids = prior.gaussian_ids
    if ids.numel() and int(ids.max()) >= int(full_gt.numel()):
        metrics["gt_error"] = "GT mask shorter than prior gaussian_ids"
        return metrics
    truth = full_gt.index_select(0, ids)

    def score_subset(name: str, mask: torch.Tensor) -> dict[str, float | int]:
        mask = mask.bool() & prior.valid.bool()
        if int(mask.sum()) == 0:
            return {f"{name}_count": 0}
        p = prior.probability[mask]
        y = truth[mask]
        pred = p >= float(threshold)
        tp = float((pred & y).sum())
        fp = float((pred & ~y).sum())
        fn = float((~pred & y).sum())
        tn = float((~pred & ~y).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
        iou = tp / max(1.0, tp + fp + fn)
        return {
            f"{name}_count": int(mask.sum()),
            f"{name}_precision": precision,
            f"{name}_recall": recall,
            f"{name}_f1": f1,
            f"{name}_iou": iou,
            f"{name}_accuracy": (tp + tn) / max(1.0, tp + fp + fn + tn),
            f"{name}_roc_auc": _binary_auc(p, y),
            f"{name}_pr_auc": _pr_auc(p, y),
            f"{name}_brier": float((p - y.float()).square().mean()),
            f"{name}_ece": _ece(p, y),
        }

    metrics.update(score_subset("all", torch.ones_like(truth, dtype=torch.bool)))
    metrics.update(score_subset("high_confidence", prior.confidence >= float(high_confidence_threshold)))
    return metrics
