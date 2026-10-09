"""Persistent static/moving material assignments."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class MaterialAssignmentField(nn.Module):
    """Persistent static/moving-part assignment shared across the whole sequence.

    Initialization preserves the *continuous* motion-evidence score whenever it is
    available.  The historical hard dynamic mask is retained only for diagnostics
    and for asymmetric prior weighting; it is not the value used to initialize the
    probabilities.
    """

    STATE_VERSION = 3

    def __init__(
        self,
        initial_dynamic_mask: torch.Tensor,
        num_moving_parts: int,
        *,
        initial_dynamic_score: torch.Tensor | None = None,
        initial_static_score: torch.Tensor | None = None,
        initial_unknown_score: torch.Tensor | None = None,
        dynamic_probability: float = 0.95,
        static_dynamic_probability: float = 0.10,
        evidence_temperature: float = 0.08,
        evidence_probability_floor: float = 0.03,
        evidence_probability_ceiling: float = 0.97,
    ):
        super().__init__()
        hard = initial_dynamic_mask.detach().reshape(-1).bool()
        self.num_points = int(hard.numel())
        self.num_moving_parts = max(1, int(num_moving_parts))
        self.num_experts = self.num_moving_parts + 1

        if initial_dynamic_score is not None:
            score = (
                initial_dynamic_score.detach()
                .reshape(-1)
                .to(device=hard.device, dtype=torch.float32)
            )
            if score.numel() != self.num_points:
                raise ValueError("initial_dynamic_score must match the Gaussian count")
            finite = torch.isfinite(score)
            score = torch.where(finite, score, torch.zeros_like(score)).clamp(0.0, 1.0)
            if initial_static_score is not None:
                static_score = (
                    initial_static_score.detach()
                    .reshape(-1)
                    .to(device=hard.device, dtype=torch.float32)
                )
                if static_score.numel() != self.num_points:
                    raise ValueError(
                        "initial_static_score must match the Gaussian count"
                    )
                static_score = torch.nan_to_num(
                    static_score, nan=0.0, posinf=0.0, neginf=0.0
                ).clamp(0.0, 1.0)
                if initial_unknown_score is None:
                    unknown_score = (1.0 - torch.maximum(score, static_score)).clamp(
                        0.0, 1.0
                    )
                else:
                    unknown_score = (
                        initial_unknown_score.detach()
                        .reshape(-1)
                        .to(device=hard.device, dtype=torch.float32)
                    )
                    if unknown_score.numel() != self.num_points:
                        raise ValueError(
                            "initial_unknown_score must match the Gaussian count"
                        )
                    unknown_score = torch.nan_to_num(
                        unknown_score, nan=1.0, posinf=1.0, neginf=1.0
                    ).clamp(0.0, 1.0)
                # Explicit three-state evidence.  UNKNOWN pulls the initial
                # material probability toward 0.5 instead of pretending to be
                # static.  This is the key occlusion-safe initialization.
                sd = static_score + score
                evidence_p_dyn = torch.where(
                    sd > 1.0e-8,
                    score / sd.clamp_min(1.0e-8),
                    torch.full_like(score, 0.5),
                )
                confidence = (1.0 - unknown_score).clamp(0.0, 1.0)
                p_dyn = 0.5 + confidence * (evidence_p_dyn - 0.5)
                threshold = torch.tensor(0.5, device=hard.device, dtype=torch.float32)
                self.register_buffer("initial_static_score", static_score)
                self.register_buffer("initial_unknown_score", unknown_score)
                self.register_buffer("initial_evidence_confidence", confidence)
            else:
                static_values = score[~hard]
                dynamic_values = score[hard]
                if static_values.numel() and dynamic_values.numel():
                    threshold = 0.5 * (static_values.max() + dynamic_values.min())
                else:
                    threshold = score.median()
                robust_scale = torch.quantile(
                    torch.abs(score - threshold), 0.50
                ).clamp_min(1.0e-3)
                temperature = max(
                    float(evidence_temperature), float(0.25 * robust_scale)
                )
                p_dyn = torch.sigmoid((score - threshold) / temperature)
                self.register_buffer("initial_static_score", 1.0 - score)
                self.register_buffer("initial_unknown_score", torch.zeros_like(score))
                self.register_buffer(
                    "initial_evidence_confidence", torch.ones_like(score)
                )
            lo = float(evidence_probability_floor)
            hi = float(evidence_probability_ceiling)
            p_dyn = lo + (hi - lo) * p_dyn
            self.register_buffer("initial_dynamic_score", score)
            self.register_buffer("initial_evidence_threshold", threshold.reshape(()))
        else:
            p_dyn = torch.where(
                hard,
                torch.full(
                    (self.num_points,),
                    float(dynamic_probability),
                    device=hard.device,
                    dtype=torch.float32,
                ),
                torch.full(
                    (self.num_points,),
                    float(static_dynamic_probability),
                    device=hard.device,
                    dtype=torch.float32,
                ),
            )
            self.register_buffer("initial_dynamic_score", hard.to(torch.float32))
            self.register_buffer("initial_static_score", (~hard).to(torch.float32))
            self.register_buffer(
                "initial_unknown_score",
                torch.zeros(self.num_points, device=hard.device, dtype=torch.float32),
            )
            self.register_buffer(
                "initial_evidence_confidence",
                torch.ones(self.num_points, device=hard.device, dtype=torch.float32),
            )
            self.register_buffer(
                "initial_evidence_threshold", torch.tensor(0.5, device=hard.device)
            )

        p_dyn = p_dyn.clamp(1.0e-4, 1.0 - 1.0e-4)
        probs = torch.zeros(
            (self.num_points, self.num_experts), device=hard.device, dtype=torch.float32
        )
        probs[:, 0] = 1.0 - p_dyn
        probs[:, 1:] = p_dyn[:, None] / float(self.num_moving_parts)
        self.logits = nn.Parameter(torch.log(probs.clamp_min(1.0e-8)))
        self.register_buffer("initial_probabilities", probs.detach().clone())
        evidence_hard = p_dyn > 0.5
        self.register_buffer(
            "initial_class",
            torch.where(
                evidence_hard,
                torch.ones_like(hard, dtype=torch.long),
                torch.zeros_like(hard, dtype=torch.long),
            ),
        )
        self.register_buffer("initial_dynamic_mask", evidence_hard)

    def soft_probabilities(self, temperature: float = 1.0) -> torch.Tensor:
        return F.softmax(self.logits / max(float(temperature), 1.0e-4), dim=-1)

    def hard_labels(self) -> torch.Tensor:
        return torch.argmax(self.logits.detach(), dim=-1)

    def asymmetric_prior_loss(
        self, static_weight: float, dynamic_weight: float
    ) -> torch.Tensor:
        # Preserve the continuous initial probability rather than collapsing the
        # prior back to a binary label.  Dynamic-attribution anchors receive a
        # stronger weight because false statics are empirically more frequent.
        p = F.softmax(self.logits, dim=-1)
        target = self.initial_probabilities.to(p)
        cross_entropy = -(target * torch.log(p.clamp_min(1.0e-8))).sum(dim=-1)
        weights = torch.where(
            self.initial_dynamic_mask,
            torch.full_like(cross_entropy, float(dynamic_weight)),
            torch.full_like(cross_entropy, float(static_weight)),
        )
        weights = weights * self.initial_evidence_confidence.to(weights).clamp(
            0.05, 1.0
        )
        return (weights * cross_entropy).sum() / weights.sum().clamp_min(1.0e-8)

    def entropy_loss(self, temperature: float = 1.0) -> torch.Tensor:
        p = self.soft_probabilities(temperature)
        return (-(p * torch.log(p.clamp_min(1.0e-8))).sum(dim=-1)).mean()

    def capture_state(self) -> Dict:
        return {
            "version": self.STATE_VERSION,
            "num_moving_parts": self.num_moving_parts,
            "logits": self.logits.detach().cpu(),
            "initial_class": self.initial_class.detach().cpu(),
            "initial_dynamic_mask": self.initial_dynamic_mask.detach().cpu(),
            "initial_dynamic_score": self.initial_dynamic_score.detach().cpu(),
            "initial_static_score": self.initial_static_score.detach().cpu(),
            "initial_unknown_score": self.initial_unknown_score.detach().cpu(),
            "initial_evidence_confidence": self.initial_evidence_confidence.detach().cpu(),
            "initial_probabilities": self.initial_probabilities.detach().cpu(),
            "initial_evidence_threshold": self.initial_evidence_threshold.detach().cpu(),
        }

    @classmethod
    def from_state(cls, state: Dict, device="cuda") -> "MaterialAssignmentField":
        version = int(state.get("version", 0))
        if version not in (1, 2, 3):
            raise RuntimeError("Unsupported MaterialAssignmentField state")
        initial = state["initial_dynamic_mask"].to(device)
        score = state.get("initial_dynamic_score", None)
        static_score = state.get("initial_static_score", None)
        unknown_score = state.get("initial_unknown_score", None)
        result = cls(
            initial,
            int(state["num_moving_parts"]),
            initial_dynamic_score=None if score is None else score.to(device),
            initial_static_score=None
            if static_score is None
            else static_score.to(device),
            initial_unknown_score=None
            if unknown_score is None
            else unknown_score.to(device),
        )
        with torch.no_grad():
            result.logits.copy_(state["logits"].to(device))
            if version >= 2 and "initial_probabilities" in state:
                result.initial_probabilities.copy_(
                    state["initial_probabilities"].to(device)
                )
        return result
