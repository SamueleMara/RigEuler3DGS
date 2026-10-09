"""Optional diffusion hypothesis tests for progressive rigid discovery.

Patch 6 of the spec plugs the trained denoiser in here.  Until that patch is
enabled, the progressive Stage-1 code treats diffusion as absent rather than as
a normalized fake vote.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiffusionHypothesisConfig:
    enabled: bool = False
    trials: int = 2


def score_diffusion_hypotheses(*_, cfg: DiffusionHypothesisConfig | None = None, **__) -> dict[int, dict[str, float]]:
    cfg = cfg or DiffusionHypothesisConfig()
    if cfg.enabled:
        raise NotImplementedError(
            "Diffusion hypothesis testing is Patch 6 in "
            "CODEX_PROGRESSIVE_RIGID_DISCOVERY_PIPELINE.md; Stage 1A-D runs without "
            "fabricating a diffusion vote."
        )
    return {}
