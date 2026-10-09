"""Soft per-endpoint "this probably moved" weights, with no ground truth.

These weights decide which cross-frame pairs are even allowed to exist, so
where the gate sits matters more than anything downstream: place it too low
and the candidate space refills with the static geometry the restriction was
supposed to remove; too high and the moved part is excluded from its own
search.

Nothing here may see a label, and nothing may carry a metric threshold
between scenes -- every cut is derived from the scene's own mismatch
distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DynamicSubsetWeights:
    """Soft dynamic weight at each endpoint, plus the gates that produced it."""

    source_weight: torch.Tensor   # [N0] q^0
    target_weight: torch.Tensor   # [NT] q^T
    source_gate: dict[str, float] = field(default_factory=dict)
    target_gate: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def _otsu_split(value: torch.Tensor, *, bins: int = 256) -> float:
    """Between-class-variance-maximising split of a 1-D distribution.

    Used instead of the median because the median assumes half the scene
    moved.  Mismatch is genuinely bimodal -- a tight static mode set by the
    disagreement between two independent trainings, and a heavy dynamic tail
    -- so the boundary belongs where the data puts it, which is the only way
    the same code can serve a scene with 5% motion and one with 50%.
    """
    v = value[torch.isfinite(value)].float()
    if int(v.numel()) < 2:
        return float(v.median()) if int(v.numel()) else 0.0
    low, high = float(v.min()), float(v.max())
    if not high > low:
        return low
    hist = torch.histc(v, bins=int(bins), min=low, max=high)
    weight = hist / hist.sum().clamp_min(1.0)
    centre = torch.linspace(low, high, int(bins), device=v.device, dtype=v.dtype)
    w0 = torch.cumsum(weight, dim=0)
    w1 = 1.0 - w0
    m0 = torch.cumsum(weight * centre, dim=0) / w0.clamp_min(1.0e-12)
    total = float((weight * centre).sum())
    m1 = (total - torch.cumsum(weight * centre, dim=0)) / w1.clamp_min(1.0e-12)
    between = w0 * w1 * (m0 - m1).square()
    return float(centre[int(torch.argmax(between))])


def robust_dynamic_gate(
    mismatch: torch.Tensor,
    *,
    mode: str = "otsu",
    center_quantile: float = 0.50,
    temperature_scale: float = 1.0,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``q = sigmoid((log(E + eps) - mu) / tau)``, returned with its gate.

    The log is not cosmetic.  Mismatch is a squared distance ratio spanning
    several orders of magnitude, so in linear units a robust spread is set
    almost entirely by the few largest values and the gate ends up
    effectively binary.  In log units the static mode is a roughly symmetric
    cluster and MAD means what it should.

    ``mode``:
      ``otsu``     split where the scene's own bimodality puts it (default);
      ``median``   the literal spec form, centred on the median;
      ``quantile`` centred on an explicit quantile, for ablations.
    """
    value = torch.as_tensor(mismatch).detach().reshape(-1)
    logv = torch.log(value.clamp_min(0.0) + float(eps))
    finite = logv[torch.isfinite(logv)]
    if int(finite.numel()) == 0:
        return torch.zeros_like(value), {"mode": mode, "mu": 0.0, "tau": 1.0}

    if mode == "otsu":
        mu = _otsu_split(finite)
    elif mode == "median":
        mu = float(finite.median())
    elif mode == "quantile":
        mu = float(torch.quantile(finite.float(), float(center_quantile)))
    else:
        raise ValueError(f"unknown dynamic gate mode {mode!r}")

    # The temperature is the STATIC mode's own spread, not the spread of the
    # whole distribution.  Mismatch is bimodal by construction, so a global
    # MAD is set mostly by the gap BETWEEN the modes: on a scene whose static
    # points matched exactly and whose moved block sat at ~76, the global MAD
    # gave tau = 10.2 in log units and the gate never left the middle of the
    # sigmoid, even though the two populations did not overlap at all.  What
    # the gate needs is the scale of static-side disagreement -- how much two
    # independently trained models differ where nothing moved -- because that
    # is what makes a large mismatch surprising.
    lower = finite[finite <= float(mu)]
    spread = float((lower - lower.median()).abs().median()) * 1.4826 if int(lower.numel()) > 1 else 0.0
    if not spread > 0.0:
        # A static mode with no spread at all (synthetic, or an exactly
        # reused cloud) means any excursion above it is significant, so fall
        # back to a small floor rather than to the between-mode gap.
        spread = max(0.05 * float((finite - finite.median()).abs().median() * 1.4826), 0.1)
    tau = max(spread * float(temperature_scale), 1.0e-6)
    weight = torch.sigmoid((logv - float(mu)) / tau)
    return weight, {
        "mode": str(mode),
        "mu": float(mu),
        "tau": float(tau),
        "mu_linear": float(torch.exp(torch.tensor(mu)) - float(eps)),
        "active_fraction": float((weight > 0.5).float().mean()),
        "mean_weight": float(weight.mean()),
    }


def build_soft_dynamic_weights(
    mismatch,
    *,
    mode: str = "otsu",
    center_quantile: float = 0.50,
    temperature_scale: float = 1.0,
) -> DynamicSubsetWeights:
    """``q^0`` and ``q^T`` from a :class:`BidirectionalSpatialMismatch`.

    The two gates are fitted INDEPENDENTLY.  The endpoints have different
    point counts, densification histories and local scales, so a gate fitted
    on one and applied to the other would silently make the two directions
    mean different things.
    """
    source_weight, source_gate = robust_dynamic_gate(
        mismatch.forward, mode=mode, center_quantile=center_quantile, temperature_scale=temperature_scale
    )
    target_weight, target_gate = robust_dynamic_gate(
        mismatch.backward, mode=mode, center_quantile=center_quantile, temperature_scale=temperature_scale
    )
    return DynamicSubsetWeights(
        source_weight=source_weight,
        target_weight=target_weight,
        source_gate=source_gate,
        target_gate=target_gate,
        stats={
            "source_count": int(source_weight.numel()),
            "target_count": int(target_weight.numel()),
            "source_active_fraction": source_gate["active_fraction"],
            "target_active_fraction": target_gate["active_fraction"],
        },
    )


def build_hard_dynamic_subset(
    weight: torch.Tensor,
    *,
    threshold: float = 0.5,
    min_fraction: float = 0.02,
    max_fraction: float = 0.95,
) -> torch.Tensor:
    """Boolean subset for the hard-restriction ablation.

    The fraction is clamped rather than left free: an empty subset makes the
    dynamic branch vacuously "consistent" (nothing to be wrong about) and a
    near-total one is not a restriction at all.  Both would be reported as
    results rather than as the degenerate configurations they are.
    """
    weight = torch.as_tensor(weight).detach().reshape(-1)
    subset = weight > float(threshold)
    fraction = float(subset.float().mean())
    count = int(weight.numel())
    if fraction < float(min_fraction):
        keep = max(1, int(float(min_fraction) * count))
        subset = torch.zeros_like(subset)
        subset[torch.topk(weight, k=keep).indices] = True
    elif fraction > float(max_fraction):
        keep = max(1, int(float(max_fraction) * count))
        subset = torch.zeros_like(subset)
        subset[torch.topk(weight, k=keep).indices] = True
    return subset


def diagnostic_mismatch_bands(
    mismatch: torch.Tensor,
    *,
    low_quantile: float = 0.40,
    high_quantile: float = 0.70,
) -> dict[str, torch.Tensor]:
    """§8's low / medium / high bands. Diagnostics only -- never a hard label."""
    value = torch.as_tensor(mismatch).detach().reshape(-1).float()
    low_cut = float(torch.quantile(value, float(low_quantile)))
    high_cut = float(torch.quantile(value, float(high_quantile)))
    return {
        "low": value <= low_cut,
        "medium": (value > low_cut) & (value <= high_cut),
        "high": value > high_cut,
    }


def summarise_dynamic_subset(
    weight: torch.Tensor,
    mismatch: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """GT-free description of one endpoint's dynamic weighting."""
    weight = torch.as_tensor(weight).detach().reshape(-1).float()
    value = torch.as_tensor(mismatch).detach().reshape(-1).float()
    active = weight > float(threshold)
    bands = diagnostic_mismatch_bands(value)
    return {
        "count": int(weight.numel()),
        "active_count": int(active.sum()),
        "active_fraction": float(active.float().mean()),
        "mean_weight": float(weight.mean()),
        "median_weight": float(weight.median()),
        "mismatch_median": float(value.median()),
        "mismatch_p90": float(torch.quantile(value, 0.90)),
        "mismatch_median_active": float(value[active].median()) if bool(active.any()) else 0.0,
        "mismatch_median_inactive": float(value[~active].median()) if bool((~active).any()) else 0.0,
        "band_fractions": {name: float(mask.float().mean()) for name, mask in bands.items()},
    }
