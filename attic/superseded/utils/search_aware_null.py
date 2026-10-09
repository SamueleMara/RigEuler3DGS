"""Nulls that reproduce the search, not a single random pair.

§10 of the audit.  A source patch evaluates ``M`` candidates and keeps the
best, so the observed statistic is a MINIMUM OVER M:

    E_obs = min_{m=1..M} E_{i,j_m}

Comparing that to the distribution of a single random pair is invalid, and
invalid in the direction that manufactures discoveries: the minimum of M
draws from any distribution sits far in its lower tail by construction, so
the p-value shrinks as the search widens even when nothing real is there.
Searching harder would look like better physics.

The fix is to make the null do the same search.  Each null replicate draws
its own set of M candidates, runs the identical metric and rigid stages, and
contributes its own MINIMUM.  The observed minimum is then compared against
the distribution of null minima, which is the only comparison that means what
it claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

EPS = 1.0e-12


@dataclass
class SearchAwareNull:
    """Null minima over the same search width the observation used."""

    metric_minima: torch.Tensor      # [B, replicates]
    se3_minima: torch.Tensor         # [B, replicates]
    candidates_per_search: int
    replicates: int
    stats: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def search_aware_pvalue(
    observed: torch.Tensor,
    null_minima: torch.Tensor,
) -> torch.Tensor:
    """``p = (1 + #{null minima <= observed}) / (B + 1)``.

    The ``+1`` in both places is the standard finite-sample correction: with
    ``B`` replicates a p-value of exactly zero is not supported by the sample,
    and reporting one would overstate the evidence by however many replicates
    were not run.
    """
    observed = torch.as_tensor(observed).detach().float().reshape(-1)
    null_minima = torch.as_tensor(null_minima).detach().float()
    better = (null_minima <= observed[:, None]).sum(dim=1).float()
    return (1.0 + better) / float(int(null_minima.shape[1]) + 1)


@torch.no_grad()
def build_search_aware_null(
    evaluate: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    *,
    batch: int,
    pool_size: int,
    candidates_per_search: int,
    replicates: int = 16,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    progress_desc: str = "",
) -> SearchAwareNull:
    """Run ``replicates`` independent searches of the same width.

    ``evaluate`` takes ``[B]`` target-pool indices and returns
    ``(metric distortion, se3 residual)`` for that pairing -- the SAME
    function the observation uses, because a null evaluated by a cheaper or
    different procedure would be measuring a different experiment.

    Cost is ``replicates x candidates_per_search`` evaluations, which is the
    honest price of the correction; a cheaper null is a wrong null.
    """
    from utils.progress import bar

    generator = generator or torch.Generator(device="cpu").manual_seed(0)
    metric_minima = torch.empty((int(batch), int(replicates)), device=device)
    se3_minima = torch.empty((int(batch), int(replicates)), device=device)

    outer = bar(range(int(replicates)), desc=progress_desc or "null replicates", unit="rep")
    for replicate in outer:
        best_metric = None
        best_se3 = None
        for _ in range(max(1, int(candidates_per_search))):
            picks = torch.randint(0, int(pool_size), (int(batch),), generator=generator).to(device)
            distortion, residual = evaluate(picks)
            best_metric = distortion if best_metric is None else torch.minimum(best_metric, distortion)
            best_se3 = residual if best_se3 is None else torch.minimum(best_se3, residual)
        metric_minima[:, replicate] = best_metric
        se3_minima[:, replicate] = best_se3
        outer.set_postfix({
            "min E_met": f"{float(best_metric.median()):.4f}",
            "min E_SE3": f"{float(best_se3.median()):.4f}",
        })
    outer.close()

    return SearchAwareNull(
        metric_minima=metric_minima,
        se3_minima=se3_minima,
        candidates_per_search=int(candidates_per_search),
        replicates=int(replicates),
        stats={
            "median_null_metric_minimum": float(metric_minima.median()),
            "median_null_se3_minimum": float(se3_minima.median()),
            "evaluations": int(replicates) * int(candidates_per_search),
        },
    )


def benjamini_hochberg(p_values: torch.Tensor, *, alpha: float = 0.05) -> tuple[torch.Tensor, float]:
    """One global FDR over the COMPLETE tested hypothesis family (§12, §13).

    Applied across every hypothesis of every source anchor, not per anchor:
    running BH separately per anchor would control the error rate within each
    anchor while letting it grow without bound across thousands of them.
    """
    p = torch.as_tensor(p_values).detach().float().reshape(-1)
    n = int(p.numel())
    if n == 0:
        return torch.zeros(0, dtype=torch.bool), 0.0
    ranked = p[torch.argsort(p)]
    limits = float(alpha) * torch.arange(1, n + 1, device=p.device, dtype=p.dtype) / float(n)
    passed = ranked <= limits
    if not bool(passed.any()):
        return torch.zeros(n, dtype=torch.bool, device=p.device), 0.0
    cut = float(ranked[int(torch.nonzero(passed).max())])
    return p <= cut, cut


@dataclass
class CoverageReport:
    """§16/§42: what the search actually looked at, stated without flattery."""

    target_domain_size: int
    unique_target_anchors_tested: int
    candidates_per_source: int
    per_source_coverage: torch.Tensor

    @property
    def target_anchor_search_fraction(self) -> float:
        return float(self.unique_target_anchors_tested) / max(1, int(self.target_domain_size))

    def summary(self) -> dict[str, Any]:
        coverage = self.per_source_coverage.detach().float().cpu()
        return {
            # Three different numbers that the old single "coverage" conflated.
            "requested_candidate_slots_filled": float(
                self.candidates_per_source / max(1, self.unique_target_anchors_tested)
            ),
            "target_anchor_search_fraction": self.target_anchor_search_fraction,
            "physical_target_domain_size": int(self.target_domain_size),
            "unique_target_anchors_tested": int(self.unique_target_anchors_tested),
            "per_source_coverage_median": float(coverage.median()) if coverage.numel() else 0.0,
            "per_source_coverage_p10": float(torch.quantile(coverage, 0.10)) if coverage.numel() else 0.0,
            "per_source_coverage_p90": float(torch.quantile(coverage, 0.90)) if coverage.numel() else 0.0,
        }


def build_coverage_report(
    target_domain_size: int,
    tested_target_indices: torch.Tensor,
    candidates_per_source: int,
) -> CoverageReport:
    """Coverage against the physical target domain, not against the slots asked for.

    Filling every requested candidate slot says nothing about whether the
    true counterpart was among them: 256 slots drawn from 5936 dynamic target
    anchors is 4.3% of the domain, and reporting that as 100% coverage --
    which the previous probe did -- turns a search limit into an apparent
    physical result.
    """
    tested = torch.as_tensor(tested_target_indices).detach().reshape(int(tested_target_indices.shape[0]), -1)
    unique = int(torch.unique(tested).numel())
    per_source = torch.tensor([
        float(torch.unique(row).numel()) / max(1, int(target_domain_size)) for row in tested.cpu()
    ])
    return CoverageReport(
        target_domain_size=int(target_domain_size),
        unique_target_anchors_tested=unique,
        candidates_per_source=int(candidates_per_source),
        per_source_coverage=per_source,
    )
