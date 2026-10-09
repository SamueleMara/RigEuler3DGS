"""§40 check 5: significance must not grow with the width of the search.

The old null drew a single random pair.  The observation, however, was the
MINIMUM over M candidates, and the minimum of M draws sits far in the lower
tail of any distribution by construction.  So p-values shrank as M grew even
when nothing real was present: searching harder looked like better physics,
which is the most dangerous failure a statistical test can have.

These tests measure that directly -- run the same pure-noise experiment at
several search widths and check the false-positive rate stays at alpha.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.search_aware_null import (
    benjamini_hochberg,
    build_coverage_report,
    build_search_aware_null,
    search_aware_pvalue,
)

CHECKS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def noise_experiment(batch: int, candidates: int, *, seed: int):
    """A pure-noise 'search': every candidate is an independent draw.

    Nothing in this scene is real, so a valid test must reject at exactly the
    nominal rate no matter how many candidates are examined.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def evaluate(index: torch.Tensor):
        # The statistic ignores the index entirely -- pure noise by design.
        metric = torch.rand((int(batch),), generator=generator)
        residual = torch.rand((int(batch),), generator=generator)
        return metric, residual

    observed_metric = None
    observed_se3 = None
    for _ in range(int(candidates)):
        metric, residual = evaluate(torch.zeros(int(batch), dtype=torch.long))
        observed_metric = metric if observed_metric is None else torch.minimum(observed_metric, metric)
        observed_se3 = residual if observed_se3 is None else torch.minimum(observed_se3, residual)
    return evaluate, observed_metric, observed_se3, generator


def test_false_positive_rate_does_not_grow_with_search_width() -> None:
    batch = 200
    rates = {}
    for candidates in (2, 8, 32):
        evaluate, observed_metric, _observed_se3, generator = noise_experiment(
            batch, candidates, seed=100 + candidates
        )
        null = build_search_aware_null(
            evaluate, batch=batch, pool_size=1000, candidates_per_search=candidates,
            replicates=40, generator=generator, device="cpu",
        )
        p = search_aware_pvalue(observed_metric, null.metric_minima)
        rates[candidates] = float((p <= 0.05).float().mean())

    for candidates, rate in rates.items():
        check(
            rate < 0.15,
            f"with M={candidates} candidates on pure noise the false-positive rate stays near alpha "
            f"({rate:.3f})",
        )
    spread = max(rates.values()) - min(rates.values())
    check(
        spread < 0.12,
        f"and it does not grow with the search width ({rates}, spread {spread:.3f})",
    )


def test_single_pair_null_would_have_been_invalid() -> None:
    """The failure the correction removes, demonstrated rather than asserted.

    Comparing a minimum-over-M observation against single-draw nulls inflates
    significance without limit as M grows.
    """
    batch = 300
    generator = torch.Generator(device="cpu").manual_seed(7)
    single_draw_null = torch.rand((batch, 200), generator=generator)

    inflated = {}
    for candidates in (1, 4, 16, 64):
        draws = torch.rand((batch, candidates), generator=generator)
        observed = draws.min(dim=1).values
        better = (single_draw_null <= observed[:, None]).sum(dim=1).float()
        p = (1.0 + better) / float(single_draw_null.shape[1] + 1)
        inflated[candidates] = float((p <= 0.05).float().mean())

    check(
        inflated[64] > 3.0 * max(inflated[1], 0.02),
        f"a single-pair null makes pure noise look ever more significant as the search widens "
        f"({inflated})",
    )
    check(
        inflated[64] > 0.4,
        f"at M=64 it would call {inflated[64]:.0%} of pure noise significant at alpha=0.05",
    )


def test_search_aware_null_still_detects_a_real_signal() -> None:
    """Correct calibration must not cost all the power.

    A test that rejects nothing is trivially valid and useless, so the same
    machinery is checked against a planted signal.

    Power is bounded by the replicate count: the smallest p-value an
    empirical null of B replicates can produce is 1/(B+1), so B has to be
    large enough for BH at the family size to be satisfiable at all.  At
    B=60 over 120 tests this recovered 0.88 of planted cases purely because
    of that floor, not because the correction lost the signal.
    """
    batch = 120
    replicates = 150
    generator = torch.Generator(device="cpu").manual_seed(11)

    def evaluate(index: torch.Tensor):
        metric = torch.rand((batch,), generator=generator)
        return metric, metric.clone()

    null = build_search_aware_null(
        evaluate, batch=batch, pool_size=500, candidates_per_search=8,
        replicates=replicates, generator=generator, device="cpu",
    )
    # Half the population has a genuinely exceptional best candidate.
    observed = null.metric_minima.median(dim=1).values.clone()
    observed[: batch // 2] = 1.0e-4

    p = search_aware_pvalue(observed, null.metric_minima)
    accepted, _cut = benjamini_hochberg(p, alpha=0.05)
    check(
        float(accepted[: batch // 2].float().mean()) > 0.9,
        f"a real signal is still detected after the correction "
        f"({float(accepted[:batch // 2].float().mean()):.2f} of planted cases at B={replicates})",
    )
    check(
        float(accepted[batch // 2:].float().mean()) < 0.1,
        f"while the null half stays rejected ({float(accepted[batch // 2:].float().mean()):.2f})",
    )


def test_fdr_controls_the_whole_hypothesis_family() -> None:
    generator = torch.Generator().manual_seed(13)
    p = torch.cat((torch.rand(500, generator=generator), torch.full((20,), 1.0e-6)))
    accepted, _cut = benjamini_hochberg(p, alpha=0.05)
    check(int(accepted[500:].sum()) == 20, "BH keeps every planted signal")
    check(
        int(accepted[:500].sum()) <= 5,
        f"and admits at most a few nulls out of 500 ({int(accepted[:500].sum())})",
    )


# --------------------------------------------------------------------------
# §16 / §42 coverage honesty
# --------------------------------------------------------------------------
def test_coverage_is_reported_against_the_physical_domain() -> None:
    """The previous probe reported 100% coverage while testing 4% of the domain.

    Filling every requested candidate slot says nothing about whether the true
    counterpart was among them, and calling that full coverage turns a search
    limit into an apparent physical result.
    """
    domain = 5936
    sources, candidates = 64, 256
    generator = torch.Generator().manual_seed(17)
    tested = torch.randint(0, domain, (sources, candidates), generator=generator)

    report = build_coverage_report(domain, tested, candidates)
    summary = report.summary()

    # The aggregate and the per-source fractions are DIFFERENT numbers, and
    # conflating them is what produced the old "coverage 1.0". Across 64
    # anchors the union of draws does touch most of the domain, but no single
    # anchor ever sees more than its own M candidates -- and it is the single
    # anchor that has to contain the true counterpart.
    check(
        summary["target_anchor_search_fraction"] > 0.5,
        f"the union of all searches covers much of the domain "
        f"({summary['target_anchor_search_fraction']:.3f})",
    )
    check(
        summary["per_source_coverage_median"] < 0.10,
        f"while each source anchor individually sees only a few percent of it "
        f"({summary['per_source_coverage_median']:.4f}) -- the number that limits discovery",
    )
    check(
        summary["target_anchor_search_fraction"] > 10.0 * summary["per_source_coverage_median"],
        "so the two must be reported separately, never collapsed into one 'coverage'",
    )
    check(
        summary["physical_target_domain_size"] == domain,
        "with the domain size stated explicitly rather than implied",
    )


def test_coverage_reaches_one_only_when_the_domain_is_exhausted() -> None:
    domain = 32
    tested = torch.arange(domain)[None, :].repeat(8, 1)
    summary = build_coverage_report(domain, tested, domain).summary()
    check(
        abs(summary["target_anchor_search_fraction"] - 1.0) < 1.0e-9,
        "full coverage is reported only when every target anchor really was tested",
    )
    check(
        abs(summary["per_source_coverage_median"] - 1.0) < 1.0e-9,
        "and per-source coverage agrees",
    )


def main() -> None:
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed, {len(CHECKS)} assertions:")
    for message in CHECKS:
        print(f"  - {message}")


if __name__ == "__main__":
    main()
