"""Within-cohort ranking of screen survivors on a cheap x quality x safety score.

WHY within-cohort (never across). A value multiple only means something against
a comparable book. JGAAP amortizes goodwill where IFRS impairment-tests, retains
extraordinary items IFRS abolished, and defines operating income differently
(PLAN.md 1.4); a "cheap" EV/EBIT under one standard is not the same claim as
under another, and a Japanese microcap and a US biotech are not peers. So every
z-score is computed against the entity's own (country x accounting-standard x
sector) cohort and never pooled across cohorts.

WHY winsorize before z-scoring. Microcap ratios have wild fat tails -- a
near-zero denominator produces a 5000% "yield" that is an artefact, not a signal.
One such value would drag the cohort mean and inflate the std so hard that every
real peer collapses toward z=0. So each metric's values are clipped to the
cohort's 5th/95th percentile before the mean/std (and the entity's own value) are
taken, bounding the outlier's influence without discarding the row.

WHY renormalize the weights over present blocks. Price is a separate, paid,
deferred feed (see MarketData); most names arrive without one, so their whole
cheap block is absent. Scoring a missing block as zero would shove every
unpriced-but-healthy company to the bottom -- punishing a name for a gap in OUR
data, not for being a worse business. Instead the composite is a weighted mean
over the blocks that ARE present, weights renormalized, so quality+safety alone
still rank a name fairly.

Standard library only (statistics) -- no numpy/pandas dependency.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

from scout.screen.models import CohortKey, ScoredCandidate


@dataclass(slots=True)
class RankInput:
    """One survivor's identity, cohort and the metrics that actually computed."""

    entity_id: str
    name: str | None
    cohort: CohortKey
    metrics: dict[str, float]
    """Metric name -> value. Only metrics that computed (ok) belong here."""

    underfollowed: bool = False


@dataclass(frozen=True, slots=True)
class RankWeights:
    """Composite weights per block. Defaults from PLAN.md phase 4."""

    cheap: float = 0.40
    quality: float = 0.35
    safety: float = 0.25


DEFAULT_WEIGHTS = RankWeights()

# Self-funding firms report an infinite cash runway. Left as inf it would poison
# the cohort mean/std; capped it still lands at the top of the safety block
# without a nan/inf leaking into any composite.
_CASH_RUNWAY_CAP_MONTHS = 120.0

# Direction map: metric -> higher_is_better. A False flips the z-score's sign so
# that a HIGHER block score is always more attractive, whatever the raw metric's
# natural direction. Get these signs wrong and the screen ranks upside down.
_CHEAP: dict[str, bool] = {
    "earnings_yield": True,
    "fcf_yield": True,
    "net_cash_to_mcap": True,
    "ev_ebit": False,  # a lower multiple is cheaper
    "ev_sales": False,  # a lower multiple is cheaper
    "ncav_to_mcap": True,
}
_QUALITY: dict[str, bool] = {
    "gp_to_assets": True,
    "roic": True,
    "piotroski_f": True,
    "accruals": False,  # high accruals flag low earnings quality
}
_SAFETY: dict[str, bool] = {
    "altman_z": True,
    "cash_runway_months": True,
    "beneish_m": False,  # more negative = less manipulation risk
    "share_issuance": False,  # dilution is bad
}
_BLOCKS: dict[str, dict[str, bool]] = {
    "cheap": _CHEAP,
    "quality": _QUALITY,
    "safety": _SAFETY,
}
_RANKED_METRICS: frozenset[str] = frozenset(
    name for block in _BLOCKS.values() for name in block
)


@dataclass(frozen=True, slots=True)
class _MetricStat:
    """Winsorization bounds and post-clip mean/std for one metric in one cohort."""

    lo: float
    hi: float
    mean: float
    std: float

    def z(self, raw: float) -> float:
        """Z-score of a raw value: clip to the winsor bounds first, then standardise."""
        clipped = min(max(raw, self.lo), self.hi)
        # std == 0 means every peer had the same value -- no information, so z is
        # flat 0 rather than a division by zero.
        if self.std == 0:
            return 0.0
        return (clipped - self.mean) / self.std


def rank(
    inputs: list[RankInput],
    *,
    weights: RankWeights = DEFAULT_WEIGHTS,
    min_cohort: int = 3,
) -> list[ScoredCandidate]:
    """Rank survivors within each cohort, then globally by composite (None last).

    Cohorts smaller than `min_cohort` cannot be z-scored meaningfully, so their
    members are emitted unranked (composite/blocks None) with a warning rather
    than dropped -- the caller still needs to see them.
    """
    by_cohort: dict[CohortKey, list[RankInput]] = defaultdict(list)
    for item in inputs:
        by_cohort[item.cohort].append(item)

    results: list[ScoredCandidate] = []
    for cohort, members in by_cohort.items():
        if len(members) < min_cohort:
            results.extend(_emit_unranked(cohort, members, min_cohort))
            continue
        results.extend(_rank_cohort(cohort, members, weights))

    # Global order: best composite first, None (unrankable) last. Python's sort
    # is stable, so members of a cohort keep their relative order within a tier.
    results.sort(key=lambda c: (c.composite is None, -(c.composite or 0.0)))
    return results


def _emit_unranked(
    cohort: CohortKey, members: list[RankInput], min_cohort: int
) -> list[ScoredCandidate]:
    """Candidates for a too-small cohort: present but unranked, with a note."""
    warning = (
        f"cohort '{cohort.label()}' has {len(members)} < {min_cohort} members "
        "-- not enough peers to z-score; ranked by nothing"
    )
    return [
        ScoredCandidate(
            entity_id=m.entity_id,
            name=m.name,
            cohort=cohort,
            composite=None,
            cheap=None,
            quality=None,
            safety=None,
            underfollowed=m.underfollowed,
            metric_values=dict(m.metrics),
            warnings=[warning],
        )
        for m in members
    ]


def _rank_cohort(
    cohort: CohortKey, members: list[RankInput], weights: RankWeights
) -> list[ScoredCandidate]:
    """Score every member of a large-enough cohort."""
    sanitized = [_sanitize(m.metrics) for m in members]
    stats = _cohort_stats(sanitized)

    scored: list[ScoredCandidate] = []
    for member, clean in zip(members, sanitized, strict=True):
        blocks = {name: _block_score(clean, block, stats) for name, block in _BLOCKS.items()}
        composite = _composite(blocks, weights)
        scored.append(
            ScoredCandidate(
                entity_id=member.entity_id,
                name=member.name,
                cohort=cohort,
                composite=composite,
                cheap=blocks["cheap"],
                quality=blocks["quality"],
                safety=blocks["safety"],
                underfollowed=member.underfollowed,
                metric_values=dict(member.metrics),
                warnings=_block_warnings(blocks),
            )
        )
    return scored


def _sanitize(metrics: dict[str, float]) -> dict[str, float]:
    """Keep the ranked, finite metrics, applying the cash-runway inf cap.

    Non-finite values are treated as absent (like a metric that did not compute)
    rather than propagated -- a nan/inf must never reach a mean or a composite.
    This is data hygiene on trusted-but-messy inputs, not a swallowed error.
    """
    clean: dict[str, float] = {}
    for name, value in metrics.items():
        if name not in _RANKED_METRICS:
            continue
        if name == "cash_runway_months" and math.isinf(value) and value > 0:
            value = _CASH_RUNWAY_CAP_MONTHS
        if not math.isfinite(value):
            continue
        clean[name] = value
    return clean


def _cohort_stats(sanitized: list[dict[str, float]]) -> dict[str, _MetricStat]:
    """Winsorized mean/std for each metric, over the members that reported it."""
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for clean in sanitized:
        for name, value in clean.items():
            values_by_metric[name].append(value)

    stats: dict[str, _MetricStat] = {}
    for name, values in values_by_metric.items():
        lo = _percentile(values, 0.05)
        hi = _percentile(values, 0.95)
        clipped = [min(max(v, lo), hi) for v in values]
        stats[name] = _MetricStat(
            lo=lo,
            hi=hi,
            mean=statistics.fmean(clipped),
            std=statistics.pstdev(clipped),
        )
    return stats


def _block_score(
    clean: dict[str, float], block: dict[str, bool], stats: dict[str, _MetricStat]
) -> float | None:
    """Mean of the entity's available z-scores in a block, or None if it has none."""
    zs: list[float] = []
    for name, higher_is_better in block.items():
        if name not in clean:
            continue
        stat = stats[name]  # present in clean => present in stats by construction
        z = stat.z(clean[name])
        # Flip so a higher block score is always more attractive.
        zs.append(z if higher_is_better else -z)
    if not zs:
        return None
    return statistics.fmean(zs)


def _composite(blocks: dict[str, float | None], weights: RankWeights) -> float | None:
    """Weighted mean over the PRESENT blocks, weights renormalized over them.

    A missing block does not count as zero -- it is dropped and the remaining
    weights are rescaled, so a name is never penalised for a block it could not
    populate (typically the price-dependent cheap block).
    """
    weight_of = {"cheap": weights.cheap, "quality": weights.quality, "safety": weights.safety}
    present = [(score, weight_of[name]) for name, score in blocks.items() if score is not None]
    if not present:
        return None
    total_weight = sum(w for _, w in present)
    if total_weight == 0:
        return None
    return sum(score * w for score, w in present) / total_weight


def _block_warnings(blocks: dict[str, float | None]) -> list[str]:
    """Transparency notes for blocks that could not be scored."""
    if all(score is None for score in blocks.values()):
        return ["no ranked metrics available in any block; composite is None"]
    return [
        f"no {name}-block metrics available; composite renormalized over the rest"
        for name, score in blocks.items()
        if score is None
    ]


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default 'linear' method), p in [0, 1].

    Kept explicit rather than pulled from statistics.quantiles so the
    small-cohort winsorization behaviour is exactly what the module docstring
    claims and does not shift with a stdlib method change.
    """
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank_pos = p * (len(ordered) - 1)
    low = math.floor(rank_pos)
    high = math.ceil(rank_pos)
    if low == high:
        return ordered[low]
    frac = rank_pos - low
    return ordered[low] + frac * (ordered[high] - ordered[low])


__all__ = ["DEFAULT_WEIGHTS", "RankInput", "RankWeights", "rank"]
