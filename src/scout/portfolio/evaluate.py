"""Forward scoring: grade pre-registered picks against what price actually did.

This is the only credible evidence the project has (PLAN.md 1.3), so it is
written to be honest to a fault. Three disciplines, all carried over from the old
`score.js` in spirit:

  - **Report the distribution, not the hit rate.** Bessembinder (2018): returns
    are so right-skewed that a strategy can be right 55% of the time and still
    lose money. So every strategy gets its full quantile spread and its median
    stated next to its mean; the hit rate is shown but never alone.
  - **Say when the sample is too small to mean anything.** The single most
    important line `score.js` printed: a 70% hit rate on a handful of picks is
    what a coin flip returns routinely. Below ~30 scored picks the verdict is
    'insufficient evidence', full stop.
  - **Name the bar the agent must clear.** If the agent does not beat the
    EV/EBIT-decile baseline, its whole research pipeline is cost, not signal --
    and the evaluation says exactly that.

Prices are inputs, never fetched here (there is no price feed yet, and when there
is one it should be swappable): the caller passes a forward-price map, mirroring
how the metrics layer takes `MarketData` rather than reaching for a vendor.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import date

from scout.portfolio.baselines import LabeledObservation
from scout.portfolio.models import (
    Comparison,
    Distribution,
    Evaluation,
    PaperPick,
    ScoredPick,
    Strategy,
    StrategyScore,
)

# PLAN.md 6 cost model: 3-6% round-trip plus impact and withholding drag. A flat
# 4% round trip is the honest simplification until an ADV-scaled impact model and
# per-venue withholding are wired in; it is a parameter so it is never hidden.
DEFAULT_COST_BPS = 400.0

# Below this many scored picks, no return difference is distinguishable from
# noise -- the same threshold score.js used, for the same reason.
MIN_CREDIBLE_SAMPLE = 30

# The dumb baselines, in the order PLAN.md lists them. The EV/EBIT decile is the
# one that decides whether the LLM layer earns its keep.
_BASELINES = (Strategy.UNIVERSE_EW, Strategy.SCREEN, Strategy.EV_EBIT_DECILE, Strategy.GBDT)


def evaluate(
    picks: list[PaperPick],
    forward_prices: dict[str, float],
    *,
    as_of_exit: date,
    cost_bps: float = DEFAULT_COST_BPS,
) -> Evaluation:
    """Grade every strategy in `picks` on the same forward-price map.

    A pick is graded only if it has both a reference price (recorded at pick
    time) and a forward price (supplied here); anything missing either is counted
    as ungradeable and reported, never guessed to zero.
    """
    by_strategy: dict[Strategy, list[PaperPick]] = defaultdict(list)
    for pick in picks:
        by_strategy[pick.strategy].append(pick)

    scores: dict[Strategy, StrategyScore] = {}
    for strategy, strat_picks in by_strategy.items():
        scores[strategy] = _score_strategy(strategy, strat_picks, forward_prices, cost_bps)

    comparisons = _compare_agent_to_baselines(scores)
    total_scored = sum(s.n_scored for s in scores.values())
    warnings = _sample_warnings(scores)

    return Evaluation(
        as_of_exit=as_of_exit,
        cost_bps=cost_bps,
        scores=scores,
        comparisons=comparisons,
        total_scored=total_scored,
        warnings=warnings,
    )


def score_picks(
    picks: list[PaperPick], forward_prices: dict[str, float], *, cost_bps: float
) -> list[ScoredPick]:
    """Grade the individual gradeable picks in a book. Weight is not applied here
    -- this is the per-pick return, which the distribution is built from."""
    scored: list[ScoredPick] = []
    for pick in picks:
        entry = pick.reference_price
        exit_price = forward_prices.get(pick.entity_id)
        if entry is None or entry == 0 or exit_price is None:
            continue
        gross = exit_price / entry - 1.0
        net = gross - cost_bps / 10_000.0
        scored.append(
            ScoredPick(pick=pick, exit_price=exit_price, gross_return=gross, net_return=net)
        )
    return scored


def _score_strategy(
    strategy: Strategy,
    picks: list[PaperPick],
    forward_prices: dict[str, float],
    cost_bps: float,
) -> StrategyScore:
    # The book is the names actually held: weight > 0. A weight-0 pick (an agent
    # veto) is recorded history, not a position, so it never enters the return.
    book = [p for p in picks if p.weight > 0]
    scored = score_picks(book, forward_prices, cost_bps=cost_bps)
    ungradeable = len(book) - len(scored)

    if not scored:
        note = "nothing in this book could be graded (missing reference or forward prices)."
        return StrategyScore(
            strategy=strategy,
            n_picks=len(picks),
            n_scored=0,
            portfolio_return=None,
            distribution=None,
            ungradeable=ungradeable,
            notes=[note],
        )

    portfolio_return = _weighted_return(scored)
    distribution = _distribution([s.net_return for s in scored])
    notes: list[str] = []
    if distribution.mean - distribution.median > 0.02:
        notes.append(
            "mean sits well above median: the book's return leans on a few big "
            "winners (Bessembinder skew), so the median is the more honest summary."
        )
    return StrategyScore(
        strategy=strategy,
        n_picks=len(picks),
        n_scored=len(scored),
        portfolio_return=portfolio_return,
        distribution=distribution,
        ungradeable=ungradeable,
        notes=notes,
    )


def _weighted_return(scored: list[ScoredPick]) -> float:
    """The book's net return, weights renormalized over the gradeable names.

    Renormalizing (rather than treating an ungradeable name as a zero-return
    holding) keeps a data gap in OUR prices from masquerading as a real cash
    position -- the same principle the screen uses when a metric block is absent.
    """
    total_weight = sum(s.pick.weight for s in scored)
    if total_weight == 0:
        return statistics.fmean(s.net_return for s in scored)
    return sum(s.pick.weight * s.net_return for s in scored) / total_weight


def _distribution(returns: list[float]) -> Distribution:
    ordered = sorted(returns)
    n = len(ordered)
    hits = sum(1 for r in ordered if r > 0)
    return Distribution(
        n=n,
        mean=statistics.fmean(ordered),
        median=statistics.median(ordered),
        stdev=statistics.pstdev(ordered) if n > 1 else 0.0,
        minimum=ordered[0],
        p10=_percentile(ordered, 0.10),
        p25=_percentile(ordered, 0.25),
        p75=_percentile(ordered, 0.75),
        p90=_percentile(ordered, 0.90),
        maximum=ordered[-1],
        hit_rate=hits / n,
    )


def _compare_agent_to_baselines(scores: dict[Strategy, StrategyScore]) -> list[Comparison]:
    """The agent against each dumb baseline. No agent book => no comparison."""
    agent = scores.get(Strategy.AGENT)
    comparisons: list[Comparison] = []
    for baseline in _BASELINES:
        base = scores.get(baseline)
        if base is None:
            continue
        comparisons.append(_compare(agent, base, baseline))
    return comparisons


def _compare(
    agent: StrategyScore | None, base: StrategyScore, baseline: Strategy
) -> Comparison:
    agent_ret = agent.portfolio_return if agent else None
    base_ret = base.portfolio_return
    n_agent = agent.n_scored if agent else 0

    # Insufficient evidence dominates: a difference on a handful of picks is noise,
    # so we refuse to call a winner before either side clears the sample bar.
    if agent is None or agent_ret is None or base_ret is None:
        return Comparison(baseline, agent_ret, base_ret, None, "no agent book to compare yet.")
    if n_agent < MIN_CREDIBLE_SAMPLE or base.n_scored < MIN_CREDIBLE_SAMPLE:
        return Comparison(
            baseline,
            agent_ret,
            base_ret,
            agent_ret - base_ret,
            f"insufficient evidence (agent N={n_agent}, {baseline.value} N={base.n_scored}; "
            f"need >= {MIN_CREDIBLE_SAMPLE} each). Any gap here is noise.",
        )

    delta = agent_ret - base_ret
    if baseline == Strategy.EV_EBIT_DECILE and delta <= 0:
        verdict = (
            f"the agent did NOT beat the EV/EBIT decile ({delta:+.1%}). On this evidence "
            "the LLM research layer is cost, not signal -- a one-line value screen did as "
            "well or better."
        )
    elif delta > 0:
        verdict = f"the agent beat {baseline.value} by {delta:+.1%}."
    else:
        verdict = f"the agent trailed {baseline.value} by {delta:+.1%}."
    return Comparison(baseline, agent_ret, base_ret, delta, verdict)


def _sample_warnings(scores: dict[Strategy, StrategyScore]) -> list[str]:
    """The load-bearing honesty line from score.js, adapted to returns."""
    best_n = max((s.n_scored for s in scores.values()), default=0)
    warnings: list[str] = []
    if best_n == 0:
        warnings.append(
            "Nothing scoreable yet: no pick had both a reference and a forward price. "
            "Supply forward prices and let the picks age before reading anything into this."
        )
    elif best_n < MIN_CREDIBLE_SAMPLE:
        warnings.append(
            f"!! {best_n} scored pick(s) is FAR too few to mean anything. A coin flip "
            "returns 70%+ hit rates on small samples routinely, and one big winner swings "
            f"the mean entirely. Treat this as a plumbing check, not evidence -- aim for "
            f"{MIN_CREDIBLE_SAMPLE}+ per strategy, ideally 100+, before judging."
        )
    return warnings


def labeled_observations(scored: list[ScoredPick]) -> list[LabeledObservation]:
    """Turn scored picks into GBDT training rows (features -> realized return).

    The bridge that eventually lets baseline #3 train on the forward archive this
    very module accumulates. Only picks that carry features and were graded
    contribute a row.
    """
    return [
        LabeledObservation(features=dict(s.pick.features), forward_return=s.net_return)
        for s in scored
        if s.pick.features
    ]


def _percentile(ordered: list[float], p: float) -> float:
    """Linear-interpolation percentile, matching screen/rank.py so the two layers
    describe a distribution the same way. `ordered` must be pre-sorted."""
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    pos = p * (len(ordered) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (pos - low) * (ordered[high] - ordered[low])
