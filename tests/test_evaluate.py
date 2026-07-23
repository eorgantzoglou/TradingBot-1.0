"""Forward scoring: the return math, the distribution, and the honest verdicts.

These are the numbers the whole project is judged on, so they are checked to the
digit against hand calculation, and the honesty rules (small-sample warning, the
'cost not signal' verdict, weight renormalization over gradeable names) each get
their own test.
"""

from __future__ import annotations

from datetime import date

import pytest

from scout.portfolio.evaluate import (
    MIN_CREDIBLE_SAMPLE,
    evaluate,
    labeled_observations,
    score_picks,
)
from scout.portfolio.models import PaperPick, Strategy

EXIT = date(2026, 10, 23)


def _pick(
    entity_id: str,
    *,
    strategy: Strategy = Strategy.SCREEN,
    reference_price: float | None = 100.0,
    weight: float = 1.0,
    features: dict | None = None,
) -> PaperPick:
    return PaperPick(
        as_of=date(2026, 7, 23),
        strategy=strategy,
        entity_id=entity_id,
        name=f"Co {entity_id}",
        cohort="US / US-GAAP / manufacturing",
        reference_price=reference_price,
        currency=None,
        weight=weight,
        rank=None,
        score=None,
        vetoed=None,
        features=features or {},
        run_id="run-1",
    )


def test_score_picks_return_math():
    picks = [_pick("up", reference_price=100.0), _pick("down", reference_price=100.0)]
    scored = score_picks(picks, {"up": 110.0, "down": 90.0}, cost_bps=0.0)
    by_id = {s.pick.entity_id: s for s in scored}
    assert by_id["up"].gross_return == pytest.approx(0.10)
    assert by_id["down"].gross_return == pytest.approx(-0.10)
    # A 400bps round trip subtracts 0.04 from every gross return.
    scored_cost = score_picks(picks, {"up": 110.0, "down": 90.0}, cost_bps=400.0)
    assert {s.pick.entity_id: s.net_return for s in scored_cost}["up"] == pytest.approx(0.06)


def test_missing_prices_make_a_pick_ungradeable_not_zero():
    picks = [
        _pick("has_both", reference_price=100.0),
        _pick("no_ref", reference_price=None),      # never got an entry price
        _pick("no_fwd", reference_price=100.0),     # no forward price supplied
    ]
    scored = score_picks(picks, {"has_both": 120.0, "no_ref": 50.0}, cost_bps=0.0)
    # Only the fully-priced name is graded; the others are absent, not scored 0.
    assert [s.pick.entity_id for s in scored] == ["has_both"]


def test_weighted_return_renormalizes_over_gradeable_names():
    # Two names weighted 0.25 each in a book that also held an unpriceable third.
    # The book return must be the mean of the two graded ones (weights rescaled),
    # NOT dragged toward zero by treating the gap as a flat holding.
    picks = [
        _pick("a", reference_price=100.0, weight=0.25),
        _pick("b", reference_price=100.0, weight=0.25),
        _pick("c", reference_price=None, weight=0.5),
    ]
    result = evaluate(picks, {"a": 120.0, "b": 100.0}, as_of_exit=EXIT, cost_bps=0.0)
    score = result.scores[Strategy.SCREEN]
    assert score.n_scored == 2
    assert score.ungradeable == 1
    assert score.portfolio_return == pytest.approx(0.10)  # mean(0.20, 0.00)


def test_weight_zero_picks_are_not_in_the_book():
    # An agent veto is recorded with weight 0 -- history, not a position -- so it
    # never enters the strategy's return even though it has prices.
    picks = [
        _pick("held", strategy=Strategy.AGENT, weight=1.0),
        _pick("vetoed", strategy=Strategy.AGENT, weight=0.0),
    ]
    result = evaluate(picks, {"held": 110.0, "vetoed": 10.0}, as_of_exit=EXIT, cost_bps=0.0)
    score = result.scores[Strategy.AGENT]
    assert score.n_scored == 1
    assert score.portfolio_return == pytest.approx(0.10)


def test_distribution_quantiles_and_median():
    picks = [_pick(str(i), reference_price=100.0) for i in range(5)]
    # returns: -0.20, -0.10, 0.0, +0.10, +0.20
    fwd = {"0": 80.0, "1": 90.0, "2": 100.0, "3": 110.0, "4": 120.0}
    result = evaluate(picks, fwd, as_of_exit=EXIT, cost_bps=0.0)
    dist = result.scores[Strategy.SCREEN].distribution
    assert dist is not None
    assert dist.n == 5
    assert dist.median == pytest.approx(0.0)
    assert dist.minimum == pytest.approx(-0.20)
    assert dist.maximum == pytest.approx(0.20)
    assert dist.hit_rate == pytest.approx(0.4)  # 2 of 5 positive


def test_small_sample_warning_fires():
    picks = [_pick("a", reference_price=100.0)]
    result = evaluate(picks, {"a": 110.0}, as_of_exit=EXIT, cost_bps=0.0)
    assert any("too few" in w for w in result.warnings)


def test_no_scoreable_picks_warns_and_scores_none():
    picks = [_pick("a", reference_price=None)]
    result = evaluate(picks, {}, as_of_exit=EXIT, cost_bps=0.0)
    assert result.total_scored == 0
    assert any("Nothing scoreable" in w for w in result.warnings)
    assert result.scores[Strategy.SCREEN].portfolio_return is None


def _many(strategy: Strategy, ret: float, n: int, start: int) -> tuple[list, dict]:
    """n picks for one strategy, each returning exactly `ret`, plus their forward
    prices. Entry is 100, so exit = 100 * (1 + ret)."""
    picks = [_pick(f"{strategy.value}{start + i}", strategy=strategy) for i in range(n)]
    prices = {p.entity_id: 100.0 * (1 + ret) for p in picks}
    return picks, prices


def test_agent_beats_ev_ebit_with_a_sufficient_sample():
    agent_p, agent_fwd = _many(Strategy.AGENT, 0.12, MIN_CREDIBLE_SAMPLE, 0)
    base_p, base_fwd = _many(Strategy.EV_EBIT_DECILE, 0.04, MIN_CREDIBLE_SAMPLE, 100)
    result = evaluate(
        agent_p + base_p, {**agent_fwd, **base_fwd}, as_of_exit=EXIT, cost_bps=0.0
    )
    comp = next(c for c in result.comparisons if c.baseline == Strategy.EV_EBIT_DECILE)
    assert comp.delta == pytest.approx(0.08)
    assert "beat" in comp.verdict


def test_agent_that_trails_ev_ebit_is_called_cost_not_signal():
    agent_p, agent_fwd = _many(Strategy.AGENT, 0.02, MIN_CREDIBLE_SAMPLE, 0)
    base_p, base_fwd = _many(Strategy.EV_EBIT_DECILE, 0.09, MIN_CREDIBLE_SAMPLE, 100)
    result = evaluate(
        agent_p + base_p, {**agent_fwd, **base_fwd}, as_of_exit=EXIT, cost_bps=0.0
    )
    comp = next(c for c in result.comparisons if c.baseline == Strategy.EV_EBIT_DECILE)
    assert comp.delta < 0
    assert "cost, not signal" in comp.verdict


def test_comparison_is_insufficient_below_the_sample_bar():
    agent_p, agent_fwd = _many(Strategy.AGENT, 0.30, 3, 0)
    base_p, base_fwd = _many(Strategy.EV_EBIT_DECILE, 0.01, 3, 100)
    result = evaluate(
        agent_p + base_p, {**agent_fwd, **base_fwd}, as_of_exit=EXIT, cost_bps=0.0
    )
    comp = next(c for c in result.comparisons if c.baseline == Strategy.EV_EBIT_DECILE)
    # A 29pp lead on 3 picks is noise, and the verdict says so rather than
    # crowning the agent.
    assert "insufficient evidence" in comp.verdict


def test_labeled_observations_bridge_to_the_gbdt():
    picks = [_pick("a", reference_price=100.0, features={"roic": 0.2})]
    scored = score_picks(picks, {"a": 110.0}, cost_bps=0.0)
    obs = labeled_observations(scored)
    assert len(obs) == 1
    assert obs[0].features == {"roic": 0.2}
    assert obs[0].forward_return == pytest.approx(0.10)
