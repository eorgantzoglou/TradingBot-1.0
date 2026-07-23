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
    scope_to_vintage,
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
    # Pin the quantiles too (linear interpolation, numpy 'linear' method), so a
    # regression in _percentile can't slip through this test.
    assert dist.p10 == pytest.approx(-0.16)  # 0.10*(5-1)=0.4 between -0.20 and -0.10
    assert dist.p25 == pytest.approx(-0.10)  # 0.25*4=1.0 -> exactly the 2nd value
    assert dist.p75 == pytest.approx(0.10)
    assert dist.p90 == pytest.approx(0.16)


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


def test_score_picks_skips_non_finite_and_non_positive_prices():
    """H1 defensive backstop: a bad price must be ungradeable, never poison the
    distribution with inf/nan (nan == 0 is False, so a naive guard misses it)."""
    picks = [
        _pick("nan_ref", reference_price=float("nan")),
        _pick("inf_ref", reference_price=float("inf")),
        _pick("neg_ref", reference_price=-100.0),
        _pick("zero_ref", reference_price=0.0),
        _pick("ok", reference_price=100.0),
        _pick("nan_fwd", reference_price=100.0),
        _pick("neg_fwd", reference_price=100.0),
    ]
    fwd = {
        "nan_ref": 100.0, "inf_ref": 100.0, "neg_ref": 100.0, "zero_ref": 100.0,
        "ok": 110.0, "nan_fwd": float("nan"), "neg_fwd": -50.0,
    }
    scored = score_picks(picks, fwd, cost_bps=0.0)
    assert [s.pick.entity_id for s in scored] == ["ok"]


def test_evaluate_dedups_same_pick_id_keeping_last():
    """H2: a same-day re-pick (identical pick_id) must not double-count; the last
    occurrence supersedes."""
    first = _pick("111", reference_price=100.0)
    second = _pick("111", reference_price=50.0)  # same as_of+strategy+entity => same id
    assert first.pick_id == second.pick_id
    result = evaluate([first, second], {"111": 110.0}, as_of_exit=EXIT, cost_bps=0.0)
    score = result.scores[Strategy.SCREEN]
    assert score.n_scored == 1                      # counted once, not twice
    assert score.portfolio_return == pytest.approx(1.2)  # 110/50 - 1, the LAST ref price


def test_scope_to_vintage_defaults_to_latest_and_warns():
    """H2: scoring must not pool picks from several dates against one price snapshot;
    default to the latest vintage and say so."""
    old = PaperPick(
        as_of=date(2026, 7, 1), strategy=Strategy.SCREEN, entity_id="a", name=None,
        cohort="", reference_price=100.0, currency=None, weight=1.0, rank=None,
        score=None, vetoed=None, features={}, run_id="r1",
    )
    new = _pick("b")  # as_of 2026-07-23
    scoped, note = scope_to_vintage([old, new])
    assert [p.entity_id for p in scoped] == ["b"]  # only the latest date
    assert note is not None and "spans" in note


def test_scope_to_vintage_explicit_vintage_and_run_id():
    old = PaperPick(
        as_of=date(2026, 7, 1), strategy=Strategy.SCREEN, entity_id="a", name=None,
        cohort="", reference_price=100.0, currency=None, weight=1.0, rank=None,
        score=None, vetoed=None, features={}, run_id="r1",
    )
    new = _pick("b")
    by_date, _ = scope_to_vintage([old, new], vintage=date(2026, 7, 1))
    assert [p.entity_id for p in by_date] == ["a"]
    by_run, _ = scope_to_vintage([old, new], run_id="r1")
    assert [p.entity_id for p in by_run] == ["a"]
    empty, note = scope_to_vintage([old, new], run_id="does-not-exist")
    assert empty == [] and note is not None


def test_verdict_reports_median_and_flags_sign_disagreement():
    """H3: a book lifted above the baseline on the MEAN by one big winner, while its
    MEDIAN pick trails, must read as inconclusive -- not as 'the agent beat it'."""
    # Agent: 29 flat picks + 1 huge winner -> mean high, median 0.
    agent = [_pick(f"a{i}", strategy=Strategy.AGENT) for i in range(29)]
    agent.append(_pick("a_win", strategy=Strategy.AGENT))
    agent_fwd = {p.entity_id: 100.0 for p in agent}      # 0% for the 29
    agent_fwd["a_win"] = 400.0                            # +300% winner
    # Baseline: 30 flat picks at +5%.
    base = [_pick(f"b{i}", strategy=Strategy.EV_EBIT_DECILE) for i in range(30)]
    base_fwd = {p.entity_id: 105.0 for p in base}

    result = evaluate(agent + base, {**agent_fwd, **base_fwd}, as_of_exit=EXIT, cost_bps=0.0)
    comp = next(c for c in result.comparisons if c.baseline == Strategy.EV_EBIT_DECILE)
    assert comp.delta is not None and comp.delta > 0        # mean: agent ahead
    assert comp.median_delta is not None and comp.median_delta < 0  # median: agent behind
    assert comp.signs_disagree
    assert "mixed" in comp.verdict and "inconclusive" in comp.verdict
    # Crucially, a skew-driven mean win is NOT reported as the agent beating the bar.
    assert "beat" not in comp.verdict


def test_insufficient_comparison_has_no_delta():
    """M1: below the sample bar, delta must be None so no reader charts noise as real."""
    agent_p, agent_fwd = _many(Strategy.AGENT, 0.30, 3, 0)
    base_p, base_fwd = _many(Strategy.EV_EBIT_DECILE, 0.01, 3, 100)
    result = evaluate(agent_p + base_p, {**agent_fwd, **base_fwd}, as_of_exit=EXIT, cost_bps=0.0)
    comp = next(c for c in result.comparisons if c.baseline == Strategy.EV_EBIT_DECILE)
    assert comp.delta is None
    assert comp.median_delta is None


def test_sample_warning_keys_off_smallest_book_not_largest():
    """M2: a big universe book reaching 30 must not silence the warning while a thin
    book is shown next to it."""
    big, big_fwd = _many(Strategy.UNIVERSE_EW, 0.05, MIN_CREDIBLE_SAMPLE, 0)
    thin, thin_fwd = _many(Strategy.SCREEN, 0.05, 4, 500)
    result = evaluate(big + thin, {**big_fwd, **thin_fwd}, as_of_exit=EXIT, cost_bps=0.0)
    assert any("too few" in w for w in result.warnings)


def test_tie_reads_as_matched_not_trailed():
    agent_p, agent_fwd = _many(Strategy.AGENT, 0.05, MIN_CREDIBLE_SAMPLE, 0)
    base_p, base_fwd = _many(Strategy.UNIVERSE_EW, 0.05, MIN_CREDIBLE_SAMPLE, 100)
    result = evaluate(agent_p + base_p, {**agent_fwd, **base_fwd}, as_of_exit=EXIT, cost_bps=0.0)
    comp = next(c for c in result.comparisons if c.baseline == Strategy.UNIVERSE_EW)
    assert "matched" in comp.verdict
    assert "trailed" not in comp.verdict


def test_empty_baseline_message_does_not_blame_the_agent():
    """LOW: when the agent book is fine but a baseline had nothing gradeable, the
    message must name the baseline, not claim there is no agent book."""
    agent_p, agent_fwd = _many(Strategy.AGENT, 0.05, 3, 0)
    # An EV/EBIT book that is entirely ungradeable (no forward prices for it).
    base = [_pick(f"b{i}", strategy=Strategy.EV_EBIT_DECILE) for i in range(3)]
    result = evaluate(agent_p + base, agent_fwd, as_of_exit=EXIT, cost_bps=0.0)
    comp = next(c for c in result.comparisons if c.baseline == Strategy.EV_EBIT_DECILE)
    assert "no agent book" not in comp.verdict
    assert "ev_ebit_decile" in comp.verdict


def test_labeled_observations_bridge_to_the_gbdt():
    picks = [_pick("a", reference_price=100.0, features={"roic": 0.2})]
    scored = score_picks(picks, {"a": 110.0}, cost_bps=0.0)
    obs = labeled_observations(scored)
    assert len(obs) == 1
    assert obs[0].features == {"roic": 0.2}
    assert obs[0].forward_return == pytest.approx(0.10)
