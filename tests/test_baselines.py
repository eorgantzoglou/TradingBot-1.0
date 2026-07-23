"""The three dumb baselines, and the GBDT's honest gate.

Baselines are pure functions of the screen's candidates, so they are tested
directly on hand-built ScoredCandidates -- no store, no prices fetched.
"""

from __future__ import annotations

import math

import pytest

from scout.portfolio.baselines import (
    GradientBoostedTree,
    InsufficientHistory,
    LabeledObservation,
    equal_weight_universe,
    ev_ebit_decile,
    top_composite,
)
from scout.screen.models import CohortKey, ScoredCandidate

_MFG = CohortKey("US", "US-GAAP", "manufacturing")
_SVC = CohortKey("US", "US-GAAP", "services")


def _cand(entity_id: str, *, composite=None, cohort=_MFG, **metrics) -> ScoredCandidate:
    return ScoredCandidate(
        entity_id=entity_id,
        name=f"Co {entity_id}",
        cohort=cohort,
        composite=composite,
        metric_values=dict(metrics),
    )


def test_equal_weight_universe_holds_everyone_ordered_by_composite():
    cands = [_cand("a", composite=0.1), _cand("b", composite=0.9), _cand("c", composite=None)]
    sel = equal_weight_universe(cands)
    # Everyone is held (baseline #1 = the whole universe), best composite first,
    # unrankable name last but present.
    assert [s.entity_id for s in sel] == ["b", "a", "c"]
    assert all(s.rank is None for s in sel)


def test_top_composite_takes_best_n_and_drops_unrankable():
    cands = [_cand("a", composite=0.1), _cand("b", composite=0.9), _cand("c", composite=None)]
    sel = top_composite(cands, top=2)
    assert [s.entity_id for s in sel] == ["b", "a"]
    assert [s.rank for s in sel] == [1, 2]
    # 'c' has no composite -> the screen makes no claim about it -> excluded.
    assert "c" not in {s.entity_id for s in sel}


def test_ev_ebit_decile_is_per_cohort_and_picks_cheapest():
    # Two cohorts; the decile rounds up to >=1 name per cohort, so each cohort's
    # single cheapest name is chosen.
    cands = [
        _cand("m1", cohort=_MFG, ev_ebit=8.0),
        _cand("m2", cohort=_MFG, ev_ebit=25.0),
        _cand("s1", cohort=_SVC, ev_ebit=12.0),
    ]
    sel = ev_ebit_decile(cands)
    chosen = {s.entity_id for s in sel}
    assert chosen == {"m1", "s1"}  # cheapest of each cohort
    # score = -EV/EBIT so higher is more preferred, and m1 (8) beats s1 (12).
    assert sel[0].entity_id == "m1"
    assert sel[0].score == pytest.approx(-8.0)


def test_ev_ebit_decile_is_empty_without_prices():
    # No ev_ebit anywhere (the real state without a price feed) -> no selection,
    # honestly, rather than a fabricated one.
    assert ev_ebit_decile([_cand("a"), _cand("b")]) == []


def test_ev_ebit_decile_ignores_non_positive_and_infinite():
    cands = [
        _cand("loss", ev_ebit=-5.0),          # loss-maker: not 'cheap', excluded
        _cand("inf", ev_ebit=float("inf")),   # degenerate ratio: excluded
        _cand("real", ev_ebit=10.0),
    ]
    sel = ev_ebit_decile(cands)
    assert [s.entity_id for s in sel] == ["real"]


def test_gbdt_is_insufficient_until_enough_history():
    tree = GradientBoostedTree(min_train_rows=200)
    with pytest.raises(InsufficientHistory, match="200"):
        tree.select([_cand("a", roic=0.1)], history=[], top=5)


def test_gbdt_trains_and_ranks_when_history_is_deep_enough():
    lgb = pytest.importorskip("lightgbm")  # optional 'gbdt' extra
    assert lgb is not None

    # A learnable signal: higher roic -> higher forward return.
    history = [
        LabeledObservation(features={"roic": r / 100.0}, forward_return=r / 100.0)
        for r in range(1, 121)
    ] * 2  # 240 rows, over the default 200 threshold
    tree = GradientBoostedTree(min_train_rows=200, num_boost_round=20)
    sel = tree.select(
        [_cand("low", roic=0.05), _cand("high", roic=1.10)],
        history=history,
        top=2,
    )
    assert [s.entity_id for s in sel] == ["high", "low"]
    assert all(math.isfinite(s.score) for s in sel)
