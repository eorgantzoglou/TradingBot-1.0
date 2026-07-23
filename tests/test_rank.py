"""Tests for scout.screen.rank -- within-cohort z-score ranking.

Every cohort here is hand-built so the expected direction, block score or
ordering can be reasoned about by hand. The point is not to pin exact z-values
(those follow from the winsorized mean/std) but to nail the things that would
silently corrupt a ranking: the sign flips per metric, missing-block
renormalization, small-cohort handling, the std==0 and inf edge cases, and the
final ordering.
"""

from __future__ import annotations

import math

from scout.screen.models import CohortKey
from scout.screen.rank import RankInput, RankWeights, rank

_COHORT = CohortKey(country="US", accounting_standard="US-GAAP", sector="manufacturing")


def _by_id(results):
    return {c.entity_id: c for c in results}


def test_direction_signs_per_block():
    """Highest gp -> best quality; LOWEST ev_ebit -> best cheap; LOWEST beneish -> best safety.

    Each entity carries exactly one metric per block so a block score is that
    single metric's (signed) z, isolating the direction of each map entry. The
    winners are spread across different entities so nothing wins by coincidence.
    """
    inputs = [
        # gp best here (0.4), but worst ev_ebit and worst beneish.
        RankInput("E1", "E1", _COHORT, {"gp_to_assets": 0.4, "ev_ebit": 20.0, "beneish_m": 0.0}),
        # ev_ebit best here (5 -> lower is cheaper).
        RankInput("E2", "E2", _COHORT, {"gp_to_assets": 0.1, "ev_ebit": 5.0, "beneish_m": -2.0}),
        # beneish best here (-3 -> more negative is safer).
        RankInput("E3", "E3", _COHORT, {"gp_to_assets": 0.2, "ev_ebit": 15.0, "beneish_m": -3.0}),
        RankInput("E4", "E4", _COHORT, {"gp_to_assets": 0.3, "ev_ebit": 10.0, "beneish_m": -1.0}),
    ]

    r = _by_id(rank(inputs))

    best_quality = max(r.values(), key=lambda c: c.quality)
    best_cheap = max(r.values(), key=lambda c: c.cheap)
    best_safety = max(r.values(), key=lambda c: c.safety)

    assert best_quality.entity_id == "E1"  # highest gp_to_assets
    assert best_cheap.entity_id == "E2"  # LOWEST ev_ebit wins after the sign flip
    assert best_safety.entity_id == "E3"  # LOWEST beneish_m wins after the sign flip


def test_lower_is_better_metrics_flip_sign():
    """A pure lower-is-better metric ranks the smallest value highest."""
    inputs = [
        RankInput("hi", None, _COHORT, {"accruals": 0.5}),
        RankInput("mid", None, _COHORT, {"accruals": 0.0}),
        RankInput("lo", None, _COHORT, {"accruals": -0.5}),
    ]
    r = _by_id(rank(inputs))
    # Lowest accruals is best earnings quality -> highest quality block.
    assert r["lo"].quality > r["mid"].quality > r["hi"].quality


def test_missing_cheap_block_is_ranked_not_last():
    """A name with no price (no cheap metrics) is scored on quality+safety and can beat a peer."""
    strong_no_price = RankInput(
        "noprice", None, _COHORT, {"gp_to_assets": 0.9, "beneish_m": -4.0}
    )
    inputs = [
        strong_no_price,
        RankInput("weak", None, _COHORT, {"gp_to_assets": 0.1, "beneish_m": 0.0, "ev_ebit": 8.0}),
        RankInput("mid", None, _COHORT, {"gp_to_assets": 0.5, "beneish_m": -1.0, "ev_ebit": 6.0}),
    ]

    r = _by_id(rank(inputs))
    noprice = r["noprice"]

    assert noprice.cheap is None  # no price -> no cheap block
    assert noprice.composite is not None  # still scored...
    assert noprice.composite > r["weak"].composite  # ...and not forced to the bottom
    assert any("renormalized" in w for w in noprice.warnings)


def test_small_cohort_emitted_unranked():
    """Below min_cohort: composite/blocks None, a warning, but still emitted."""
    inputs = [
        RankInput("A", None, _COHORT, {"gp_to_assets": 0.3}),
        RankInput("B", None, _COHORT, {"gp_to_assets": 0.4}),
    ]
    results = rank(inputs, min_cohort=3)

    assert len(results) == 2  # nothing dropped
    for c in results:
        assert c.composite is None
        assert c.cheap is None and c.quality is None and c.safety is None
        assert any("not enough peers" in w for w in c.warnings)


def test_identical_cohort_all_zero_no_crash():
    """std==0 everywhere -> every z is 0, composites equal 0, no division blow-up."""
    metrics = {"gp_to_assets": 0.2, "ev_ebit": 10.0, "beneish_m": -1.0}
    inputs = [RankInput(f"E{i}", None, _COHORT, dict(metrics)) for i in range(4)]

    results = rank(inputs)

    for c in results:
        assert c.cheap == 0.0
        assert c.quality == 0.0
        assert c.safety == 0.0
        assert c.composite == 0.0


def test_winsorization_bounds_outlier():
    """A single 100x outlier must not dominate: all z-scores stay sane after clipping.

    Twenty peers on gp_to_assets = 1..20 plus one absurd 10000. With p95 clipping
    the outlier is pulled down to the 95th-percentile value, so no entity -- the
    outlier included -- ends up with a runaway z-score. Without winsorization the
    outlier's z would exceed 4 and crush everyone else toward zero.
    """
    inputs = [
        RankInput(f"N{i}", None, _COHORT, {"gp_to_assets": float(i)}) for i in range(1, 21)
    ]
    inputs.append(RankInput("OUT", None, _COHORT, {"gp_to_assets": 10_000.0}))

    r = _by_id(rank(inputs))

    for c in r.values():
        assert c.quality is not None
        assert abs(c.quality) < 2.5  # bounded; would be > 4 for OUT if unwinsorized


def test_infinite_cash_runway_capped():
    """inf cash_runway (self-funding) is capped, never producing nan/inf composites."""
    inputs = [
        RankInput("self_funded", None, _COHORT, {"cash_runway_months": math.inf}),
        RankInput("ok", None, _COHORT, {"cash_runway_months": 24.0}),
        RankInput("tight", None, _COHORT, {"cash_runway_months": 6.0}),
    ]

    r = _by_id(rank(inputs))

    for c in r.values():
        assert c.composite is not None
        assert math.isfinite(c.composite)
        assert math.isfinite(c.safety)
    # A self-funding firm should still rank at the top of the safety block.
    assert r["self_funded"].safety == max(c.safety for c in r.values())


def test_sorting_desc_with_none_last():
    """Ranked names come back best-composite-first; unrankable (None) names last."""
    big = [
        RankInput("best", None, _COHORT, {"gp_to_assets": 0.9}),
        RankInput("worst", None, _COHORT, {"gp_to_assets": 0.1}),
        RankInput("middle", None, _COHORT, {"gp_to_assets": 0.5}),
    ]
    tiny_cohort = CohortKey(country="JP", accounting_standard="JGAAP", sector="services")
    tiny = [RankInput("lonely", None, tiny_cohort, {"gp_to_assets": 0.3})]

    results = rank(big + tiny)

    composites = [c.composite for c in results]
    ranked = [c for c in composites if c is not None]
    assert ranked == sorted(ranked, reverse=True)  # descending
    assert composites[-1] is None  # the tiny-cohort name is last
    assert results[0].entity_id == "best"


def test_custom_weights_shift_composite():
    """Weights are honoured: a safety-only weighting ranks purely on the safety block."""
    inputs = [
        RankInput("A", None, _COHORT, {"gp_to_assets": 0.9, "beneish_m": 0.0}),
        RankInput("B", None, _COHORT, {"gp_to_assets": 0.1, "beneish_m": -5.0}),
        RankInput("C", None, _COHORT, {"gp_to_assets": 0.5, "beneish_m": -1.0}),
    ]
    weights = RankWeights(cheap=0.0, quality=0.0, safety=1.0)

    r = _by_id(rank(inputs, weights=weights))

    # B has the safest (most negative) beneish, so it must win under safety-only weights.
    assert r["B"].composite == max(c.composite for c in r.values())
