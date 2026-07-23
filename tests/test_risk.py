"""Tests for scout.metrics.risk -- the fraud/bankruptcy/dilution/runway gates.

Two layers, mirroring test_normalize.py:

  * UNIT tests on hand-built snapshots where every input was chosen so the
    expected result can be computed by hand. The Beneish worked example below
    derives all eight indices explicitly in comments so the -1.36 M-Score can be
    cross-checked without trusting the code under test.

  * GOLDEN tests against the REAL archived filings (3M's 10-Q under us-gaap, a
    Ukrainian ESEF issuer under ifrs-full), building the snapshot exactly as
    test_normalize.py does. These pin the Altman Z and cash-runway arithmetic to
    figures a human verified against the filing, so buggy code cannot agree with
    a fabricated fixture -- the failure mode this project shipped once before.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pytest

from scout.fundamentals.concepts import Concept
from scout.fundamentals.models import CanonicalFact, EntityRef, FundamentalsSnapshot
from scout.metrics.risk import (
    altman_z_double_prime,
    altman_zone,
    beneish_flags_manipulation,
    beneish_from_history,
    beneish_m_score,
    cash_runway_months,
    dilution_from_history,
    is_cash_critical,
    is_dilution_machine,
    share_issuance_rate,
)

# --------------------------------------------------------------------------- #
# Snapshot builder for hand-built cases
# --------------------------------------------------------------------------- #

_ENTITY = EntityRef(source="sec", entity_id="1", identifier_scheme="cik", name="Test Co")


def _snap(
    values: dict[Concept, float],
    *,
    period_end: date,
    fiscal_period: str | None = "FY",
) -> FundamentalsSnapshot:
    """A snapshot carrying exactly the given concept values.

    Only `value` is load-bearing for these metrics (metrics read
    `snapshot.get(concept)`); the rest of each CanonicalFact is filled with inert
    placeholders so the frozen dataclass is well-formed.
    """
    facts = {
        concept: CanonicalFact(
            entity_id="1",
            concept=concept,
            value=value,
            currency="USD",
            period_end=period_end,
            period_start=None,
            fiscal_year=period_end.year,
            fiscal_period=fiscal_period,
            accession="unit-acc",
            source_concept=f"test:{concept.value}",
            taxonomy="us-gaap",
        )
        for concept, value in values.items()
    }
    return FundamentalsSnapshot(
        entity=_ENTITY,
        period_end=period_end,
        fiscal_year=period_end.year,
        fiscal_period=fiscal_period,
        currency="USD",
        taxonomy="us-gaap",
        accession="unit-acc",
        filing_date=None,
        facts=facts,
    )


_PRIOR_END = date(2024, 12, 31)
_CURRENT_END = date(2025, 12, 31)  # exactly one fiscal year later


# --------------------------------------------------------------------------- #
# Beneish M-Score -- fully worked example
# --------------------------------------------------------------------------- #
#
# Prior (p) and current (t) chosen for clean hand arithmetic:
#
#   REVENUE:            p=1000   t=1500   -> SGI  = 1500/1000            = 1.50000
#   RECEIVABLES:        p=100    t=200    -> DSRI = (200/1500)/(100/1000)= 1.33333
#   COST_OF_REVENUE:    p=600    t=1000
#     gross margin      p=400/1000=0.4    t=500/1500=0.33333
#                                          -> GMI = 0.4/0.33333          = 1.20000
#   CURRENT_ASSETS:     p=500    t=600
#   PPE_NET:            p=300    t=500
#   TOTAL_ASSETS:       p=1000   t=1500
#     asset quality     p=1-(500+300)/1000=0.2   t=1-(600+500)/1500=0.26667
#                                          -> AQI = 0.26667/0.2          = 1.33333
#   D&A:                p=60     t=80
#     dep rate          p=60/360=0.16667  t=80/580=0.13793
#                                          -> DEPI= 0.16667/0.13793      = 1.20833
#   SGA_EXPENSE:        p=150    t=300
#     sga/rev           p=0.15            t=0.20
#                                          -> SGAI= 0.20/0.15            = 1.33333
#   NET_INCOME t=100, CASH_FROM_OPERATIONS t=50, TOTAL_ASSETS t=1500
#                       -> TATA = (100-50)/1500                          = 0.03333
#   CURRENT_LIABILITIES:p=200   t=300 ; LONG_TERM_DEBT: p=100  t=150
#     leverage          p=(200+100)/1000=0.30  t=(300+150)/1500=0.30
#                                          -> LVGI = 0.30/0.30           = 1.00000
#
#   M = -4.84 + 0.920*1.33333 + 0.528*1.20 + 0.404*1.33333 + 0.892*1.50
#            + 0.115*1.20833 - 0.172*1.33333 + 4.679*0.03333 - 0.327*1.00
#     = -1.3645   (>-1.78 -> flags manipulation)

_BENEISH_PRIOR = {
    Concept.REVENUE: 1000.0,
    Concept.RECEIVABLES: 100.0,
    Concept.COST_OF_REVENUE: 600.0,
    Concept.CURRENT_ASSETS: 500.0,
    Concept.PPE_NET: 300.0,
    Concept.TOTAL_ASSETS: 1000.0,
    Concept.DEPRECIATION_AMORTIZATION: 60.0,
    Concept.SGA_EXPENSE: 150.0,
    Concept.CURRENT_LIABILITIES: 200.0,
    Concept.LONG_TERM_DEBT: 100.0,
}
_BENEISH_CURRENT = {
    Concept.REVENUE: 1500.0,
    Concept.RECEIVABLES: 200.0,
    Concept.COST_OF_REVENUE: 1000.0,
    Concept.CURRENT_ASSETS: 600.0,
    Concept.PPE_NET: 500.0,
    Concept.TOTAL_ASSETS: 1500.0,
    Concept.DEPRECIATION_AMORTIZATION: 80.0,
    Concept.SGA_EXPENSE: 300.0,
    Concept.CURRENT_LIABILITIES: 300.0,
    Concept.LONG_TERM_DEBT: 150.0,
    Concept.NET_INCOME: 100.0,
    Concept.CASH_FROM_OPERATIONS: 50.0,
}


class TestBeneish:
    def test_worked_example_indices_and_m_score(self) -> None:
        prior = _snap(_BENEISH_PRIOR, period_end=_PRIOR_END)
        current = _snap(_BENEISH_CURRENT, period_end=_CURRENT_END)
        result = beneish_m_score(current, prior)

        assert result.ok
        # Each index to 4 dp against the hand derivation above.
        assert result.inputs["DSRI"] == pytest.approx(1.33333, abs=1e-4)
        assert result.inputs["GMI"] == pytest.approx(1.20000, abs=1e-4)
        assert result.inputs["AQI"] == pytest.approx(1.33333, abs=1e-4)
        assert result.inputs["SGI"] == pytest.approx(1.50000, abs=1e-4)
        assert result.inputs["DEPI"] == pytest.approx(1.20833, abs=1e-4)
        assert result.inputs["SGAI"] == pytest.approx(1.33333, abs=1e-4)
        assert result.inputs["TATA"] == pytest.approx(0.03333, abs=1e-4)
        assert result.inputs["LVGI"] == pytest.approx(1.00000, abs=1e-4)
        # M to 3 dp.
        assert result.value == pytest.approx(-1.3645, abs=1e-3)

    def test_worked_example_flags_manipulation(self) -> None:
        prior = _snap(_BENEISH_PRIOR, period_end=_PRIOR_END)
        current = _snap(_BENEISH_CURRENT, period_end=_CURRENT_END)
        result = beneish_m_score(current, prior)
        assert beneish_flags_manipulation(result.value) is True

    def test_flag_threshold_both_sides(self) -> None:
        # -1.78 is the cutoff; strictly greater flags.
        assert beneish_flags_manipulation(-1.77) is True
        assert beneish_flags_manipulation(-1.79) is False
        assert beneish_flags_manipulation(-1.78) is False  # boundary is not a flag

    def test_missing_depi_defaults_to_neutral_with_warning(self) -> None:
        # Drop current D&A -> dep rate_t is undefined -> DEPI cannot be computed.
        current = dict(_BENEISH_CURRENT)
        del current[Concept.DEPRECIATION_AMORTIZATION]
        result = beneish_m_score(
            _snap(current, period_end=_CURRENT_END),
            _snap(_BENEISH_PRIOR, period_end=_PRIOR_END),
        )
        assert result.ok  # a missing secondary index must not sink the score
        assert result.inputs["DEPI"] == 1.0
        assert any("DEPI" in w for w in result.warnings), result.warnings

    def test_missing_sgi_returns_missing(self) -> None:
        # Drop prior revenue -> SGI (a CORE index) cannot be computed -> missing.
        prior = dict(_BENEISH_PRIOR)
        del prior[Concept.REVENUE]
        result = beneish_m_score(
            _snap(_BENEISH_CURRENT, period_end=_CURRENT_END),
            _snap(prior, period_end=_PRIOR_END),
        )
        assert not result.ok
        assert result.value is None
        assert result.reason is not None

    def test_non_annual_pair_is_missing(self) -> None:
        result = beneish_m_score(
            _snap(_BENEISH_CURRENT, period_end=_CURRENT_END, fiscal_period="Q4"),
            _snap(_BENEISH_PRIOR, period_end=_PRIOR_END),
        )
        assert not result.ok

    def test_two_year_gap_is_rejected(self) -> None:
        result = beneish_m_score(
            _snap(_BENEISH_CURRENT, period_end=date(2026, 12, 31)),
            _snap(_BENEISH_PRIOR, period_end=date(2024, 12, 31)),
        )
        assert not result.ok

    def test_from_history_pairs_two_recent_annuals(self) -> None:
        prior = _snap(_BENEISH_PRIOR, period_end=_PRIOR_END)
        current = _snap(_BENEISH_CURRENT, period_end=_CURRENT_END)
        interim = _snap({Concept.REVENUE: 1.0}, period_end=date(2025, 6, 30), fiscal_period="YTD6")
        result = beneish_from_history([prior, interim, current])
        assert result.ok
        assert result.value == pytest.approx(-1.3645, abs=1e-3)


# --------------------------------------------------------------------------- #
# Altman Z''
# --------------------------------------------------------------------------- #


class TestAltman:
    def test_zone_boundaries(self) -> None:
        assert altman_zone(2.61) == "safe"
        assert altman_zone(2.6) == "grey"  # boundary is not "safe"
        assert altman_zone(1.1) == "grey"
        assert altman_zone(1.09) == "distress"

    def test_safe_zone_case(self) -> None:
        # WC/TA=.2, RE/TA=.4, EBIT/TA=.15, EQ/L=2.0
        #   Z = 3.25 + 6.56*.2 + 3.26*.4 + 6.72*.15 + 1.05*2.0
        #     = 3.25 + 1.312 + 1.304 + 1.008 + 2.10 = 8.974
        snap = _snap(
            {
                Concept.CURRENT_ASSETS: 300.0,
                Concept.CURRENT_LIABILITIES: 100.0,  # WC = 200, /TA(1000)=0.2
                Concept.RETAINED_EARNINGS: 400.0,
                Concept.OPERATING_INCOME: 150.0,
                Concept.TOTAL_ASSETS: 1000.0,
                Concept.TOTAL_EQUITY: 600.0,
                Concept.TOTAL_LIABILITIES: 300.0,  # EQ/L = 2.0
                Concept.REVENUE: 800.0,  # sales/TA = 0.8 -> no pre-revenue warning
            },
            period_end=_CURRENT_END,
        )
        result = altman_z_double_prime(snap)
        assert result.ok
        assert result.value == pytest.approx(8.974, abs=1e-3)
        assert altman_zone(result.value) == "safe"
        assert not result.warnings

    def test_distress_zone_case(self) -> None:
        # Negative WC and a big accumulated deficit push Z below 1.1.
        #   WC/TA = -200/1000 = -0.2 ; RE/TA = -500/1000 = -0.5
        #   EBIT/TA = -100/1000 = -0.1 ; EQ/L = 50/950 = 0.05263
        #   Z = 3.25 + 6.56*-0.2 + 3.26*-0.5 + 6.72*-0.1 + 1.05*0.05263
        #     = 3.25 - 1.312 - 1.63 - 0.672 + 0.05526 = -0.30874
        snap = _snap(
            {
                Concept.CURRENT_ASSETS: 100.0,
                Concept.CURRENT_LIABILITIES: 300.0,  # WC = -200
                Concept.RETAINED_EARNINGS: -500.0,
                Concept.OPERATING_INCOME: -100.0,
                Concept.TOTAL_ASSETS: 1000.0,
                Concept.TOTAL_EQUITY: 50.0,
                Concept.TOTAL_LIABILITIES: 950.0,
                Concept.REVENUE: 700.0,
            },
            period_end=_CURRENT_END,
        )
        result = altman_z_double_prime(snap)
        assert result.ok
        assert result.value == pytest.approx(-0.30874, abs=1e-3)
        assert altman_zone(result.value) == "distress"

    def test_pre_revenue_warns_but_still_computes(self) -> None:
        snap = _snap(
            {
                Concept.CURRENT_ASSETS: 300.0,
                Concept.CURRENT_LIABILITIES: 100.0,
                Concept.RETAINED_EARNINGS: -400.0,
                Concept.OPERATING_INCOME: -50.0,
                Concept.TOTAL_ASSETS: 1000.0,
                Concept.TOTAL_EQUITY: 600.0,
                Concept.TOTAL_LIABILITIES: 300.0,
                # No REVENUE at all -> pre-revenue warning.
            },
            period_end=_CURRENT_END,
        )
        result = altman_z_double_prime(snap)
        assert result.ok
        assert any("pre-revenue" in w for w in result.warnings), result.warnings

    def test_missing_input_is_missing(self) -> None:
        snap = _snap(
            {
                Concept.CURRENT_ASSETS: 300.0,
                Concept.CURRENT_LIABILITIES: 100.0,
                Concept.RETAINED_EARNINGS: 400.0,
                # OPERATING_INCOME absent
                Concept.TOTAL_ASSETS: 1000.0,
                Concept.TOTAL_EQUITY: 600.0,
                Concept.TOTAL_LIABILITIES: 300.0,
            },
            period_end=_CURRENT_END,
        )
        result = altman_z_double_prime(snap)
        assert not result.ok
        assert result.value is None


# --------------------------------------------------------------------------- #
# Dilution
# --------------------------------------------------------------------------- #


class TestDilution:
    def test_fifty_percent_issuance_flags(self) -> None:
        prior = _snap({Concept.SHARES_OUTSTANDING: 1_000_000.0}, period_end=_PRIOR_END)
        current = _snap({Concept.SHARES_OUTSTANDING: 1_500_000.0}, period_end=_CURRENT_END)
        result = share_issuance_rate(current, prior)
        assert result.ok
        assert result.value == pytest.approx(0.50)
        assert is_dilution_machine(result.value) is True

    def test_buyback_is_negative_and_not_flagged(self) -> None:
        prior = _snap({Concept.SHARES_OUTSTANDING: 1_000_000.0}, period_end=_PRIOR_END)
        current = _snap({Concept.SHARES_OUTSTANDING: 900_000.0}, period_end=_CURRENT_END)
        result = share_issuance_rate(current, prior)
        assert result.ok
        assert result.value == pytest.approx(-0.10)
        assert is_dilution_machine(result.value) is False

    def test_twenty_percent_boundary(self) -> None:
        # Exactly 20% is NOT a dilution machine (strictly greater than 0.20).
        assert is_dilution_machine(0.20) is False
        assert is_dilution_machine(0.2001) is True

    def test_from_history(self) -> None:
        prior = _snap({Concept.SHARES_OUTSTANDING: 1_000_000.0}, period_end=_PRIOR_END)
        current = _snap({Concept.SHARES_OUTSTANDING: 1_300_000.0}, period_end=_CURRENT_END)
        result = dilution_from_history([current, prior])
        assert result.ok
        assert result.value == pytest.approx(0.30)


# --------------------------------------------------------------------------- #
# Cash runway
# --------------------------------------------------------------------------- #


class TestCashRunway:
    def test_burning_firm_runway_in_months(self) -> None:
        # FY burn of 1200 -> monthly 100; 600 cash -> 6 months.
        snap = _snap(
            {Concept.CASH_AND_EQUIVALENTS: 600.0, Concept.CASH_FROM_OPERATIONS: -1200.0},
            period_end=_CURRENT_END,
        )
        result = cash_runway_months(snap)
        assert result.ok
        assert result.value == pytest.approx(6.0)
        assert is_cash_critical(result.value) is True

    def test_interim_half_year_burn(self) -> None:
        # A half-year interim: 600 burn over 6 months -> monthly 100; 300 cash -> 3 months.
        snap = _snap(
            {Concept.CASH_AND_EQUIVALENTS: 300.0, Concept.CASH_FROM_OPERATIONS: -600.0},
            period_end=date(2025, 6, 30),
            fiscal_period="YTD6",
        )
        result = cash_runway_months(snap)
        assert result.ok
        assert result.value == pytest.approx(3.0)

    def test_cash_flow_positive_is_infinite_runway(self) -> None:
        snap = _snap(
            {Concept.CASH_AND_EQUIVALENTS: 100.0, Concept.CASH_FROM_OPERATIONS: 500.0},
            period_end=_CURRENT_END,
        )
        result = cash_runway_months(snap)
        assert result.ok
        assert result.value == math.inf
        assert any("not burning" in w for w in result.warnings), result.warnings
        assert is_cash_critical(result.value) is False

    def test_critical_flag_boundary(self) -> None:
        assert is_cash_critical(11.9) is True
        assert is_cash_critical(12.0) is False  # exactly a year is not critical

    def test_missing_cash_is_missing(self) -> None:
        snap = _snap({Concept.CASH_FROM_OPERATIONS: -100.0}, period_end=_CURRENT_END)
        assert not cash_runway_months(snap).ok


# --------------------------------------------------------------------------- #
# GOLDEN: real archived filings (built exactly as test_normalize.py does)
# --------------------------------------------------------------------------- #

from scout.fundamentals.normalize import normalize_filing  # noqa: E402
from scout.fundamentals.parse.base import ParsedFiling  # noqa: E402
from scout.fundamentals.parse.esef import EsefJsonParser  # noqa: E402
from scout.fundamentals.parse.sec import SecXbrlParser  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SEC_PATH = (
    _REPO_ROOT
    / "data" / "archive" / "sec" / "2026" / "07" / "21"
    / "0000066740-26-000246" / "v1" / "0000066740-26-000246.txt"
)
_SEC_ACCESSION = "0000066740-26-000246"

_ESEF_PATH = (
    _REPO_ROOT
    / "data" / "archive" / "esef" / "2026" / "07" / "21"
    / "25825" / "v1" / "25825.json"
)
_ESEF_ACCESSION = "25825"

_needs_sec = pytest.mark.skipif(
    not _SEC_PATH.exists(), reason=f"real SEC archive not present at {_SEC_PATH}"
)
_needs_esef = pytest.mark.skipif(
    not _ESEF_PATH.exists(), reason=f"real ESEF archive not present at {_ESEF_PATH}"
)


@pytest.fixture(scope="session")
def snapshot_3m() -> FundamentalsSnapshot:
    payload = _SEC_PATH.read_bytes()
    parsed: ParsedFiling = SecXbrlParser().parse(
        payload, accession=_SEC_ACCESSION, entity_hint={"cik": "66740"}
    )
    entity = EntityRef(
        source="sec", entity_id="66740", identifier_scheme="cik", name=parsed.entity_name
    )
    snap = normalize_filing(parsed, entity)
    assert snap is not None
    return snap


@pytest.fixture(scope="session")
def snapshot_esef() -> FundamentalsSnapshot:
    payload = _ESEF_PATH.read_bytes()
    parsed: ParsedFiling = EsefJsonParser().parse(
        payload, accession=_ESEF_ACCESSION, entity_hint={"lei": "44356194", "country": "UA"}
    )
    entity = EntityRef(
        source="esef",
        entity_id=parsed.entity_id,
        identifier_scheme="national_id",
        country="UA",
        name=parsed.entity_name,
    )
    snap = normalize_filing(parsed, entity)
    assert snap is not None
    return snap


@_needs_sec
class TestGolden3M:
    def test_altman_is_in_safe_zone(self, snapshot_3m: FundamentalsSnapshot) -> None:
        # Hand-verified figures from the filing:
        #   WC = 14,112M - 11,379M = 2,733M ; RE = 38,633M ; EBIT = 2,381M
        #   TA = 34,924M ; EQ = 2,952M ; LIAB = 31,919M
        #   Z'' = 3.25 + 6.56*(2733/34924) + 3.26*(38633/34924)
        #             + 6.72*(2381/34924) + 1.05*(2952/31919) ~= 7.925
        result = altman_z_double_prime(snapshot_3m)
        assert result.ok
        assert result.value == pytest.approx(7.925, abs=0.01)
        assert result.value > 2.6
        assert altman_zone(result.value) == "safe"

    def test_cash_runway_is_infinite(self, snapshot_3m: FundamentalsSnapshot) -> None:
        # 3M's operating cash flow is +1,560M -> self-funding -> inf runway.
        result = cash_runway_months(snapshot_3m)
        assert result.ok
        assert result.value == math.inf
        assert is_cash_critical(result.value) is False


@_needs_esef
class TestGoldenEsef:
    def test_altman_computes(self, snapshot_esef: FundamentalsSnapshot) -> None:
        # Hand-verified: WC = 7,262 - 22,542 = -15,280 (negative); RE = 100,304;
        #   EBIT = 7,822 ; TA = 152,914 ; EQ = 102,879 ; LIAB = 50,035 (thousands).
        #   The very high equity/liabilities term (~2.06) lifts Z despite negative WC.
        result = altman_z_double_prime(snapshot_esef)
        assert result.ok
        assert result.value == pytest.approx(7.236, abs=0.01)

    def test_cash_runway_is_tiny_and_critical(self, snapshot_esef: FundamentalsSnapshot) -> None:
        # Burning 11,166 over the year with only 60 of cash -> ~0.06 months left.
        result = cash_runway_months(snapshot_esef)
        assert result.ok
        assert result.value == pytest.approx(0.0645, abs=0.005)
        assert is_cash_critical(result.value) is True
