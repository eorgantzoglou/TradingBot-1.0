"""Tests for scout.metrics.valuation.

Two layers, same discipline as test_normalize.py:

  * UNIT tests on small hand-built `FundamentalsSnapshot`s with round numbers,
    verified by hand, pinning each formula's happy path, its missing-input
    path, safe_div's zero-denominator behaviour, the negative-book-equity
    warn-but-compute rule, the net-net boundary, and the
    SHORT_TERM_INVESTMENTS-absent-counts-as-zero rule.

  * A GOLDEN test against the real 3M interim filing on disk, built exactly
    the way test_normalize.py does. Guarded by skipif so a checkout without
    the archive still runs, but it MUST pass here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scout.fundamentals.concepts import Concept
from scout.fundamentals.models import CanonicalFact, EntityRef, FundamentalsSnapshot
from scout.metrics.base import MarketData
from scout.metrics.valuation import (
    earnings_yield,
    ev_ebit,
    ev_sales,
    fcf_yield,
    is_net_net,
    ncav,
    ncav_to_market_cap,
    net_cash_to_market_cap,
    price_to_book,
)

_ENTITY = EntityRef(source="sec", entity_id="1", identifier_scheme="cik", name="Test Co")
_END = date(2026, 6, 30)


def _snapshot(
    values: dict[Concept, float],
    *,
    fiscal_period: str = "FY",
    period_end: date = _END,
) -> FundamentalsSnapshot:
    """Build a FundamentalsSnapshot directly from {Concept: value}, skipping
    the raw-fact/normalization pipeline entirely -- metrics only ever read
    `snapshot.get`/`snapshot.has`, so this is a faithful stand-in.
    """
    facts = {
        concept: CanonicalFact(
            entity_id=_ENTITY.entity_id,
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


# ---------------------------------------------------------------------------
# UNIT: EV/EBIT, EV/Sales, earnings yield
# ---------------------------------------------------------------------------


class TestEvEbit:
    def test_happy_path_annual(self) -> None:
        # mcap = 10 * 100 = 1000; debt = 100 + 200 = 300; EV = 1000+300-50-0 = 1250
        snap = _snapshot(
            {
                Concept.OPERATING_INCOME: 250.0,
                Concept.SHORT_TERM_DEBT: 100.0,
                Concept.LONG_TERM_DEBT: 200.0,
                Concept.CASH_AND_EQUIVALENTS: 50.0,
            },
            fiscal_period="FY",
        )
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = ev_ebit(snap, market)
        assert result.ok
        assert result.value == pytest.approx(5.0)
        assert result.basis == "annual"
        assert result.warnings == []
        assert result.inputs == {"ev": 1250.0, "operating_income": 250.0}

    def test_interim_basis_warns_not_annualized(self) -> None:
        snap = _snapshot(
            {
                Concept.OPERATING_INCOME: 250.0,
                Concept.SHORT_TERM_DEBT: 100.0,
                Concept.LONG_TERM_DEBT: 200.0,
                Concept.CASH_AND_EQUIVALENTS: 50.0,
            },
            fiscal_period="YTD6",
        )
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = ev_ebit(snap, market)
        assert result.ok
        assert result.basis == "interim"
        assert any("not annualized" in w for w in result.warnings)

    def test_missing_operating_income(self) -> None:
        snap = _snapshot(
            {Concept.SHORT_TERM_DEBT: 100.0, Concept.CASH_AND_EQUIVALENTS: 50.0},
            fiscal_period="FY",
        )
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = ev_ebit(snap, market)
        assert not result.ok
        assert result.value is None
        assert result.reason == "no operating income reported"

    def test_missing_market_cap_propagates_ev_reason(self) -> None:
        snap = _snapshot({Concept.OPERATING_INCOME: 250.0}, fiscal_period="FY")
        market = MarketData(price=None, shares_outstanding=None)
        result = ev_ebit(snap, market)
        assert not result.ok
        assert "market capitalisation" in result.reason

    def test_zero_operating_income_is_missing_not_infinite(self) -> None:
        # safe_div zero-denominator: EBIT == 0 must yield ok=False, never inf.
        snap = _snapshot(
            {
                Concept.OPERATING_INCOME: 0.0,
                Concept.SHORT_TERM_DEBT: 100.0,
                Concept.CASH_AND_EQUIVALENTS: 50.0,
            },
            fiscal_period="FY",
        )
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = ev_ebit(snap, market)
        assert not result.ok
        assert result.reason == "operating income is zero"


class TestEvSales:
    def test_happy_path(self) -> None:
        snap = _snapshot(
            {
                Concept.REVENUE: 500.0,
                Concept.SHORT_TERM_DEBT: 100.0,
                Concept.LONG_TERM_DEBT: 200.0,
                Concept.CASH_AND_EQUIVALENTS: 50.0,
            },
            fiscal_period="FY",
        )
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = ev_sales(snap, market)
        assert result.ok
        assert result.value == pytest.approx(2.5)  # EV 1250 / revenue 500

    def test_missing_revenue(self) -> None:
        snap = _snapshot({Concept.SHORT_TERM_DEBT: 100.0, Concept.CASH_AND_EQUIVALENTS: 50.0})
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = ev_sales(snap, market)
        assert not result.ok
        assert result.reason == "no revenue reported"


class TestEarningsYield:
    def test_is_inverse_of_ev_ebit(self) -> None:
        snap = _snapshot(
            {
                Concept.OPERATING_INCOME: 250.0,
                Concept.SHORT_TERM_DEBT: 100.0,
                Concept.LONG_TERM_DEBT: 200.0,
                Concept.CASH_AND_EQUIVALENTS: 50.0,
            },
            fiscal_period="FY",
        )
        market = MarketData(price=10.0, shares_outstanding=100.0)
        yield_result = earnings_yield(snap, market)
        ebit_multiple = ev_ebit(snap, market)
        assert yield_result.ok and ebit_multiple.ok
        assert yield_result.value == pytest.approx(1.0 / ebit_multiple.value)
        assert yield_result.value == pytest.approx(250.0 / 1250.0)

    def test_missing_ev_is_missing(self) -> None:
        snap = _snapshot({Concept.OPERATING_INCOME: 250.0})
        market = MarketData(price=None)
        result = earnings_yield(snap, market)
        assert not result.ok


# ---------------------------------------------------------------------------
# UNIT: P/B, net cash / market cap
# ---------------------------------------------------------------------------


class TestPriceToBook:
    def test_happy_path(self) -> None:
        snap = _snapshot({Concept.TOTAL_EQUITY: 400.0})
        market = MarketData(price=10.0, shares_outstanding=100.0)  # mcap 1000
        result = price_to_book(snap, market)
        assert result.ok
        assert result.value == pytest.approx(2.5)
        assert result.basis == "point-in-time"
        assert result.warnings == []

    def test_negative_equity_still_computes_and_warns(self) -> None:
        snap = _snapshot({Concept.TOTAL_EQUITY: -400.0})
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = price_to_book(snap, market)
        assert result.ok
        assert result.value == pytest.approx(-2.5)
        assert any("not meaningful" in w for w in result.warnings)

    def test_zero_equity_is_missing_safe_div(self) -> None:
        snap = _snapshot({Concept.TOTAL_EQUITY: 0.0})
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = price_to_book(snap, market)
        assert not result.ok
        assert result.reason == "total equity is zero"

    def test_missing_equity(self) -> None:
        snap = _snapshot({})
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = price_to_book(snap, market)
        assert not result.ok
        assert result.reason == "no total equity reported"

    def test_missing_market_cap(self) -> None:
        snap = _snapshot({Concept.TOTAL_EQUITY: 400.0})
        market = MarketData(price=None)
        result = price_to_book(snap, market)
        assert not result.ok


class TestNetCashToMarketCap:
    def test_happy_path_with_short_term_investments(self) -> None:
        snap = _snapshot(
            {
                Concept.CASH_AND_EQUIVALENTS: 300.0,
                Concept.SHORT_TERM_INVESTMENTS: 200.0,
                Concept.SHORT_TERM_DEBT: 100.0,
            },
        )
        market = MarketData(price=10.0, shares_outstanding=100.0)  # mcap 1000
        result = net_cash_to_market_cap(snap, market)
        assert result.ok
        # net cash = 300 + 200 - 100 = 400; / 1000 = 0.4
        assert result.value == pytest.approx(0.4)
        assert result.warnings == []

    def test_short_term_investments_absent_counts_as_zero(self) -> None:
        snap = _snapshot({Concept.CASH_AND_EQUIVALENTS: 300.0, Concept.SHORT_TERM_DEBT: 100.0})
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = net_cash_to_market_cap(snap, market)
        assert result.ok
        # net cash = 300 + 0 - 100 = 200; / 1000 = 0.2
        assert result.value == pytest.approx(0.2)
        assert result.inputs["short_term_investments"] == 0.0

    def test_no_debt_reported_assumes_zero_and_warns(self) -> None:
        snap = _snapshot({Concept.CASH_AND_EQUIVALENTS: 300.0})
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = net_cash_to_market_cap(snap, market)
        assert result.ok
        assert result.value == pytest.approx(0.3)  # 300 / 1000
        assert any("assumed zero" in w for w in result.warnings)

    def test_above_one_signals_trading_below_net_cash(self) -> None:
        snap = _snapshot({Concept.CASH_AND_EQUIVALENTS: 1500.0})
        market = MarketData(price=10.0, shares_outstanding=100.0)  # mcap 1000
        result = net_cash_to_market_cap(snap, market)
        assert result.ok
        assert result.value > 1.0

    def test_missing_cash_is_missing(self) -> None:
        snap = _snapshot({})
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = net_cash_to_market_cap(snap, market)
        assert not result.ok
        assert result.reason == "no cash figure reported"


# ---------------------------------------------------------------------------
# UNIT: FCF yield
# ---------------------------------------------------------------------------


class TestFcfYield:
    def test_happy_path(self) -> None:
        snap = _snapshot(
            {Concept.CASH_FROM_OPERATIONS: 300.0, Concept.CAPEX: 100.0}, fiscal_period="FY"
        )
        market = MarketData(price=10.0, shares_outstanding=100.0)  # mcap 1000
        result = fcf_yield(snap, market)
        assert result.ok
        assert result.value == pytest.approx(0.2)  # (300-100)/1000
        assert result.basis == "annual"

    def test_missing_capex_assumed_zero_with_warning(self) -> None:
        snap = _snapshot({Concept.CASH_FROM_OPERATIONS: 300.0}, fiscal_period="FY")
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = fcf_yield(snap, market)
        assert result.ok
        assert result.value == pytest.approx(0.3)  # 300/1000
        assert any("capex" in w for w in result.warnings)

    def test_missing_cfo_is_missing(self) -> None:
        snap = _snapshot({Concept.CAPEX: 100.0}, fiscal_period="FY")
        market = MarketData(price=10.0, shares_outstanding=100.0)
        result = fcf_yield(snap, market)
        assert not result.ok
        assert result.reason == "no cash from operations reported"


# ---------------------------------------------------------------------------
# UNIT: NCAV and the net-net flag
# ---------------------------------------------------------------------------


class TestNcav:
    def test_happy_path(self) -> None:
        snap = _snapshot({Concept.CURRENT_ASSETS: 1000.0, Concept.TOTAL_LIABILITIES: 400.0})
        result = ncav(snap)
        assert result.ok
        assert result.value == pytest.approx(600.0)
        assert result.kind == "currency"
        assert result.basis == "point-in-time"

    def test_missing_current_assets(self) -> None:
        snap = _snapshot({Concept.TOTAL_LIABILITIES: 400.0})
        result = ncav(snap)
        assert not result.ok
        assert result.reason == "no current assets reported"

    def test_missing_total_liabilities(self) -> None:
        snap = _snapshot({Concept.CURRENT_ASSETS: 1000.0})
        result = ncav(snap)
        assert not result.ok
        assert result.reason == "no total liabilities reported"


class TestNcavToMarketCap:
    def test_happy_path(self) -> None:
        snap = _snapshot({Concept.CURRENT_ASSETS: 1000.0, Concept.TOTAL_LIABILITIES: 400.0})
        market = MarketData(price=6.0, shares_outstanding=100.0)  # mcap 600
        result = ncav_to_market_cap(snap, market)
        assert result.ok
        assert result.value == pytest.approx(1.0)  # 600 / 600

    def test_missing_ncav_propagates(self) -> None:
        snap = _snapshot({})
        market = MarketData(price=6.0, shares_outstanding=100.0)
        result = ncav_to_market_cap(snap, market)
        assert not result.ok


class TestIsNetNet:
    def test_below_two_thirds_ncav_is_net_net(self) -> None:
        # ncav = 600; two-thirds = 400; mcap 399 < 400 -> net-net
        snap = _snapshot({Concept.CURRENT_ASSETS: 1000.0, Concept.TOTAL_LIABILITIES: 400.0})
        market = MarketData(price=3.99, shares_outstanding=100.0)
        result = is_net_net(snap, market)
        assert result.ok
        assert result.value == 1.0

    def test_at_two_thirds_boundary_is_not_net_net(self) -> None:
        # mcap exactly equals two-thirds of ncav -> strict inequality fails
        snap = _snapshot({Concept.CURRENT_ASSETS: 1000.0, Concept.TOTAL_LIABILITIES: 400.0})
        market = MarketData(price=4.00, shares_outstanding=100.0)  # mcap 400 == (2/3)*600
        result = is_net_net(snap, market)
        assert result.ok
        assert result.value == 0.0

    def test_above_two_thirds_ncav_is_not_net_net(self) -> None:
        snap = _snapshot({Concept.CURRENT_ASSETS: 1000.0, Concept.TOTAL_LIABILITIES: 400.0})
        market = MarketData(price=5.0, shares_outstanding=100.0)  # mcap 500 > 400
        result = is_net_net(snap, market)
        assert result.ok
        assert result.value == 0.0

    def test_negative_ncav_is_not_net_net(self) -> None:
        snap = _snapshot({Concept.CURRENT_ASSETS: 100.0, Concept.TOTAL_LIABILITIES: 400.0})
        market = MarketData(price=0.1, shares_outstanding=100.0)  # mcap 10, tiny
        result = is_net_net(snap, market)
        assert result.ok
        assert result.value == 0.0  # ncav <= 0, criterion never met regardless of price

    def test_missing_ncav_is_missing(self) -> None:
        snap = _snapshot({})
        market = MarketData(price=5.0, shares_outstanding=100.0)
        result = is_net_net(snap, market)
        assert not result.ok


# ---------------------------------------------------------------------------
# GOLDEN: real 3M interim filing
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEC_PATH = (
    _REPO_ROOT
    / "data" / "archive" / "sec" / "2026" / "07" / "21"
    / "0000066740-26-000246" / "v1" / "0000066740-26-000246.txt"
)
_needs_sec = pytest.mark.skipif(
    not _SEC_PATH.exists(), reason=f"real SEC archive not present at {_SEC_PATH}"
)


@pytest.fixture(scope="session")
def snapshot_3m() -> FundamentalsSnapshot:
    from scout.fundamentals.normalize import normalize_filing
    from scout.fundamentals.parse.sec import SecXbrlParser

    raw = _SEC_PATH.read_bytes()
    parsed = SecXbrlParser().parse(raw, accession="0000066740-26-000246", entity_hint={"cik": "66740"})
    snap = normalize_filing(parsed, EntityRef("sec", "66740", "cik", name="3M"))
    assert snap is not None
    return snap


@_needs_sec
class TestGolden3M:
    def test_snapshot_is_interim(self, snapshot_3m: FundamentalsSnapshot) -> None:
        assert snapshot_3m.fiscal_period != "FY"

    def test_ev_ebit(self, snapshot_3m: FundamentalsSnapshot) -> None:
        market = MarketData(price=150.0, shares_outstanding=515_722_417)
        result = ev_ebit(snapshot_3m, market)
        assert result.ok, result.reason
        assert result.basis == "interim"
        assert any("not annualized" in w for w in result.warnings)
        assert result.value > 0
        print(f"\n3M ev_ebit = {result.value!r} (inputs={result.inputs})")

    def test_price_to_book(self, snapshot_3m: FundamentalsSnapshot) -> None:
        market = MarketData(price=150.0, shares_outstanding=515_722_417)
        result = price_to_book(snapshot_3m, market)
        assert result.ok, result.reason
        mcap = market.market_cap(snapshot_3m)
        assert result.value == pytest.approx(mcap / 2_952_000_000, rel=1e-6)
        print(f"3M price_to_book = {result.value!r} (market_cap={mcap!r})")

    def test_fcf_yield(self, snapshot_3m: FundamentalsSnapshot) -> None:
        market = MarketData(price=150.0, shares_outstanding=515_722_417)
        result = fcf_yield(snapshot_3m, market)
        assert result.ok, result.reason
        assert result.basis == "interim"
        print(f"3M fcf_yield = {result.value!r}")

    def test_ncav_and_ncav_to_market_cap(self, snapshot_3m: FundamentalsSnapshot) -> None:
        ncav_result = ncav(snapshot_3m)
        assert ncav_result.ok, ncav_result.reason
        # 14,112,000,000 - 31,919,000,000 = -17,807,000,000 -- 3M is not remotely a net-net.
        assert ncav_result.value == pytest.approx(14_112_000_000 - 31_919_000_000)
        market = MarketData(price=150.0, shares_outstanding=515_722_417)
        ratio = ncav_to_market_cap(snapshot_3m, market)
        assert ratio.ok, ratio.reason
        assert ratio.value < 0
        net_net = is_net_net(snapshot_3m, market)
        assert net_net.ok
        assert net_net.value == 0.0  # negative NCAV can never satisfy the criterion
        print(f"3M ncav = {ncav_result.value!r}, ncav_to_market_cap = {ratio.value!r}")

    def test_net_cash_to_market_cap(self, snapshot_3m: FundamentalsSnapshot) -> None:
        market = MarketData(price=150.0, shares_outstanding=515_722_417)
        result = net_cash_to_market_cap(snapshot_3m, market)
        assert result.ok, result.reason
        print(f"3M net_cash_to_market_cap = {result.value!r}")

    def test_ev_sales_and_earnings_yield(self, snapshot_3m: FundamentalsSnapshot) -> None:
        market = MarketData(price=150.0, shares_outstanding=515_722_417)
        ev_sales_result = ev_sales(snapshot_3m, market)
        earnings_yield_result = earnings_yield(snapshot_3m, market)
        assert ev_sales_result.ok, ev_sales_result.reason
        assert earnings_yield_result.ok, earnings_yield_result.reason
        print(f"3M ev_sales = {ev_sales_result.value!r}, earnings_yield = {earnings_yield_result.value!r}")
