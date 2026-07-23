"""Integration test for the metric aggregator against a real filing.

The individual formulas are covered exhaustively in test_valuation/quality/risk;
this guards the wiring in report.compute_metrics -- that the right snapshot feeds
each metric, that market-dependent metrics appear only with a price, and that
cross-period metrics stay absent when there is only one filing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scout.fundamentals.models import EntityRef
from scout.fundamentals.normalize import normalize_filing
from scout.fundamentals.parse.sec import SecXbrlParser
from scout.metrics.base import MarketData
from scout.metrics.report import compute_metrics

_3M = Path("data/archive/sec/2026/07/21/0000066740-26-000246/v1/0000066740-26-000246.txt")

MARKET_DEPENDENT = {
    "ev_ebit", "ev_sales", "earnings_yield", "price_to_book",
    "net_cash_to_mcap", "fcf_yield", "ncav_to_mcap", "is_net_net",
}
CROSS_PERIOD = {"piotroski_f", "beneish_m", "share_issuance"}
FUNDAMENTALS_ONLY = {"gp_to_assets", "accruals", "roic", "altman_z", "cash_runway_months", "ncav"}


@pytest.fixture(scope="module")
def snapshot_3m():
    if not _3M.exists():
        pytest.skip("3M filing not in archive")
    parsed = SecXbrlParser().parse(
        _3M.read_bytes(), accession="0000066740-26-000246", entity_hint={"cik": "66740"}
    )
    return normalize_filing(parsed, EntityRef("sec", "66740", "cik", name="3M"))


def test_compute_returns_none_for_no_snapshots():
    assert compute_metrics([]) is None


def test_fundamentals_only_metrics_compute_without_price(snapshot_3m):
    report = compute_metrics([snapshot_3m], market=None)

    assert report is not None
    assert report.entity_id == "66740"
    assert report.has_market_data is False
    assert report.has_annual_pair is False  # only one filing archived

    # The whole no-price set computes anyway -- that's the point of not being
    # blocked on a price feed we don't have yet.
    for name in FUNDAMENTALS_ONLY:
        assert report.metrics[name].ok, f"{name} should compute without a price"

    # Market-dependent metrics are cleanly absent, not crashed.
    for name in MARKET_DEPENDENT:
        assert not report.metrics[name].ok
        assert "market" in report.metrics[name].reason.lower()

    # Cross-period metrics need two annual filings.
    for name in CROSS_PERIOD:
        assert not report.metrics[name].ok


def test_price_unlocks_valuation_metrics(snapshot_3m):
    report = compute_metrics(
        [snapshot_3m], market=MarketData(price=150.0, shares_outstanding=515_722_417)
    )
    assert report is not None
    assert report.has_market_data is True
    for name in MARKET_DEPENDENT:
        assert report.metrics[name].ok, f"{name} should compute with a price"

    # Spot-check a value against the hand-verified figure: P/B = mcap / equity.
    mcap = 150.0 * 515_722_417
    assert report.metrics["price_to_book"].value == pytest.approx(mcap / 2_952_000_000, rel=1e-6)


def test_available_returns_only_computed(snapshot_3m):
    report = compute_metrics([snapshot_3m], market=None)
    available = report.available()
    assert set(available) >= FUNDAMENTALS_ONLY
    assert not (MARKET_DEPENDENT & set(available))
