"""Compute the full metric set for one entity.

A thin aggregator: single-period metrics run on the latest snapshot,
cross-period metrics (Piotroski, Beneish, dilution) on the latest annual pair.
It exists so the screen and the CLI have one call that returns every metric with
its provenance, rather than each caller re-deciding which snapshot feeds which
formula.

No ranking or judgement happens here -- that is the screen's job (phase 4). This
only computes and labels, keeping the "numbers are code, decisions are separate"
line clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scout.fundamentals.models import FundamentalsSnapshot
from scout.metrics import quality, risk, valuation
from scout.metrics.base import MarketData, MetricValue, select_annual_pair


@dataclass(slots=True)
class MetricReport:
    entity_id: str
    period_end: str
    fiscal_period: str | None
    currency: str | None
    has_market_data: bool
    has_annual_pair: bool
    metrics: dict[str, MetricValue] = field(default_factory=dict)

    def available(self) -> dict[str, MetricValue]:
        """Only the metrics that actually computed -- what a screen would rank on."""
        return {name: m for name, m in self.metrics.items() if m.ok}


def compute_metrics(
    snapshots: list[FundamentalsSnapshot],
    *,
    market: MarketData | None = None,
) -> MetricReport | None:
    """Every metric for the entity these snapshots belong to.

    `snapshots` should be one entity's history (newest first is not required --
    the annual-pair selector sorts). Market-dependent metrics are simply absent
    when `market` is None, so the whole fundamentals-only set still computes for
    a name we have no price for yet.
    """
    if not snapshots:
        return None

    latest = max(snapshots, key=lambda s: s.period_end)
    pair = select_annual_pair(snapshots)
    mkt = market or MarketData()

    report = MetricReport(
        entity_id=latest.entity.entity_id,
        period_end=latest.period_end.isoformat(),
        fiscal_period=latest.fiscal_period,
        currency=latest.currency,
        has_market_data=market is not None and market.price is not None,
        has_annual_pair=pair is not None,
    )
    m = report.metrics

    # Single-period, fundamentals only.
    m["gp_to_assets"] = quality.gross_profit_to_assets(latest)
    m["accruals"] = quality.accruals(latest)
    m["roic"] = quality.roic(latest)
    m["altman_z"] = risk.altman_z_double_prime(latest)
    m["cash_runway_months"] = risk.cash_runway_months(latest)
    m["ncav"] = valuation.ncav(latest)

    # Single-period, needs a price.
    m["ev_ebit"] = valuation.ev_ebit(latest, mkt)
    m["ev_sales"] = valuation.ev_sales(latest, mkt)
    m["earnings_yield"] = valuation.earnings_yield(latest, mkt)
    m["price_to_book"] = valuation.price_to_book(latest, mkt)
    m["net_cash_to_mcap"] = valuation.net_cash_to_market_cap(latest, mkt)
    m["fcf_yield"] = valuation.fcf_yield(latest, mkt)
    m["ncav_to_mcap"] = valuation.ncav_to_market_cap(latest, mkt)
    m["is_net_net"] = valuation.is_net_net(latest, mkt)

    # Cross-period: only meaningful on an annual pair.
    m["piotroski_f"] = quality.piotroski_from_history(snapshots)
    m["beneish_m"] = risk.beneish_from_history(snapshots)
    m["share_issuance"] = risk.dilution_from_history(snapshots)

    return report
