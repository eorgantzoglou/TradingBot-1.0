"""Cheapness metrics: what does the market charge for this business.

EV/EBIT is the anchor. Operating income (EBIT) is the earnings line the
concept vocabulary was deliberately built around because it is the most
comparable figure across US-GAAP, IFRS and JGAAP (concepts.py) -- unlike net
income, it sits above the financing-structure and tax-jurisdiction noise that
makes cross-standard P/E comparisons unreliable. Enterprise value is used
instead of market cap for the same reason Greenblatt's "Magic Formula" uses
it: it prices the whole capital structure, so a company that looks cheap on
P/E only because it is loaded with debt does not sneak through.

NCAV and the net-net flag are the other pole -- Graham's original deep-value
test, not an earnings multiple at all. A stock trading below two-thirds of
current assets minus ALL liabilities is priced as if the business itself is
worth less than its liquidation floor. It is rare (most net-nets today are
distressed or fraudulent), which is exactly why it is worth flagging
mechanically rather than trusting a screen built on P/E or P/B to surface it.

Every formula here is a straight application of `base.py`'s shared
derivations (`enterprise_value`, `total_debt`, `gross_profit`) plus
`safe_div`/`require` for the missing-input discipline: a metric this module
cannot compute returns `MetricValue.missing(...)` with a plain-language
reason, never a guessed or zero value (base.py's design rule 1).
"""

from __future__ import annotations

from scout.fundamentals.concepts import Concept
from scout.fundamentals.models import FundamentalsSnapshot
from scout.metrics.base import (
    MarketData,
    MetricValue,
    enterprise_value,
    gross_profit,  # noqa: F401  (re-exported for callers that expect it here)
    safe_div,
    total_debt,
)


def _flow_basis(snapshot: FundamentalsSnapshot) -> tuple[str, list[str]]:
    """Basis + warnings for a metric mixing a point-in-time market cap with a
    flow (revenue, EBIT, FCF).

    "annual" only for a full fiscal year; anything else (a quarter, a YTD
    interim) is "interim" and gets a warning that the flow was not
    annualized -- comparing an un-annualized 6-month EBIT to a full-year
    EV/EBIT screen would silently understate the multiple by roughly 2x.
    """
    if snapshot.fiscal_period == "FY":
        return "annual", []
    return "interim", ["flow figure is not annualized (interim period)"]


def ev_ebit(snapshot: FundamentalsSnapshot, market: MarketData) -> MetricValue:
    """EV / operating income (EBIT) -- Greenblatt's "Magic Formula" multiple
    and the most comparable earnings multiple across accounting standards."""
    basis, warnings = _flow_basis(snapshot)
    ev, ev_reason = enterprise_value(snapshot, market)
    if ev is None:
        return MetricValue.missing("ev_ebit", "ratio", basis, ev_reason)
    ebit = snapshot.get(Concept.OPERATING_INCOME)
    if ebit is None:
        return MetricValue.missing("ev_ebit", "ratio", basis, "no operating income reported")
    value = safe_div(ev, ebit)
    if value is None:
        return MetricValue.missing("ev_ebit", "ratio", basis, "operating income is zero")
    return MetricValue.of(
        "ev_ebit", value, "ratio", basis,
        inputs={"ev": ev, "operating_income": ebit},
        warnings=warnings,
    )


def ev_sales(snapshot: FundamentalsSnapshot, market: MarketData) -> MetricValue:
    """EV / revenue. Coarser than EV/EBIT but stays defined for loss-makers,
    where EV/EBIT is either negative or meaningless."""
    basis, warnings = _flow_basis(snapshot)
    ev, ev_reason = enterprise_value(snapshot, market)
    if ev is None:
        return MetricValue.missing("ev_sales", "ratio", basis, ev_reason)
    revenue = snapshot.get(Concept.REVENUE)
    if revenue is None:
        return MetricValue.missing("ev_sales", "ratio", basis, "no revenue reported")
    value = safe_div(ev, revenue)
    if value is None:
        return MetricValue.missing("ev_sales", "ratio", basis, "revenue is zero")
    return MetricValue.of(
        "ev_sales", value, "ratio", basis,
        inputs={"ev": ev, "revenue": revenue},
        warnings=warnings,
    )


def earnings_yield(snapshot: FundamentalsSnapshot, market: MarketData) -> MetricValue:
    """Operating income / EV -- the inverse of EV/EBIT, reported separately.

    Greenblatt ranks on this directly (higher is cheaper) rather than on
    EV/EBIT, and the two are not interchangeable for ranking purposes: when
    EBIT crosses zero, EV/EBIT flips sign and blows up toward +/-infinity
    while earnings yield stays finite and orders sensibly around zero.
    """
    basis, warnings = _flow_basis(snapshot)
    ev, ev_reason = enterprise_value(snapshot, market)
    if ev is None:
        return MetricValue.missing("earnings_yield", "ratio", basis, ev_reason)
    ebit = snapshot.get(Concept.OPERATING_INCOME)
    if ebit is None:
        return MetricValue.missing("earnings_yield", "ratio", basis, "no operating income reported")
    value = safe_div(ebit, ev)
    if value is None:
        return MetricValue.missing("earnings_yield", "ratio", basis, "enterprise value is zero")
    return MetricValue.of(
        "earnings_yield", value, "ratio", basis,
        inputs={"ev": ev, "operating_income": ebit},
        warnings=warnings,
    )


def price_to_book(snapshot: FundamentalsSnapshot, market: MarketData) -> MetricValue:
    """Market cap / total equity.

    Computed even when book equity is negative or zero -- a negative-equity
    microcap is exactly the case worth flagging, not hiding by suppressing
    the ratio. safe_div still guards the true zero-denominator case.
    """
    mcap = market.market_cap(snapshot)
    if mcap is None:
        return MetricValue.missing(
            "price_to_book", "ratio", "point-in-time", "no market capitalisation (price or shares missing)"
        )
    equity = snapshot.get(Concept.TOTAL_EQUITY)
    if equity is None:
        return MetricValue.missing("price_to_book", "ratio", "point-in-time", "no total equity reported")
    value = safe_div(mcap, equity)
    if value is None:
        return MetricValue.missing("price_to_book", "ratio", "point-in-time", "total equity is zero")
    warnings = []
    if equity <= 0:
        warnings.append("negative or zero book equity -- P/B not meaningful")
    return MetricValue.of(
        "price_to_book", value, "ratio", "point-in-time",
        inputs={"market_cap": mcap, "total_equity": equity},
        warnings=warnings,
    )


def net_cash_to_market_cap(snapshot: FundamentalsSnapshot, market: MarketData) -> MetricValue:
    """(cash + short-term investments - total debt) / market cap.

    A value above 1.0 means the company trades for less than the cash on its
    balance sheet net of debt -- a classic deep-value signal that must stay
    computable for tiny, debt-free firms, so a missing debt figure is
    treated as zero debt (with a warning) rather than making the metric
    unavailable.
    """
    mcap = market.market_cap(snapshot)
    if mcap is None:
        return MetricValue.missing(
            "net_cash_to_market_cap", "ratio", "point-in-time",
            "no market capitalisation (price or shares missing)",
        )
    cash = snapshot.get(Concept.CASH_AND_EQUIVALENTS)
    if cash is None:
        return MetricValue.missing(
            "net_cash_to_market_cap", "ratio", "point-in-time", "no cash figure reported"
        )
    sti = snapshot.get(Concept.SHORT_TERM_INVESTMENTS) or 0.0
    debt = total_debt(snapshot)
    warnings = []
    if debt is None:
        debt = 0.0
        warnings.append("no debt reported, assumed zero")
    net_cash = cash + sti - debt
    value = safe_div(net_cash, mcap)
    if value is None:
        return MetricValue.missing(
            "net_cash_to_market_cap", "ratio", "point-in-time", "market cap is zero"
        )
    return MetricValue.of(
        "net_cash_to_market_cap", value, "ratio", "point-in-time",
        inputs={"cash": cash, "short_term_investments": sti, "total_debt": debt, "market_cap": mcap},
        warnings=warnings,
    )


def fcf_yield(snapshot: FundamentalsSnapshot, market: MarketData) -> MetricValue:
    """(cash from operations - capex) / market cap.

    CAPEX missing is treated as zero (with a warning) rather than making the
    metric unavailable -- some filers bury capex in a combined investing
    line, and refusing to compute FCF yield for all of them would discard
    more signal than the approximation costs.
    """
    basis, warnings = _flow_basis(snapshot)
    mcap = market.market_cap(snapshot)
    if mcap is None:
        return MetricValue.missing(
            "fcf_yield", "ratio", basis, "no market capitalisation (price or shares missing)"
        )
    cfo = snapshot.get(Concept.CASH_FROM_OPERATIONS)
    if cfo is None:
        return MetricValue.missing("fcf_yield", "ratio", basis, "no cash from operations reported")
    capex = snapshot.get(Concept.CAPEX)
    if capex is None:
        capex = 0.0
        warnings = [*warnings, "no capex reported, assumed zero"]
    fcf = cfo - capex
    value = safe_div(fcf, mcap)
    if value is None:
        return MetricValue.missing("fcf_yield", "ratio", basis, "market cap is zero")
    return MetricValue.of(
        "fcf_yield", value, "ratio", basis,
        inputs={"cash_from_operations": cfo, "capex": capex, "market_cap": mcap},
        warnings=warnings,
    )


def ncav(snapshot: FundamentalsSnapshot) -> MetricValue:
    """Graham net current asset value = current assets - total liabilities.

    Deliberately total liabilities, not current liabilities: Graham's
    original test subtracts every claim ahead of equity, not just the ones
    due within a year, so long-term debt still counts against the company.
    """
    current_assets = snapshot.get(Concept.CURRENT_ASSETS)
    if current_assets is None:
        return MetricValue.missing("ncav", "currency", "point-in-time", "no current assets reported")
    total_liabilities = snapshot.get(Concept.TOTAL_LIABILITIES)
    if total_liabilities is None:
        return MetricValue.missing("ncav", "currency", "point-in-time", "no total liabilities reported")
    value = current_assets - total_liabilities
    return MetricValue.of(
        "ncav", value, "currency", "point-in-time",
        inputs={"current_assets": current_assets, "total_liabilities": total_liabilities},
    )


def ncav_to_market_cap(snapshot: FundamentalsSnapshot, market: MarketData) -> MetricValue:
    """NCAV / market cap -- how far the market price sits from Graham's
    liquidation-value floor."""
    ncav_result = ncav(snapshot)
    if not ncav_result.ok:
        return MetricValue.missing("ncav_to_market_cap", "ratio", "point-in-time", ncav_result.reason)
    mcap = market.market_cap(snapshot)
    if mcap is None:
        return MetricValue.missing(
            "ncav_to_market_cap", "ratio", "point-in-time",
            "no market capitalisation (price or shares missing)",
        )
    value = safe_div(ncav_result.value, mcap)
    if value is None:
        return MetricValue.missing("ncav_to_market_cap", "ratio", "point-in-time", "market cap is zero")
    return MetricValue.of(
        "ncav_to_market_cap", value, "ratio", "point-in-time",
        inputs={"ncav": ncav_result.value, "market_cap": mcap},
    )


def is_net_net(snapshot: FundamentalsSnapshot, market: MarketData) -> MetricValue:
    """Flag: market cap < (2/3) * NCAV, and NCAV > 0 -- Graham's net-net
    criterion, the classic deep-value screen for trading below liquidation
    value with a margin of safety built into the two-thirds haircut.
    """
    ncav_result = ncav(snapshot)
    if not ncav_result.ok:
        return MetricValue.missing("is_net_net", "flag", "point-in-time", ncav_result.reason)
    mcap = market.market_cap(snapshot)
    if mcap is None:
        return MetricValue.missing(
            "is_net_net", "flag", "point-in-time", "no market capitalisation (price or shares missing)"
        )
    ncav_value = ncav_result.value
    is_net_net_flag = ncav_value > 0 and mcap < (2.0 / 3.0) * ncav_value
    return MetricValue.of(
        "is_net_net", 1.0 if is_net_net_flag else 0.0, "flag", "point-in-time",
        inputs={"ncav": ncav_value, "market_cap": mcap},
    )
