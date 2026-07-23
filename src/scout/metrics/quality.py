"""Quality metrics: is the business good, and are its earnings real?

Cheapness (valuation.py) finds statistically cheap names; quality separates the
genuinely mispriced from the value traps. The three families here are chosen for
evidence, not familiarity:

  * Novy-Marx gross-profitability (GP/A) is the best-evidenced quality metric in
    the cross-section -- gross profit is the cleanest earnings line (above the
    accruals, write-offs and tax games that distort net income), and scaled by
    assets it predicts returns about as well as book-to-market does, which is why
    it leads this module.

  * Piotroski's F-Score was designed for *exactly* the low-coverage, high-book
    value stocks this project targets: it takes a basket of cheap firms and
    separates the improving from the deteriorating using nine plain
    accounting signals a microcap filing actually contains. No analyst estimates,
    no price history -- just the two most recent annual filings.

  * Sloan accruals are here as a quality/fraud FLAG, not an alpha factor. The
    accruals *anomaly* (buying low-accrual firms for excess return) has largely
    decayed since publication as it got arbitraged away; but a large gap between
    accounting earnings and the cash that backs them is still a genuine
    red flag for earnings quality and manipulation, so we compute it to warn,
    not to rank.

All arithmetic goes through base.safe_div and returns MetricValue -- a missing
input degrades to ok=False with a reason, never a guessed number.
"""

from __future__ import annotations

from scout.fundamentals.concepts import Concept
from scout.fundamentals.models import FundamentalsSnapshot
from scout.metrics.base import (
    MetricValue,
    gross_profit,
    require,
    safe_div,
    select_annual_pair,
    total_debt,
)

# When a snapshot is not a full fiscal year its ratios describe a partial period;
# a caller must never compare it to an annual one, so we stamp the basis honestly.
_ASSUMED_TAX_RATE = 0.21


def _basis(snapshot: FundamentalsSnapshot) -> str:
    return "annual" if snapshot.fiscal_period == "FY" else "interim"


# --------------------------------------------------------------------------- #
# Novy-Marx gross-profitability
# --------------------------------------------------------------------------- #


def gross_profit_to_assets(snapshot: FundamentalsSnapshot) -> MetricValue:
    """Gross profit / total assets -- the headline quality metric.

    gross_profit() prefers the reported figure and falls back to
    revenue - cost_of_revenue; when it derives we surface a warning, because a
    derived gross profit inherits any COGS-classification quirk of the filer.
    """
    basis = _basis(snapshot)
    gp, was_derived = gross_profit(snapshot)
    assets = snapshot.get(Concept.TOTAL_ASSETS)

    ratio = safe_div(gp, assets)
    if ratio is None:
        return MetricValue.missing(
            "gross_profit_to_assets", "ratio", basis, "gross profit or total assets unavailable"
        )

    warnings = ["gross profit derived as revenue - COGS"] if was_derived else []
    return MetricValue.of(
        "gross_profit_to_assets", ratio, "ratio", basis,
        inputs={"gross_profit": gp, "total_assets": assets},
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Sloan accruals -- a quality/fraud flag, not a factor (see module docstring)
# --------------------------------------------------------------------------- #


def accruals(snapshot: FundamentalsSnapshot) -> MetricValue:
    """(net income - cash from operations) / total assets.

    High positive = accounting earnings that operating cash did not back, i.e.
    earnings resting on accruals -- a lower-quality, higher-manipulation-risk
    profile. Kept as a flag to warn on, not a ranking factor.
    """
    basis = _basis(snapshot)
    values = require(
        snapshot, Concept.NET_INCOME, Concept.CASH_FROM_OPERATIONS, Concept.TOTAL_ASSETS
    )
    if values is None:
        return MetricValue.missing(
            "accruals", "ratio", basis, "net income, operating cash, or total assets unavailable"
        )

    net_income = values[Concept.NET_INCOME]
    cash_ops = values[Concept.CASH_FROM_OPERATIONS]
    assets = values[Concept.TOTAL_ASSETS]

    ratio = safe_div(net_income - cash_ops, assets)
    if ratio is None:
        return MetricValue.missing("accruals", "ratio", basis, "total assets is zero")

    return MetricValue.of(
        "accruals", ratio, "ratio", basis,
        inputs={"net_income": net_income, "cash_from_operations": cash_ops, "total_assets": assets},
    )


# --------------------------------------------------------------------------- #
# Return on invested capital
# --------------------------------------------------------------------------- #


def roic(snapshot: FundamentalsSnapshot) -> MetricValue:
    """NOPAT / invested capital.

    NOPAT = operating income * (1 - effective tax rate). Invested capital =
    equity + total debt - cash: the capital actually put to work, net of the cash
    pile a value name often sits on. Returns missing (never a divide-by-tiny
    explosion) when invested capital is non-positive.
    """
    basis = _basis(snapshot)
    operating_income = snapshot.get(Concept.OPERATING_INCOME)
    if operating_income is None:
        return MetricValue.missing("roic", "ratio", basis, "operating income unavailable")

    equity = snapshot.get(Concept.TOTAL_EQUITY)
    debt = total_debt(snapshot)
    cash = snapshot.get(Concept.CASH_AND_EQUIVALENTS)
    if equity is None or debt is None or cash is None:
        return MetricValue.missing(
            "roic", "ratio", basis, "equity, debt, or cash unavailable for invested capital"
        )

    invested_capital = equity + debt - cash
    if invested_capital <= 0:
        return MetricValue.missing(
            "roic", "ratio", basis, "invested capital is non-positive (net-cash or negative-equity firm)"
        )

    tax_rate, tax_warnings = _effective_tax_rate(snapshot)
    nopat = operating_income * (1 - tax_rate)
    ratio = safe_div(nopat, invested_capital)
    if ratio is None:  # unreachable given the guard above, but never trust that
        return MetricValue.missing("roic", "ratio", basis, "invested capital is zero")

    return MetricValue.of(
        "roic", ratio, "ratio", basis,
        inputs={
            "operating_income": operating_income,
            "effective_tax_rate": tax_rate,
            "invested_capital": invested_capital,
        },
        warnings=tax_warnings,
    )


def _effective_tax_rate(snapshot: FundamentalsSnapshot) -> tuple[float, list[str]]:
    """Effective tax rate, clamped to [0, 0.5], with any adjustment warned.

    A near-zero or negative pre-tax base produces a nonsensical raw rate (a tiny
    denominator, or a benefit on a loss) that would otherwise inflate or invert
    NOPAT. We clamp such rates into a sane band rather than propagate the
    garbage, and fall back to a statutory-ish 21% when tax or pre-tax is missing.
    """
    tax = snapshot.get(Concept.INCOME_TAX_EXPENSE)
    pretax = snapshot.get(Concept.INCOME_BEFORE_TAX)
    raw_rate = safe_div(tax, pretax)
    if raw_rate is None:
        return _ASSUMED_TAX_RATE, ["effective tax rate unavailable; assumed 21%"]

    if raw_rate < 0.0 or raw_rate > 0.5:
        clamped = min(max(raw_rate, 0.0), 0.5)
        return clamped, [f"effective tax rate {raw_rate:.2f} outside [0, 0.5]; clamped to {clamped:.2f}"]

    return raw_rate, []


# --------------------------------------------------------------------------- #
# Piotroski F-Score
# --------------------------------------------------------------------------- #

# Human-readable label per signal, used both in provenance and in the
# "could-not-evaluate" warnings.
_SIGNAL_LABELS = {
    "S1": "ROA>0",
    "S2": "CFO>0",
    "S3": "dROA>0",
    "S4": "Accrual (CFO>ROA)",
    "S5": "dLeverage down",
    "S6": "dCurrentRatio up",
    "S7": "No dilution",
    "S8": "dGrossMargin up",
    "S9": "dAssetTurnover up",
}


def piotroski_f_score(
    current: FundamentalsSnapshot, prior: FundamentalsSnapshot
) -> MetricValue:
    """The 9-signal Piotroski F-Score (0-9) over two consecutive annual filings.

    Both snapshots must be annual (FY); a year-over-year quality trend on interim
    data is meaningless, so a non-FY input is rejected outright.

    Simplification (documented on purpose): the standard F-Score divides ROA by
    AVERAGE assets across the year. With only the two period-END snapshots the
    screen carries, we use END-of-period total assets for each year's ROA. This
    is the accepted two-snapshot form and keeps every signal computable from the
    filings alone; the two ROAs (current and prior) stay internally consistent
    because both use the same convention.

    Conservatism: if any input a signal needs is absent, that signal scores 0 and
    a warning names it. A firm that omits a line does not get the benefit of the
    doubt -- an unverifiable signal is treated as failed, not skipped.
    """
    if current.fiscal_period != "FY" or prior.fiscal_period != "FY":
        return MetricValue.missing(
            "piotroski_f", "score", "cross-period",
            "Piotroski requires two annual (FY) snapshots; "
            f"got current={current.fiscal_period!r}, prior={prior.fiscal_period!r}",
        )

    conditions = _piotroski_conditions(current, prior)

    inputs: dict[str, float] = {}
    warnings: list[str] = []
    score = 0.0
    for key in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"):
        condition = conditions[key]
        if condition is None:
            # A required input was missing -> unverifiable -> scores 0 and is flagged.
            inputs[key] = 0.0
            warnings.append(f"{key} ({_SIGNAL_LABELS[key]}) not evaluable: a required input is missing")
            continue
        point = 1.0 if condition else 0.0
        inputs[key] = point
        score += point

    inputs["score"] = score
    return MetricValue.of(
        "piotroski_f", score, "score", "cross-period", inputs=inputs, warnings=warnings
    )


def _piotroski_conditions(
    current: FundamentalsSnapshot, prior: FundamentalsSnapshot
) -> dict[str, bool | None]:
    """Evaluate each of the nine signals to True / False / None.

    None means "an input was missing" (via safe_div returning None, or a raw
    get() returning None), which the scorer turns into a failed, flagged signal.
    """
    # Current-year figures.
    net_income_t = current.get(Concept.NET_INCOME)
    assets_t = current.get(Concept.TOTAL_ASSETS)
    cash_ops_t = current.get(Concept.CASH_FROM_OPERATIONS)
    revenue_t = current.get(Concept.REVENUE)
    gross_profit_t, _ = gross_profit(current)

    # Prior-year figures.
    net_income_p = prior.get(Concept.NET_INCOME)
    assets_p = prior.get(Concept.TOTAL_ASSETS)
    revenue_p = prior.get(Concept.REVENUE)
    gross_profit_p, _ = gross_profit(prior)

    # ROA each year on END-of-period assets (the two-snapshot simplification).
    roa_t = safe_div(net_income_t, assets_t)
    roa_p = safe_div(net_income_p, assets_p)

    return {
        # Profitability.
        "S1": _gt(roa_t, 0.0),
        "S2": _gt(cash_ops_t, 0.0),
        "S3": _gt(roa_t, roa_p),
        # Cash-flow return exceeds accounting return -> earnings backed by cash.
        "S4": _gt(safe_div(cash_ops_t, assets_t), roa_t),
        # Leverage / liquidity / funding.
        "S5": _lt(
            safe_div(current.get(Concept.LONG_TERM_DEBT), assets_t),
            safe_div(prior.get(Concept.LONG_TERM_DEBT), assets_p),
        ),
        "S6": _gt(
            safe_div(current.get(Concept.CURRENT_ASSETS), current.get(Concept.CURRENT_LIABILITIES)),
            safe_div(prior.get(Concept.CURRENT_ASSETS), prior.get(Concept.CURRENT_LIABILITIES)),
        ),
        "S7": _le(
            current.get(Concept.SHARES_OUTSTANDING), prior.get(Concept.SHARES_OUTSTANDING)
        ),
        # Efficiency.
        "S8": _gt(safe_div(gross_profit_t, revenue_t), safe_div(gross_profit_p, revenue_p)),
        "S9": _gt(safe_div(revenue_t, assets_t), safe_div(revenue_p, assets_p)),
    }


def piotroski_from_history(snapshots: list[FundamentalsSnapshot]) -> MetricValue:
    """Convenience: pick the two most recent consecutive annuals, then score.

    Returns missing when no valid annual pair exists (fewer than two FY snapshots,
    or the two available ones are not ~one fiscal year apart).
    """
    pair = select_annual_pair(snapshots)
    if pair is None:
        return MetricValue.missing(
            "piotroski_f", "score", "cross-period",
            "no consecutive annual (FY) snapshot pair available",
        )
    current, prior = pair
    return piotroski_f_score(current, prior)


# --------------------------------------------------------------------------- #
# None-safe comparison helpers -- a missing operand makes the signal unverifiable
# --------------------------------------------------------------------------- #


def _gt(a: float | None, b: float | None) -> bool | None:
    if a is None or b is None:
        return None
    return a > b


def _lt(a: float | None, b: float | None) -> bool | None:
    if a is None or b is None:
        return None
    return a < b


def _le(a: float | None, b: float | None) -> bool | None:
    if a is None or b is None:
        return None
    return a <= b
