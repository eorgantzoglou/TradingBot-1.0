"""Shared types and derivations for the metrics layer.

Design rule 1 from PLAN.md lives here: every number is computed in code and
unit-tested, and the LLM never touches this path. A metric that cannot be
computed returns a `MetricValue` with `ok=False` and a plain-language `reason`,
never a guess and never a silent zero -- a missing denominator and a real zero
are different facts, and microcap filings miss fields constantly.

Derivations that more than one metric needs (gross profit, total debt,
enterprise value, the two-period annual pairing) live here so there is exactly
one definition of each. A second, subtly different definition of "total debt" in
another module is the kind of error this layer exists to prevent.

Sign convention follows the stored facts: expenses and outflows are positive as
filed. So FCF = CFO - capex (capex is a positive payment), and operating income
positive means profit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from scout.fundamentals.concepts import Concept
from scout.fundamentals.models import FundamentalsSnapshot


@dataclass(frozen=True, slots=True)
class MarketData:
    """Price-side inputs, supplied by the caller.

    Fundamentals come from filings; price does not. Metrics that need a market
    capitalisation (EV/EBIT, P/B, Altman's market-value term) take this
    explicitly rather than reaching for a price feed, so the metrics layer stays
    pure and testable and the (paid, deferred) price source is someone else's
    concern. `shares_outstanding` here overrides the filing's cover-date count
    when a more current figure is available; it falls back to the snapshot's.
    """

    price: float | None = None
    shares_outstanding: float | None = None
    currency: str | None = None
    as_of: date | None = None

    def market_cap(self, snapshot: FundamentalsSnapshot | None = None) -> float | None:
        if self.price is None:
            return None
        shares = self.shares_outstanding
        if shares is None and snapshot is not None:
            shares = snapshot.get(Concept.SHARES_OUTSTANDING)
        if shares is None:
            return None
        return self.price * shares


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One metric's result, with enough provenance to audit it.

    `inputs` records the concept values the number was built from, so a
    surprising ratio can be traced to the figures behind it without re-deriving
    anything -- the same provenance discipline the fundamentals layer keeps.
    """

    name: str
    value: float | None
    kind: str
    """"ratio" | "score" | "currency" | "pct" | "flag" | "count"."""

    basis: str
    """"annual" | "interim" | "cross-period" | "point-in-time" -- the period
    shape the number describes, so a caller never compares an annual metric to an
    interim one by accident."""

    ok: bool
    reason: str | None = None
    """Why `value` is None. Set iff not ok."""

    inputs: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def missing(cls, name: str, kind: str, basis: str, reason: str) -> MetricValue:
        return cls(name=name, value=None, kind=kind, basis=basis, ok=False, reason=reason)

    @classmethod
    def of(
        cls,
        name: str,
        value: float,
        kind: str,
        basis: str,
        *,
        inputs: dict[str, float] | None = None,
        warnings: list[str] | None = None,
    ) -> MetricValue:
        return cls(
            name=name,
            value=value,
            kind=kind,
            basis=basis,
            ok=True,
            inputs=inputs or {},
            warnings=warnings or [],
        )


# --------------------------------------------------------------------------- #
# Safe arithmetic
# --------------------------------------------------------------------------- #


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Divide, returning None when the result is undefined.

    None when either operand is missing or the denominator is zero. This is the
    single most common reason a metric is unavailable for a microcap, so it is
    one function used everywhere rather than an `if` scattered across formulas.
    """
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def require(snapshot: FundamentalsSnapshot, *concepts: Concept) -> dict[Concept, float] | None:
    """Return the given concepts' values, or None if ANY is absent.

    For a metric where every input is mandatory, this collapses the presence
    check to one call and makes the missing-input path uniform.
    """
    out: dict[Concept, float] = {}
    for concept in concepts:
        value = snapshot.get(concept)
        if value is None:
            return None
        out[concept] = value
    return out


# --------------------------------------------------------------------------- #
# Shared derivations -- one definition each, used by every module
# --------------------------------------------------------------------------- #


def gross_profit(snapshot: FundamentalsSnapshot) -> tuple[float | None, bool]:
    """Gross profit, and whether it was derived.

    Prefer the reported figure; fall back to revenue - cost_of_revenue. Many
    filers (3M among them) never report gross profit as a line item, so the
    derivation is the common path, not the exception.
    """
    reported = snapshot.get(Concept.GROSS_PROFIT)
    if reported is not None:
        return reported, False
    revenue = snapshot.get(Concept.REVENUE)
    cogs = snapshot.get(Concept.COST_OF_REVENUE)
    if revenue is None or cogs is None:
        return None, False
    return revenue - cogs, True


def total_debt(snapshot: FundamentalsSnapshot) -> float | None:
    """Short-term + long-term interest-bearing debt.

    Returns None only when NEITHER is present -- a firm reporting long-term debt
    but no short-term line legitimately has zero short-term debt, so a missing
    component counts as zero as long as at least one is reported.
    """
    short = snapshot.get(Concept.SHORT_TERM_DEBT)
    long = snapshot.get(Concept.LONG_TERM_DEBT)
    if short is None and long is None:
        return None
    return (short or 0.0) + (long or 0.0)


def working_capital(snapshot: FundamentalsSnapshot) -> float | None:
    values = require(snapshot, Concept.CURRENT_ASSETS, Concept.CURRENT_LIABILITIES)
    if values is None:
        return None
    return values[Concept.CURRENT_ASSETS] - values[Concept.CURRENT_LIABILITIES]


def enterprise_value(
    snapshot: FundamentalsSnapshot, market: MarketData
) -> tuple[float | None, str | None]:
    """EV = market cap + total debt - cash - short-term investments.

    Returns (value, reason-if-None). Short-term investments are netted with cash
    because for a cash-box microcap they are effectively cash; a filer that omits
    the line simply contributes zero there.
    """
    mcap = market.market_cap(snapshot)
    if mcap is None:
        return None, "no market capitalisation (price or shares missing)"
    debt = total_debt(snapshot)
    if debt is None:
        return None, "no debt figures reported"
    cash = snapshot.get(Concept.CASH_AND_EQUIVALENTS)
    if cash is None:
        return None, "no cash figure reported"
    sti = snapshot.get(Concept.SHORT_TERM_INVESTMENTS) or 0.0
    return mcap + debt - cash - sti, None


# --------------------------------------------------------------------------- #
# Two-period alignment
# --------------------------------------------------------------------------- #


def select_annual_pair(
    snapshots: list[FundamentalsSnapshot],
) -> tuple[FundamentalsSnapshot, FundamentalsSnapshot] | None:
    """The two most recent consecutive annual snapshots (current, prior).

    Piotroski, Beneish and the dilution rate are year-over-year measures; they
    are only meaningful on annual data one fiscal year apart. Interim snapshots
    are ignored, and two annuals more than ~15 months apart are rejected rather
    than silently compared across a gap -- a two-year jump would make growth
    ratios lie.
    """
    annual = sorted(
        (s for s in snapshots if s.fiscal_period == "FY"),
        key=lambda s: s.period_end,
        reverse=True,
    )
    if len(annual) < 2:
        return None
    current, prior = annual[0], annual[1]
    gap_days = (current.period_end - prior.period_end).days
    if not (250 <= gap_days <= 460):
        return None
    return current, prior
