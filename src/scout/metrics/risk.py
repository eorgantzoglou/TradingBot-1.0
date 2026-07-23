"""Fraud, bankruptcy and dilution risk metrics -- the exclusion gates.

The plan puts most of the expected value here, in *rejection* rather than
selection: a microcap universe is thick with earnings manipulators, dilution
machines and firms months from running out of cash, and a screen that cannot
mechanically flag them is worse than useless because it hands the LLM a
plausible-looking fraud to write up. So these four are computed in code, never
guessed, and unit-tested against worked arithmetic.

Why these four specifically:

  * Beneish M-Score caught 71% of the famous accounting frauds (Enron among
    them) a year or more ahead of public disclosure, from ordinary financial-
    statement data -- no insider access required. Eight ratios comparing this
    year to last; large jumps in receivables, margins, accruals and leverage are
    the fingerprints of numbers being pushed.

  * Altman Z''-Score is the book-value variant, built precisely for non-
    manufacturers and emerging/global markets. It drops the market-value term
    the original Z used, so it needs NO price feed and works across US-GAAP,
    IFRS and JGAAP with only filed figures -- the reason it is the right
    bankruptcy model for a cross-standard global scout.

  * Share issuance / dilution is THE microcap killer (Pontiff & Woodgate found
    it the single most powerful cross-sectional predictor of bad returns).
    Serial issuers fund cash burn by printing stock; a holder's claim is quietly
    diluted away regardless of what the business does.

  * Cash runway pairs with a going-concern note: a firm burning cash with under
    a year of it left is one financing away from a death spiral or a wipeout.

Every function returns a `MetricValue` (`.of` / `.missing`), divides through
`safe_div`, and never invents a missing input -- a number that cannot be built
honestly is reported missing, following base.py's design rule 1.
"""

from __future__ import annotations

import math

from scout.fundamentals.concepts import Concept
from scout.fundamentals.models import FundamentalsSnapshot
from scout.metrics.base import (
    MetricValue,
    gross_profit,
    safe_div,
    select_annual_pair,
    working_capital,
)

# --------------------------------------------------------------------------- #
# Altman Z''-Score -- bankruptcy distance, book-value variant
# --------------------------------------------------------------------------- #

# Zone cutoffs for the Z'' (four-variable, emerging-market) model. These are the
# published boundaries for this specific variant -- NOT the original Z's 2.99 /
# 1.81 -- so they must move together with the constant term (3.25) below.
_ALTMAN_SAFE = 2.6
_ALTMAN_DISTRESS = 1.1


def altman_zone(z: float) -> str:
    """Map a Z'' score to its published zone: safe / grey / distress."""
    if z > _ALTMAN_SAFE:
        return "safe"
    if z >= _ALTMAN_DISTRESS:
        return "grey"
    return "distress"


def altman_z_double_prime(snapshot: FundamentalsSnapshot) -> MetricValue:
    """Altman Z''-Score = bankruptcy distance from book figures only.

    Z'' = 3.25 + 6.56*(WC/TA) + 3.26*(RE/TA) + 6.72*(EBIT/TA)
              + 1.05*(TOTAL_EQUITY/TOTAL_LIABILITIES)

    The BOOK-equity term is what makes this the right model for a global scout:
    no market-value input, so it works identically across US-GAAP, IFRS and
    JGAAP with only what the filing already carries.

    Caveat (and a warning): the Z family was fitted on revenue-generating firms.
    For a pre-revenue company Sales/TA is ~0 and retained earnings is a deep
    accumulated deficit, so the score has little discriminating power -- we still
    compute it (the caller may want the number) but flag that it should not be
    trusted as a bankruptcy signal there.
    """
    basis = "annual" if snapshot.fiscal_period == "FY" else "interim"

    # Every term is mandatory: unlike a ratio with an optional add-back, a Z''
    # missing any of its four components is not a Z'' at all.
    wc = working_capital(snapshot)
    if wc is None:
        return MetricValue.missing(
            "altman_z_double_prime", "score", basis,
            "no working capital (current assets or liabilities missing)",
        )
    ta = snapshot.get(Concept.TOTAL_ASSETS)
    if ta is None:
        return MetricValue.missing("altman_z_double_prime", "score", basis, "no total assets reported")
    re = snapshot.get(Concept.RETAINED_EARNINGS)
    if re is None:
        return MetricValue.missing(
            "altman_z_double_prime", "score", basis, "no retained earnings reported"
        )
    ebit = snapshot.get(Concept.OPERATING_INCOME)
    if ebit is None:
        return MetricValue.missing(
            "altman_z_double_prime", "score", basis, "no operating income (EBIT) reported"
        )
    equity = snapshot.get(Concept.TOTAL_EQUITY)
    if equity is None:
        return MetricValue.missing("altman_z_double_prime", "score", basis, "no total equity reported")
    liabilities = snapshot.get(Concept.TOTAL_LIABILITIES)
    if liabilities is None:
        return MetricValue.missing(
            "altman_z_double_prime", "score", basis, "no total liabilities reported"
        )

    x1 = safe_div(wc, ta)
    x2 = safe_div(re, ta)
    x3 = safe_div(ebit, ta)
    if x1 is None or x2 is None or x3 is None:
        return MetricValue.missing("altman_z_double_prime", "score", basis, "total assets is zero")
    x4 = safe_div(equity, liabilities)
    if x4 is None:
        return MetricValue.missing("altman_z_double_prime", "score", basis, "total liabilities is zero")

    z = 3.25 + 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    warnings: list[str] = []
    # Sales/TA ~ 0 is the pre-revenue tell the caveat names; a dimensionless
    # ratio so the threshold is currency-independent.
    revenue = snapshot.get(Concept.REVENUE)
    sales_to_assets = safe_div(revenue, ta) if revenue is not None else None
    if sales_to_assets is None or abs(sales_to_assets) < 0.01:
        warnings.append("pre-revenue: Z-score has little discriminating power here")

    return MetricValue.of(
        "altman_z_double_prime", z, "score", basis,
        inputs={
            "working_capital": wc,
            "retained_earnings": re,
            "operating_income": ebit,
            "total_assets": ta,
            "total_equity": equity,
            "total_liabilities": liabilities,
        },
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Beneish M-Score -- earnings-manipulation detector
# --------------------------------------------------------------------------- #

# Above this the model calls a firm a likely manipulator. It is deliberately a
# high-recall, low-precision screen: better to over-flag and let a human read the
# filing than to let a real fraud through, which is the whole point of a gate.
_BENEISH_THRESHOLD = -1.78

# The neutral value of every index is 1.0 -- "this ratio did not change year over
# year". A microcap that never tags depreciation or SG&A is common and should not
# be un-scoreable for it, so a non-core index that cannot be built is set here.
_BENEISH_NEUTRAL = 1.0


def beneish_flags_manipulation(m: float) -> bool:
    """True when the M-Score clears the -1.78 manipulation threshold."""
    return m > _BENEISH_THRESHOLD


def _receivables_to_revenue(s: FundamentalsSnapshot) -> float | None:
    return safe_div(s.get(Concept.RECEIVABLES), s.get(Concept.REVENUE))


def _gross_margin(s: FundamentalsSnapshot) -> float | None:
    gp, _derived = gross_profit(s)
    return safe_div(gp, s.get(Concept.REVENUE))


def _asset_quality(s: FundamentalsSnapshot) -> float | None:
    """1 - (current assets + net PP&E) / total assets -- the share of assets that
    is neither current nor productive plant (soft assets a manipulator can inflate)."""
    ca = s.get(Concept.CURRENT_ASSETS)
    ppe = s.get(Concept.PPE_NET)
    if ca is None or ppe is None:
        return None
    hard_fraction = safe_div(ca + ppe, s.get(Concept.TOTAL_ASSETS))
    if hard_fraction is None:
        return None
    return 1.0 - hard_fraction


def _depreciation_rate(s: FundamentalsSnapshot) -> float | None:
    """D&A / (D&A + net PP&E). A falling rate can mean useful lives were quietly
    lengthened to slow expense recognition."""
    da = s.get(Concept.DEPRECIATION_AMORTIZATION)
    ppe = s.get(Concept.PPE_NET)
    if da is None or ppe is None:
        return None
    return safe_div(da, da + ppe)


def _sga_to_revenue(s: FundamentalsSnapshot) -> float | None:
    return safe_div(s.get(Concept.SGA_EXPENSE), s.get(Concept.REVENUE))


def _leverage(s: FundamentalsSnapshot) -> float | None:
    """(current liabilities + long-term debt) / total assets. Both debt lines are
    required rather than assuming a missing one is zero -- guessing a balance-sheet
    line is exactly what this layer refuses to do."""
    cl = s.get(Concept.CURRENT_LIABILITIES)
    ltd = s.get(Concept.LONG_TERM_DEBT)
    if cl is None or ltd is None:
        return None
    return safe_div(cl + ltd, s.get(Concept.TOTAL_ASSETS))


def _annual_pair_reason(current: FundamentalsSnapshot, prior: FundamentalsSnapshot) -> str | None:
    """Why `current`/`prior` are not a valid year-over-year annual pair, or None.

    Both must be full fiscal years, and one fiscal year (not two, not a quarter)
    apart -- a two-year jump would make every growth ratio lie, so it is rejected
    rather than silently compared across the gap.
    """
    if current.fiscal_period != "FY" or prior.fiscal_period != "FY":
        return "both snapshots must be annual (FY)"
    gap_days = (current.period_end - prior.period_end).days
    if not (250 <= gap_days <= 460):
        return "annual snapshots are not one fiscal year apart"
    return None


def beneish_m_score(current: FundamentalsSnapshot, prior: FundamentalsSnapshot) -> MetricValue:
    """8-variable Beneish M-Score comparing `current` (t) to `prior` (p).

    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI
             - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    Missing-input policy splits the indices in two. DSRI, GMI, SGI and TATA are
    CORE -- they carry the growth, margin and accrual signal the score is really
    about, so if any cannot be built the score is meaningless and we return
    missing. AQI, DEPI, SGAI and LVGI are secondary and frequently un-tagged by
    microcaps; a missing one is set to its neutral value 1.0 (no year-over-year
    change) with a warning, rather than throwing away an otherwise-good score.
    """
    reason = _annual_pair_reason(current, prior)
    if reason is not None:
        return MetricValue.missing("beneish_m_score", "score", "cross-period", reason)

    # --- Core indices: any missing -> the score is not worth reporting. ---
    dsri = safe_div(_receivables_to_revenue(current), _receivables_to_revenue(prior))
    if dsri is None:
        return MetricValue.missing(
            "beneish_m_score", "score", "cross-period",
            "cannot compute DSRI (receivables or revenue missing/zero)",
        )
    gmi = safe_div(_gross_margin(prior), _gross_margin(current))
    if gmi is None:
        return MetricValue.missing(
            "beneish_m_score", "score", "cross-period",
            "cannot compute GMI (gross margin missing/zero)",
        )
    rev_t = current.get(Concept.REVENUE)
    rev_p = prior.get(Concept.REVENUE)
    sgi = safe_div(rev_t, rev_p)
    if sgi is None:
        return MetricValue.missing(
            "beneish_m_score", "score", "cross-period",
            "cannot compute SGI (revenue missing/zero)",
        )
    ni_t = current.get(Concept.NET_INCOME)
    cfo_t = current.get(Concept.CASH_FROM_OPERATIONS)
    ta_t = current.get(Concept.TOTAL_ASSETS)
    tata = (
        safe_div(ni_t - cfo_t, ta_t)
        if ni_t is not None and cfo_t is not None
        else None
    )
    if tata is None:
        return MetricValue.missing(
            "beneish_m_score", "score", "cross-period",
            "cannot compute TATA (net income, operating cash flow or total assets missing/zero)",
        )

    # --- Secondary indices: default a missing one to neutral 1.0 and warn. ---
    warnings: list[str] = []

    def _secondary(value: float | None, name: str) -> float:
        if value is not None:
            return value
        warnings.append(f"{name} could not be computed (missing input); defaulted to neutral 1.0")
        return _BENEISH_NEUTRAL

    aqi = _secondary(safe_div(_asset_quality(current), _asset_quality(prior)), "AQI")
    depi = _secondary(safe_div(_depreciation_rate(prior), _depreciation_rate(current)), "DEPI")
    sgai = _secondary(safe_div(_sga_to_revenue(current), _sga_to_revenue(prior)), "SGAI")
    lvgi = _secondary(safe_div(_leverage(current), _leverage(prior)), "LVGI")

    m = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )

    return MetricValue.of(
        "beneish_m_score", m, "score", "cross-period",
        inputs={
            "DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
            "DEPI": depi, "SGAI": sgai, "TATA": tata, "LVGI": lvgi,
        },
        warnings=warnings,
    )


def beneish_from_history(snapshots: list[FundamentalsSnapshot]) -> MetricValue:
    """Beneish M-Score from a history, pairing the two most recent annuals."""
    pair = select_annual_pair(snapshots)
    if pair is None:
        return MetricValue.missing(
            "beneish_m_score", "score", "cross-period",
            "need two consecutive annual snapshots one fiscal year apart",
        )
    current, prior = pair
    return beneish_m_score(current, prior)


# --------------------------------------------------------------------------- #
# Share issuance / dilution rate
# --------------------------------------------------------------------------- #

# The plan's hard exclude: more than 20% year-over-year growth in the share count
# is a serial diluter funding itself by printing stock.
_DILUTION_THRESHOLD = 0.20


def is_dilution_machine(rate: float) -> bool:
    """True when share-count growth exceeds the plan's 20% YoY hard-exclude."""
    return rate > _DILUTION_THRESHOLD


def share_issuance_rate(current: FundamentalsSnapshot, prior: FundamentalsSnapshot) -> MetricValue:
    """Year-over-year change in shares outstanding, as a fraction.

    (shares_t - shares_p) / shares_p. Positive is dilution (more claims on the
    same business); negative is a buyback. Reported as a fraction -- the caller
    multiplies by 100 for display -- so 0.20 is a 20% increase.
    """
    reason = _annual_pair_reason(current, prior)
    if reason is not None:
        return MetricValue.missing("share_issuance_rate", "pct", "cross-period", reason)

    shares_t = current.get(Concept.SHARES_OUTSTANDING)
    shares_p = prior.get(Concept.SHARES_OUTSTANDING)
    if shares_t is None or shares_p is None:
        return MetricValue.missing(
            "share_issuance_rate", "pct", "cross-period", "shares outstanding missing in one period"
        )
    rate = safe_div(shares_t - shares_p, shares_p)
    if rate is None:
        return MetricValue.missing(
            "share_issuance_rate", "pct", "cross-period", "prior shares outstanding is zero"
        )
    return MetricValue.of(
        "share_issuance_rate", rate, "pct", "cross-period",
        inputs={"shares_outstanding_t": shares_t, "shares_outstanding_p": shares_p},
    )


def dilution_from_history(snapshots: list[FundamentalsSnapshot]) -> MetricValue:
    """Share-issuance rate from a history, pairing the two most recent annuals."""
    pair = select_annual_pair(snapshots)
    if pair is None:
        return MetricValue.missing(
            "share_issuance_rate", "pct", "cross-period",
            "need two consecutive annual snapshots one fiscal year apart",
        )
    current, prior = pair
    return share_issuance_rate(current, prior)


# --------------------------------------------------------------------------- #
# Cash runway
# --------------------------------------------------------------------------- #

# Pairs with a going-concern note: under a year of cash at the current burn is
# the plan's hard-exclude, because such a firm is one financing away from a
# dilutive rescue or a wipeout.
_RUNWAY_CRITICAL_MONTHS = 12.0

# Fiscal periods whose flow figure covers roughly a half-year, used to annualize
# the burn. Anything not listed and not "FY" is treated as a quarter.
_HALF_YEAR_PERIODS = frozenset({"YTD6", "H1", "Q2"})


def is_cash_critical(months: float) -> bool:
    """True when runway is under a year. `inf` (self-funding) is never critical."""
    return months < _RUNWAY_CRITICAL_MONTHS


def _burn_period_months(snapshot: FundamentalsSnapshot) -> tuple[float, list[str]]:
    """How many months the snapshot's cash-flow figure spans, with any warning.

    The operating cash flow in an annual snapshot is a full year of burn; in a
    half-year interim, six months; otherwise we treat it as a quarter. An unknown
    fiscal period is assumed annual (the least alarmist assumption) and warned,
    so a misread period cannot silently understate the runway.
    """
    period = snapshot.fiscal_period
    if period is None:
        return 12.0, ["fiscal period unknown; assumed annual (12 months) for burn"]
    if period == "FY":
        return 12.0, []
    if period in _HALF_YEAR_PERIODS:
        return 6.0, []
    return 3.0, []


def cash_runway_months(snapshot: FundamentalsSnapshot) -> MetricValue:
    """Months of cash left at the current operating burn.

    A firm with non-negative operating cash flow is self-funding: it has no
    runway risk, reported as an ok value of +inf with a note (the screen reads
    inf as "no runway concern"). Otherwise the burn is the negative operating
    cash flow, annualized to a monthly rate from the snapshot's period length, and
    runway = cash / monthly burn.
    """
    basis = "annual" if snapshot.fiscal_period == "FY" else "interim"

    cash = snapshot.get(Concept.CASH_AND_EQUIVALENTS)
    if cash is None:
        return MetricValue.missing("cash_runway_months", "count", basis, "no cash figure reported")
    cfo = snapshot.get(Concept.CASH_FROM_OPERATIONS)
    if cfo is None:
        return MetricValue.missing(
            "cash_runway_months", "count", basis, "no cash from operations reported"
        )

    if cfo >= 0:
        return MetricValue.of(
            "cash_runway_months", math.inf, "count", basis,
            inputs={"cash_and_equivalents": cash, "cash_from_operations": cfo},
            warnings=["cash-flow positive: not burning"],
        )

    period_months, warnings = _burn_period_months(snapshot)
    monthly_burn = -cfo / period_months  # cfo < 0 here, so this is a positive outflow rate
    runway = safe_div(cash, monthly_burn)
    if runway is None:
        # monthly_burn can only be zero if cfo is zero, which the >= 0 branch
        # already handled -- defensive, so a future edit cannot divide by zero.
        return MetricValue.missing("cash_runway_months", "count", basis, "burn rate is zero")
    return MetricValue.of(
        "cash_runway_months", runway, "count", basis,
        inputs={
            "cash_and_equivalents": cash,
            "cash_from_operations": cfo,
            "period_months": period_months,
            "monthly_burn": monthly_burn,
        },
        warnings=warnings,
    )
