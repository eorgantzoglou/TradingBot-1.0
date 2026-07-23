"""RawFact -> CanonicalFact: the concept-mapping "sleeper task".

Three problems have to be solved together, and getting any one wrong silently
corrupts every metric built on top:

  1. TAG HETEROGENEITY. The same economic quantity has many tags across and
     within taxonomies. 3M reports cash as
     `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents`, not the
     plain `CashAndCashEquivalentsAtCarryingValue`. So each concept has an
     ORDERED list of candidate tags and we take the first that resolves.

  2. DIMENSIONS. Revenue appears once as a consolidated total and 24 more times
     split by segment/product/geography. The metric wants the total, which is
     the NON-DIMENSIONED fact. A dimensioned fact is never accepted as a
     concept's value.

  3. PERIOD SELECTION. A single Q2 10-Q contains, ending on the same date, both
     a 90-day (discrete quarter) and a 180-day (year-to-date) duration -- and
     cash-flow items typically exist ONLY as year-to-date even when income items
     show both. So we pick the duration whose span best matches the fiscal
     period, and ALWAYS store `period_start`, so a downstream metric can see the
     exact span it is holding rather than assuming one.

This maps only what filers report. Derived figures (gross profit when only
revenue and COGS are given, TTM aggregation) belong in `metrics/`, not here --
normalization stays a faithful, auditable mapping, and every CanonicalFact keeps
`source_concept` so any number traces back to the exact tag it came from.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date

from scout.fundamentals.concepts import Concept, PeriodType
from scout.fundamentals.models import (
    CanonicalFact,
    EntityRef,
    FundamentalsSnapshot,
    RawFact,
)
from scout.fundamentals.parse.base import ParsedFiling

logger = logging.getLogger(__name__)

# Ordered candidate tags per concept, best first. "taxonomy:LocalName".
# US-GAAP and IFRS are both covered because we have real filings in both
# (3M / us-gaap, Ukrainian issuers / ifrs-full) to golden-test against. When a
# concept resolves via anything but the first candidate, normalization records a
# warning -- a snapshot leaning on fallbacks is less trustworthy and the reader
# is told.
#
# This map is expected to grow as new filers reveal tags it does not yet cover;
# that is normal and cheap. What must not happen is a wrong mapping, so additions
# should come with a filing that exercises them.
CONCEPT_MAP: dict[Concept, list[str]] = {
    Concept.REVENUE: [
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        "ifrs-full:Revenue",
        "ifrs-full:RevenueFromContractsWithCustomers",
    ],
    Concept.COST_OF_REVENUE: [
        "us-gaap:CostOfGoodsAndServicesSold",
        "us-gaap:CostOfRevenue",
        "us-gaap:CostOfGoodsSold",
        "ifrs-full:CostOfSales",
    ],
    Concept.GROSS_PROFIT: [
        "us-gaap:GrossProfit",
        "ifrs-full:GrossProfit",
    ],
    Concept.SGA_EXPENSE: [
        "us-gaap:SellingGeneralAndAdministrativeExpense",
        "us-gaap:GeneralAndAdministrativeExpense",
        "ifrs-full:SellingGeneralAndAdministrativeExpense",
    ],
    Concept.RND_EXPENSE: [
        "us-gaap:ResearchAndDevelopmentExpense",
        "ifrs-full:ResearchAndDevelopmentExpense",
    ],
    Concept.OPERATING_INCOME: [
        "us-gaap:OperatingIncomeLoss",
        "ifrs-full:ProfitLossFromOperatingActivities",
    ],
    Concept.DEPRECIATION_AMORTIZATION: [
        "us-gaap:DepreciationDepletionAndAmortization",
        "us-gaap:DepreciationAmortizationAndAccretionNet",
        "us-gaap:DepreciationAndAmortization",
        "ifrs-full:DepreciationAndAmortisationExpense",
    ],
    Concept.INTEREST_EXPENSE: [
        "us-gaap:InterestExpense",
        "us-gaap:InterestExpenseNonoperating",
        "ifrs-full:InterestExpense",
        "ifrs-full:FinanceCosts",
    ],
    Concept.INCOME_BEFORE_TAX: [
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "ifrs-full:ProfitLossBeforeTax",
    ],
    Concept.INCOME_TAX_EXPENSE: [
        "us-gaap:IncomeTaxExpenseBenefit",
        "ifrs-full:IncomeTaxExpenseContinuingOperations",
    ],
    Concept.NET_INCOME: [
        "us-gaap:NetIncomeLoss",
        "us-gaap:ProfitLoss",
        "ifrs-full:ProfitLoss",
    ],
    Concept.CASH_AND_EQUIVALENTS: [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "ifrs-full:CashAndCashEquivalents",
    ],
    Concept.SHORT_TERM_INVESTMENTS: [
        "us-gaap:ShortTermInvestments",
        "us-gaap:MarketableSecuritiesCurrent",
        "ifrs-full:CurrentInvestments",
    ],
    Concept.RECEIVABLES: [
        "us-gaap:AccountsReceivableNetCurrent",
        "us-gaap:ReceivablesNetCurrent",
        "ifrs-full:TradeAndOtherCurrentReceivables",
    ],
    Concept.INVENTORY: [
        "us-gaap:InventoryNet",
        "ifrs-full:Inventories",
    ],
    Concept.CURRENT_ASSETS: [
        "us-gaap:AssetsCurrent",
        "ifrs-full:CurrentAssets",
    ],
    Concept.PPE_NET: [
        "us-gaap:PropertyPlantAndEquipmentNet",
        "ifrs-full:PropertyPlantAndEquipment",
    ],
    Concept.GOODWILL: [
        "us-gaap:Goodwill",
        "ifrs-full:Goodwill",
    ],
    Concept.INTANGIBLE_ASSETS: [
        "us-gaap:FiniteLivedIntangibleAssetsNet",
        "us-gaap:IntangibleAssetsNetExcludingGoodwill",
        "ifrs-full:IntangibleAssetsOtherThanGoodwill",
    ],
    Concept.TOTAL_ASSETS: [
        "us-gaap:Assets",
        "ifrs-full:Assets",
    ],
    Concept.ACCOUNTS_PAYABLE: [
        "us-gaap:AccountsPayableCurrent",
        "us-gaap:AccountsPayableTradeCurrent",
        "ifrs-full:TradeAndOtherCurrentPayables",
    ],
    Concept.CURRENT_LIABILITIES: [
        "us-gaap:LiabilitiesCurrent",
        "ifrs-full:CurrentLiabilities",
    ],
    Concept.SHORT_TERM_DEBT: [
        "us-gaap:DebtCurrent",
        "us-gaap:ShortTermBorrowings",
        "us-gaap:LongTermDebtCurrent",
        "ifrs-full:CurrentBorrowings",
    ],
    Concept.LONG_TERM_DEBT: [
        "us-gaap:LongTermDebtNoncurrent",
        "us-gaap:LongTermDebt",
        "ifrs-full:NoncurrentBorrowings",
    ],
    Concept.TOTAL_LIABILITIES: [
        "us-gaap:Liabilities",
        "ifrs-full:Liabilities",
    ],
    Concept.TOTAL_EQUITY: [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "ifrs-full:Equity",
    ],
    Concept.RETAINED_EARNINGS: [
        "us-gaap:RetainedEarningsAccumulatedDeficit",
        "ifrs-full:RetainedEarnings",
    ],
    Concept.CASH_FROM_OPERATIONS: [
        "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "ifrs-full:CashFlowsFromUsedInOperatingActivities",
    ],
    Concept.CAPEX: [
        "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
        "us-gaap:PaymentsToAcquireProductiveAssets",
        "ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
    ],
    Concept.DIVIDENDS_PAID: [
        "us-gaap:PaymentsOfDividendsCommonStock",
        "us-gaap:PaymentsOfDividends",
        "ifrs-full:DividendsPaidClassifiedAsFinancingActivities",
    ],
    Concept.STOCK_ISSUANCE: [
        "us-gaap:ProceedsFromIssuanceOfCommonStock",
        "us-gaap:ProceedsFromIssuanceOrSaleOfEquity",
        "ifrs-full:ProceedsFromIssuingShares",
    ],
    Concept.STOCK_REPURCHASE: [
        "us-gaap:PaymentsForRepurchaseOfCommonStock",
        "ifrs-full:PaymentsToAcquireOrRedeemEntitysShares",
    ],
    Concept.SHARES_OUTSTANDING: [
        "dei:EntityCommonStockSharesOutstanding",
        "us-gaap:CommonStockSharesOutstanding",
        "us-gaap:CommonStockSharesIssued",
        "ifrs-full:NumberOfSharesOutstanding",
    ],
}

# Nominal span of a fiscal period, in days. Used to choose between the discrete
# quarter and the year-to-date duration that both end on the reporting date.
#
# edgartools labels interim periods as YTD3/YTD6/YTD9 (year-to-date through N
# months), not Q1/Q2/Q3 -- and that label is honest about what a filing
# features. A Q2 10-Q's headline income statement IS the 6-month cumulative, and
# its cash-flow statement is only ever the 6-month cumulative. So we target the
# year-to-date span, which keeps every line item in a snapshot on the SAME
# period. Targeting the discrete quarter would pair a 3-month income figure with
# a 6-month cash flow -- an incoherent snapshot that quietly breaks any ratio
# spanning the two statements.
_FISCAL_TARGET_DAYS: dict[str, int] = {
    "Q1": 91, "YTD3": 91,
    "Q2": 182, "H1": 182, "YTD6": 182,
    "Q3": 273, "YTD9": 273,
    "Q4": 365, "H2": 182,
    "FY": 365,
}


def normalize_filing(parsed: ParsedFiling, entity: EntityRef) -> FundamentalsSnapshot | None:
    """Turn one parsed filing into a single snapshot at its reporting date.

    Only the current reporting period is emitted. The prior-period comparatives
    in the same filing are deliberately ignored: each period is sourced from its
    own original filing, which is what keeps the eventual archive point-in-time
    (as-first-reported, not later-restated). Returns None if the filing has no
    XBRL or no identifiable reporting date -- there is nothing to normalize.
    """
    if not parsed.facts or parsed.period_of_report is None:
        return None

    period_end = parsed.period_of_report
    fiscal_period, fiscal_year, currency = _period_identity(parsed.facts, period_end)
    target_days = _FISCAL_TARGET_DAYS.get(fiscal_period or "")

    snapshot = FundamentalsSnapshot(
        entity=entity,
        period_end=period_end,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        currency=currency,
        taxonomy=parsed.taxonomy,
        accession=parsed.accession,
        filing_date=parsed.filing_date,
    )

    # Index non-dimensioned facts by concept_key once; a dimensioned fact is
    # never a concept's consolidated value, so it is excluded up front.
    by_tag: dict[str, list[RawFact]] = {}
    for fact in parsed.facts:
        if fact.is_dimensioned:
            continue
        by_tag.setdefault(fact.concept_key, []).append(fact)

    for concept in Concept:
        resolved = _resolve_concept(
            concept, by_tag, period_end, target_days, snapshot.warnings
        )
        if resolved is not None:
            snapshot.facts[concept] = _to_canonical(resolved, concept, entity, parsed, snapshot)

    if not snapshot.facts:
        snapshot.warnings.append(
            "no canonical concepts resolved -- the filing has XBRL but none of its "
            "tags matched the concept map for this reporting period"
        )
    return snapshot


def _resolve_concept(
    concept: Concept,
    by_tag: dict[str, list[RawFact]],
    period_end: date,
    target_days: int | None,
    warnings: list[str],
) -> RawFact | None:
    """First candidate tag that yields a fact at the right period, in priority
    order. Records a warning when a non-primary candidate is used."""
    want = concept.meta.period_type
    for priority, tag in enumerate(CONCEPT_MAP[concept]):
        facts = [f for f in by_tag.get(tag, []) if f.period_type == want]
        if not facts:
            continue

        chosen = _select_period(concept, facts, period_end, target_days, warnings)
        if chosen is None:
            continue

        if priority > 0:
            warnings.append(
                f"{concept.value}: resolved via fallback tag {tag!r} "
                f"(primary {CONCEPT_MAP[concept][0]!r} absent)"
            )
        return chosen
    return None


def _select_period(
    concept: Concept,
    facts: list[RawFact],
    period_end: date,
    target_days: int | None,
    warnings: list[str],
) -> RawFact | None:
    """Pick the one fact that belongs to this snapshot's period.

    Instants must sit exactly on the reporting date. Durations must end on it;
    among several (discrete quarter vs year-to-date), the span closest to the
    fiscal-period target wins, and a large deviation from target is flagged so a
    metric knows it is holding, say, a year-to-date figure where it expected a
    quarter.
    """
    if concept.meta.period_type == PeriodType.INSTANT:
        return _select_instant(concept, facts, period_end, warnings)

    ending = [f for f in facts if f.period_end == period_end and f.period_start]
    if not ending:
        return None

    def span_of(fact: RawFact) -> int:
        return (fact.period_end - fact.period_start).days  # type: ignore[operator]

    if target_days is None:
        # No fiscal period to match against: the shortest span ending here is
        # the most granular (discrete) period available.
        chosen = min(ending, key=span_of)
    else:
        chosen = min(ending, key=lambda f: abs(span_of(f) - target_days))
        if abs(span_of(chosen) - target_days) > 45:
            warnings.append(
                f"{concept.value}: closest available span is {span_of(chosen)} days for a "
                f"{target_days}-day fiscal period -- period may not line up with peers"
            )

    # Ambiguity means two facts of the SAME span disagreeing. Facts of different
    # spans ending here (the discrete quarter and the year-to-date) are expected,
    # not a conflict, so they must not be compared against each other.
    same_span = [f for f in ending if span_of(f) == span_of(chosen)]
    _warn_if_ambiguous(concept, same_span, chosen, warnings)
    return chosen


def _select_instant(
    concept: Concept, facts: list[RawFact], period_end: date, warnings: list[str]
) -> RawFact | None:
    """Instant facts sit on the reporting date -- except shares outstanding,
    which is stated 'as of' the cover date near filing and can miss it by days;
    for that concept we take the instant nearest the reporting date."""
    exact = [f for f in facts if f.period_end == period_end]
    if exact:
        _warn_if_ambiguous(concept, exact, exact[0], warnings)
        return exact[0]

    if concept == Concept.SHARES_OUTSTANDING and facts:
        nearest = min(facts, key=lambda f: abs((f.period_end - period_end).days))
        warnings.append(
            f"shares_outstanding: no fact on {period_end}, used cover-date value "
            f"as of {nearest.period_end}"
        )
        return nearest
    return None


def _warn_if_ambiguous(
    concept: Concept, candidates: list[RawFact], chosen: RawFact, warnings: list[str]
) -> None:
    """More than one non-dimensioned fact for the same tag and period means the
    filing is internally inconsistent (or we mis-filtered). Keep the first but
    say so -- a silent pick would hide a data problem."""
    distinct = {round(f.value, 4) for f in candidates}
    if len(distinct) > 1:
        warnings.append(
            f"{concept.value}: {len(distinct)} differing non-dimensioned values for the "
            f"same period ({sorted(distinct)[:4]}); used {chosen.value}"
        )


def _to_canonical(
    fact: RawFact,
    concept: Concept,
    entity: EntityRef,
    parsed: ParsedFiling,
    snapshot: FundamentalsSnapshot,
) -> CanonicalFact:
    currency = fact.unit if fact.unit and fact.unit.isalpha() and len(fact.unit) == 3 else None
    # A canonical fact belongs to its SNAPSHOT's period key, so it carries the
    # snapshot's resolved fiscal_period/year -- not the raw fact's own, which is
    # often absent (IFRS) or inconsistent between statements (a balance-sheet
    # instant tagged differently from the income statement it sits beside). The
    # raw fact's original metadata is preserved in raw_facts for provenance;
    # here, using it would break the store's join between a snapshot header and
    # its facts. `period_start`/`period_end` stay the fact's own, so the exact
    # span each figure covers is never lost.
    return CanonicalFact(
        entity_id=entity.entity_id,
        concept=concept,
        value=fact.value,
        currency=currency,
        period_end=snapshot.period_end,
        period_start=fact.period_start,
        fiscal_year=snapshot.fiscal_year,
        fiscal_period=snapshot.fiscal_period,
        accession=parsed.accession,
        source_concept=fact.concept_key,
        taxonomy=fact.taxonomy,
    )


def _period_identity(
    facts: list[RawFact], period_end: date
) -> tuple[str | None, int | None, str | None]:
    """Best-effort fiscal period, year and reporting currency for the snapshot.

    Taken by majority vote over the facts that sit on the reporting date, since
    individual facts occasionally carry stale or blank fiscal metadata.
    """
    at_end = [f for f in facts if f.period_end == period_end]
    fiscal_period = _majority(f.fiscal_period for f in at_end)
    fiscal_year = _majority(f.fiscal_year for f in at_end)
    currency = _majority(
        f.unit for f in at_end if f.unit and f.unit.isalpha() and len(f.unit) == 3
    )

    # Filers outside the US-GAAP/dei world (ESEF/IFRS) often carry no fiscal
    # metadata at all. If an annual-length duration ends on the reporting date,
    # the filing is an annual report -- infer FY so the period selector has a
    # target and downstream can tell annual snapshots from interim ones.
    if fiscal_period is None:
        durations = [
            (f.period_end - f.period_start).days
            for f in at_end
            if f.period_start is not None
        ]
        if durations and any(350 <= d <= 380 for d in durations):
            fiscal_period = "FY"
    if fiscal_year is None:
        fiscal_year = period_end.year

    return fiscal_period, fiscal_year, currency


def _majority(values):  # type: ignore[no-untyped-def]
    counter = Counter(v for v in values if v is not None)
    if not counter:
        return None
    return counter.most_common(1)[0][0]
