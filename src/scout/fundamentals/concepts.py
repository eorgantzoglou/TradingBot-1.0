"""The canonical financial concept vocabulary.

This is the contract that makes a Japanese JGAAP filing, a US-GAAP 10-K and an
IFRS ESEF report comparable. Every raw XBRL tag from every taxonomy is mapped to
one of these concepts (in `normalize.py`); everything downstream -- metrics,
screening -- speaks only `Concept`, never a raw tag.

Deliberately a *small* vocabulary. The plan (PLAN.md section 1.4) is explicit
that P/E, ROE and EBITDA multiples are not comparable across accounting
standards, so we do not try to reconstruct them here. We capture the line items
that the phase-3 metrics actually need -- Piotroski F, Beneish M, Altman Z,
GP/A, EV/EBIT, the dilution screen -- and nothing speculative. Adding a concept
is cheap; removing one that metrics depend on is not, so the bar for adding is
"a named metric needs it".

Sign convention: values are stored with their natural reporting sign as filed.
Expenses are positive, contra-items follow the filer. Normalization records the
XBRL `balance` (debit/credit) so metrics can reason about sign explicitly rather
than guessing -- an LLM guessing signs is exactly the arithmetic we refuse to
delegate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Statement(StrEnum):
    """Which primary statement a concept belongs to.

    COVER holds entity/period facts that are not financial line items but are
    needed to place the numbers in time (shares outstanding is here because it
    is reported on the cover of every filing, not only in equity).
    """

    INCOME = "income"
    BALANCE = "balance"
    CASHFLOW = "cashflow"
    COVER = "cover"


class PeriodType(StrEnum):
    """Instant facts are a balance at a point in time (balance-sheet items);
    duration facts cover a span (income and cash-flow items).

    This is not cosmetic: comparing an instant to a duration, or summing two
    overlapping durations, is a category error, and the normalizer uses this to
    reject a fact that arrives with the wrong period shape for its concept.
    """

    INSTANT = "instant"
    DURATION = "duration"


@dataclass(frozen=True, slots=True)
class ConceptMeta:
    statement: Statement
    period_type: PeriodType
    description: str


class Concept(StrEnum):
    """Canonical line items. Values are the storage keys -- keep them stable."""

    # --- Income statement (all DURATION) ---
    REVENUE = "revenue"
    COST_OF_REVENUE = "cost_of_revenue"
    GROSS_PROFIT = "gross_profit"
    SGA_EXPENSE = "sga_expense"
    RND_EXPENSE = "rnd_expense"
    OPERATING_INCOME = "operating_income"
    """Operating income / EBIT. The primary earnings figure for EV/EBIT and
    Altman Z -- chosen over net income precisely because it is the most
    comparable earnings line across JGAAP/IFRS/US-GAAP."""
    DEPRECIATION_AMORTIZATION = "depreciation_amortization"
    INTEREST_EXPENSE = "interest_expense"
    INCOME_BEFORE_TAX = "income_before_tax"
    INCOME_TAX_EXPENSE = "income_tax_expense"
    NET_INCOME = "net_income"

    # --- Balance sheet (all INSTANT) ---
    CASH_AND_EQUIVALENTS = "cash_and_equivalents"
    SHORT_TERM_INVESTMENTS = "short_term_investments"
    RECEIVABLES = "receivables"
    INVENTORY = "inventory"
    CURRENT_ASSETS = "current_assets"
    PPE_NET = "ppe_net"
    GOODWILL = "goodwill"
    INTANGIBLE_ASSETS = "intangible_assets"
    TOTAL_ASSETS = "total_assets"
    ACCOUNTS_PAYABLE = "accounts_payable"
    CURRENT_LIABILITIES = "current_liabilities"
    SHORT_TERM_DEBT = "short_term_debt"
    LONG_TERM_DEBT = "long_term_debt"
    TOTAL_LIABILITIES = "total_liabilities"
    TOTAL_EQUITY = "total_equity"
    RETAINED_EARNINGS = "retained_earnings"

    # --- Cash flow (all DURATION) ---
    CASH_FROM_OPERATIONS = "cash_from_operations"
    CAPEX = "capex"
    DIVIDENDS_PAID = "dividends_paid"
    STOCK_ISSUANCE = "stock_issuance"
    """Proceeds from issuing shares. Piotroski signal #7 and the single most
    important microcap red flag (Pontiff & Woodgate) -- the dilution fingerprint."""
    STOCK_REPURCHASE = "stock_repurchase"

    # --- Cover / shares (INSTANT) ---
    SHARES_OUTSTANDING = "shares_outstanding"
    """Common shares outstanding. Combined with a later change series this is the
    dilution rate; combined with price it is market cap."""

    @property
    def meta(self) -> ConceptMeta:
        return _META[self]


_META: dict[Concept, ConceptMeta] = {
    Concept.REVENUE: ConceptMeta(Statement.INCOME, PeriodType.DURATION, "Total net revenue / sales"),
    Concept.COST_OF_REVENUE: ConceptMeta(Statement.INCOME, PeriodType.DURATION, "Cost of goods/services sold"),
    Concept.GROSS_PROFIT: ConceptMeta(Statement.INCOME, PeriodType.DURATION, "Revenue minus cost of revenue"),
    Concept.SGA_EXPENSE: ConceptMeta(Statement.INCOME, PeriodType.DURATION, "Selling, general & administrative expense"),
    Concept.RND_EXPENSE: ConceptMeta(Statement.INCOME, PeriodType.DURATION, "Research & development expense"),
    Concept.OPERATING_INCOME: ConceptMeta(Statement.INCOME, PeriodType.DURATION, "Operating income (EBIT)"),
    Concept.DEPRECIATION_AMORTIZATION: ConceptMeta(Statement.INCOME, PeriodType.DURATION, "Depreciation & amortization"),
    Concept.INTEREST_EXPENSE: ConceptMeta(Statement.INCOME, PeriodType.DURATION, "Interest expense"),
    Concept.INCOME_BEFORE_TAX: ConceptMeta(Statement.INCOME, PeriodType.DURATION, "Pre-tax income"),
    Concept.INCOME_TAX_EXPENSE: ConceptMeta(Statement.INCOME, PeriodType.DURATION, "Income tax expense/benefit"),
    Concept.NET_INCOME: ConceptMeta(Statement.INCOME, PeriodType.DURATION, "Net income attributable to the entity"),
    Concept.CASH_AND_EQUIVALENTS: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Cash and cash equivalents"),
    Concept.SHORT_TERM_INVESTMENTS: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Short-term / marketable investments"),
    Concept.RECEIVABLES: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Trade and other receivables, net"),
    Concept.INVENTORY: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Inventories, net"),
    Concept.CURRENT_ASSETS: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Total current assets"),
    Concept.PPE_NET: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Property, plant & equipment, net"),
    Concept.GOODWILL: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Goodwill"),
    Concept.INTANGIBLE_ASSETS: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Intangible assets excluding goodwill"),
    Concept.TOTAL_ASSETS: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Total assets"),
    Concept.ACCOUNTS_PAYABLE: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Trade and other payables"),
    Concept.CURRENT_LIABILITIES: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Total current liabilities"),
    Concept.SHORT_TERM_DEBT: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Short-term debt + current portion of long-term debt"),
    Concept.LONG_TERM_DEBT: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Long-term debt excluding current portion"),
    Concept.TOTAL_LIABILITIES: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Total liabilities"),
    Concept.TOTAL_EQUITY: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Total shareholders' equity"),
    Concept.RETAINED_EARNINGS: ConceptMeta(Statement.BALANCE, PeriodType.INSTANT, "Retained earnings / accumulated deficit"),
    Concept.CASH_FROM_OPERATIONS: ConceptMeta(Statement.CASHFLOW, PeriodType.DURATION, "Net cash from operating activities"),
    Concept.CAPEX: ConceptMeta(Statement.CASHFLOW, PeriodType.DURATION, "Capital expenditure (purchases of PP&E)"),
    Concept.DIVIDENDS_PAID: ConceptMeta(Statement.CASHFLOW, PeriodType.DURATION, "Cash dividends paid"),
    Concept.STOCK_ISSUANCE: ConceptMeta(Statement.CASHFLOW, PeriodType.DURATION, "Proceeds from issuance of equity"),
    Concept.STOCK_REPURCHASE: ConceptMeta(Statement.CASHFLOW, PeriodType.DURATION, "Cash paid to repurchase equity"),
    Concept.SHARES_OUTSTANDING: ConceptMeta(Statement.COVER, PeriodType.INSTANT, "Common shares outstanding"),
}

# Fail fast if a concept is ever added without metadata -- a missing entry would
# otherwise surface as a KeyError deep in normalization.
_missing = set(Concept) - set(_META)
if _missing:  # pragma: no cover - guards against edit mistakes
    raise RuntimeError(f"Concept(s) missing ConceptMeta: {sorted(c.value for c in _missing)}")
