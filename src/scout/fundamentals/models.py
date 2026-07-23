"""Data models for the fundamentals layer.

Three levels, most-raw to most-cooked:

  RawFact         one XBRL fact exactly as filed (tag, value, context, dims).
                  Provenance-complete; never discards anything.
  CanonicalFact   one RawFact mapped to a `Concept`, de-dimensioned, one value
                  per (entity, period, concept). What normalization produces.
  FundamentalsSnapshot
                  every canonical fact for one (entity, period_end,
                  fiscal_period) collapsed into a keyed bag. What metrics read.

The RawFact -> CanonicalFact step is the "sleeper task" from PLAN.md: tag usage
varies wildly across small filers and across JGAAP/IFRS/US-GAAP, so the mapping
carries ordered fallbacks and must be golden-tested against real filings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from scout.fundamentals.concepts import Concept, PeriodType


@dataclass(frozen=True, slots=True)
class RawFact:
    """One XBRL fact as filed. The unit of provenance.

    `taxonomy` + `local_name` together are the raw concept (e.g. "us-gaap",
    "Revenues" or "ifrs-full", "Revenue"). Keeping the prefix separate is what
    lets normalization apply taxonomy-specific mappings without string-splitting
    everywhere.
    """

    accession: str
    """The archive doc_id this fact came from -- the provenance anchor."""

    taxonomy: str
    """Namespace prefix: "us-gaap", "ifrs-full", "jpcrp", a national extension,
    or a company-specific extension namespace."""

    local_name: str
    """The concept local name, e.g. "Revenues"."""

    value: float
    unit: str | None
    """Currency ISO code for monetary facts (e.g. "USD"), "shares", "pure" for
    ratios, or None."""

    period_type: PeriodType
    period_start: date | None
    """None for instant facts."""

    period_end: date
    """For instant facts this is the balance date; for durations, the period end."""

    is_dimensioned: bool
    """True if the fact is broken down by a dimension (segment, product,
    geography). The consolidated total is the non-dimensioned fact, and that is
    almost always the one a metric wants -- Revenue split 24 ways plus one total
    is 25 facts for one number."""

    decimals: int | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    """FY, Q1..Q4, or H1/H2 where the filer reports halves."""

    @property
    def concept_key(self) -> str:
        return f"{self.taxonomy}:{self.local_name}"


@dataclass(frozen=True, slots=True)
class CanonicalFact:
    """A RawFact resolved to a canonical `Concept`.

    `source_concept` is retained so a surprising number can always be traced
    back to the exact tag it came from -- provenance survives normalization.
    """

    entity_id: str
    """Stable entity key within a source: CIK for SEC, LEI/EDRPOU for ESEF,
    EDINET code for Japan, corp_code for Korea. Cross-source resolution to a
    single global entity happens later, in the entity layer."""

    concept: Concept
    value: float
    currency: str | None
    period_end: date
    period_start: date | None
    fiscal_year: int | None
    fiscal_period: str | None

    accession: str
    source_concept: str
    """The raw "taxonomy:local_name" this was mapped from."""

    taxonomy: str
    """The accounting standard this came under, so metrics can refuse to compare
    across incomparable standards (PLAN.md section 1.4)."""


@dataclass(frozen=True, slots=True)
class EntityRef:
    """Identity of the filer, carried alongside a snapshot.

    `identifier_scheme` matters: filings.xbrl.org returns an LEI for most ESEF
    countries but an EDRPOU registry number for Ukraine, and treating the latter
    as an LEI (as the raw harvest metadata does) would mis-join it against
    GLEIF. The scheme is recorded so that resolution can be honest about it.
    """

    source: str
    entity_id: str
    identifier_scheme: str
    """"cik", "lei", "edrpou", "edinet", "corp_code", ..."""

    name: str | None = None
    country: str | None = None


@dataclass(slots=True)
class FundamentalsSnapshot:
    """All canonical facts for one entity at one period end.

    A snapshot is scoped to a single (period_end, fiscal_period) so that metrics
    never accidentally mix a Q2 income figure with an annual balance. Balance and
    cover facts are instants at `period_end`; income and cash-flow facts are the
    duration ending at `period_end`.
    """

    entity: EntityRef
    period_end: date
    fiscal_year: int | None
    fiscal_period: str | None
    currency: str | None
    taxonomy: str
    accession: str
    filing_date: date | None
    facts: dict[Concept, CanonicalFact] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    """Non-fatal normalization issues: a concept resolved via a low-priority
    fallback, a fact dropped for a period-type mismatch, an ambiguous duplicate.
    Surfaced the same way the extractor warnings are -- to screen and to the
    eventual memo -- because a snapshot built on fallbacks is less trustworthy
    and the reader deserves to know."""

    def get(self, concept: Concept) -> float | None:
        fact = self.facts.get(concept)
        return fact.value if fact else None

    def has(self, concept: Concept) -> bool:
        return concept in self.facts
