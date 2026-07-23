"""Types for the screen: profiles, exclusion checks, cohorts, ranked candidates.

The screen is deterministic and separate from the LLM by design (PLAN.md rule 1
and 2): it applies hard excludes, assigns each survivor to a peer cohort, and
ranks within that cohort on a cheap x quality x safety composite. The LLM never
runs until a name has already survived this. And the framework is honest about
what it could not check -- an exclude that lacks its input data reports
INSUFFICIENT rather than silently passing, because "we did not verify this" and
"this is fine" are different claims and conflating them is how a dilution machine
slips through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class FormerName:
    name: str
    from_date: date | None
    to_date: date | None


@dataclass(frozen=True, slots=True)
class EntityProfile:
    """Identity, classification and filing-history facts for one entity.

    Sourced from the SEC submissions API for SEC filers; minimal (country only)
    for others until their own registries are wired in. Every field is optional
    because non-SEC entities and sparse filers will lack most of them, and the
    screen must degrade rather than fail.
    """

    entity_id: str
    source: str
    name: str | None = None
    country: str | None = None
    """Listing/incorporation country, e.g. "US", "UA", "GB"."""

    sic: str | None = None
    sic_description: str | None = None
    sector: str | None = None
    """Coarse sector derived from SIC (see cohorts.py)."""

    is_excluded_sector: bool = False
    """Financials and utilities -- EV/EBIT is meaningless for them, so the plan
    excludes them from the ranked universe."""

    tickers: tuple[str, ...] = ()
    exchanges: tuple[str, ...] = ()
    filer_category: str | None = None
    """SEC "category": accelerated filer status. Microcaps are typically
    "Non-accelerated filer" or "Smaller reporting company"."""

    entity_type: str | None = None
    """"operating" vs other -- a non-operating entity is a shell candidate."""

    former_names: tuple[FormerName, ...] = ()
    fiscal_year_end: str | None = None

    # Filing-history derived flags. None = not determinable from available data.
    has_recent_late_filing: bool | None = None
    """An NT 10-K / NT 10-Q in the recent history -- delinquency."""

    name_changed_within_months: int | None = None
    """Months since the most recent former-name change ended, if any. A recent
    change is a classic shell-hijack pattern."""

    most_recent_form: str | None = None
    most_recent_filing_date: date | None = None


class Decision(StrEnum):
    EXCLUDE = "exclude"
    PASS = "pass"
    INSUFFICIENT = "insufficient_data"
    """The rule's input data was absent, so it could not be evaluated. Recorded,
    surfaced, and NOT treated as a pass."""


@dataclass(frozen=True, slots=True)
class ExcludeCheck:
    rule: str
    decision: Decision
    reason: str


@dataclass(frozen=True, slots=True)
class CohortKey:
    """Peers are ranked only against peers. Cross-cohort ranking is forbidden
    because P/E, ROE and EBITDA multiples are not comparable across accounting
    standards, and a Japanese microcap and a US biotech are not peers."""

    country: str
    accounting_standard: str
    sector: str

    def label(self) -> str:
        return f"{self.country} / {self.accounting_standard} / {self.sector}"


@dataclass(slots=True)
class ScoredCandidate:
    entity_id: str
    name: str | None
    cohort: CohortKey
    composite: float | None
    """Combined within-cohort score, higher = more attractive. None when the
    cohort is too small to z-score meaningfully."""

    cheap: float | None = None
    quality: float | None = None
    safety: float | None = None
    underfollowed: bool = False
    """A conditioning flag (low coverage / low institutional ownership), never a
    ranking factor -- the neglected-firm premium disappears once you control for
    size, so this only breaks ties among names already cheap and healthy."""

    metric_values: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExcludedCandidate:
    entity_id: str
    name: str | None
    reasons: list[str]


@dataclass(slots=True)
class ScreenResult:
    ranked: list[ScoredCandidate]
    excluded: list[ExcludedCandidate]
    insufficient_checks: dict[str, int] = field(default_factory=dict)
    """Per-rule count of entities where the exclude could not be evaluated for
    lack of data -- the screen's blind spots, reported rather than hidden."""

    universe_size: int = 0
    cohort_sizes: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
