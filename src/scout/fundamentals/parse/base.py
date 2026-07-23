"""The parse boundary: archived bytes -> RawFacts.

One protocol so the normalizer and store never care whether a filing came from
edgartools (US-GAAP SGML) or a raw xBRL-JSON document (ESEF). Parsers do no
normalization -- they translate a filing's own facts into `RawFact`s verbatim,
preserving taxonomy, dimensions and periods. Mapping to canonical concepts is
the normalizer's job, kept separate so a mapping fix never requires re-parsing.

Parsers operate on ARCHIVED BYTES, offline. This was verified empirically before
the design was fixed: edgartools' `XBRL.from_filing(FilingSGML.from_text(...))`
reconstructs the full fact set from a stored submission with no network. The
archive is the source of truth; nothing here re-fetches from a publisher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

from scout.fundamentals.models import RawFact


@dataclass(slots=True)
class ParsedFiling:
    """The outcome of parsing one archived document.

    A filing that carries no XBRL (a plain-text 8-K, a PDF-only OAM filing) is
    not an error -- it yields zero facts and a note. The caller decides whether
    that matters; for fundamentals it simply means nothing to ingest.
    """

    accession: str
    source: str
    entity_id: str
    entity_name: str | None
    taxonomy: str
    """The dominant taxonomy of the filing: "us-gaap", "ifrs-full", etc. Facts
    may still carry other prefixes (extensions, dei/cover tags); this is the
    accounting standard the statements are prepared under."""

    period_of_report: date | None
    filing_date: date | None
    facts: list[RawFact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_xbrl(self) -> bool:
        return bool(self.facts)


@runtime_checkable
class FilingParser(Protocol):
    """Parses one source's archived documents into `RawFact`s.

    Implementations must:
      - work entirely offline from the provided bytes;
      - never raise on a filing that simply lacks XBRL -- return an empty
        `ParsedFiling` with a warning instead;
      - preserve dimensioned facts (with `is_dimensioned=True`) rather than
        dropping them, because the normalizer needs to distinguish the
        consolidated total from its segment breakdown, and only it knows which
        concept wants which.
    """

    source: str

    def can_parse(self, *, form_type: str | None, content_type: str | None, filename: str) -> bool:
        """Cheap pre-check so the ingester skips documents with no fundamentals
        (Form 4, an 8-K with no financials) without paying to parse them."""
        ...

    def parse(self, payload: bytes, *, accession: str, entity_hint: dict[str, str]) -> ParsedFiling:
        """Parse archived bytes. `entity_hint` carries whatever the harvest
        manifest recorded (cik, lei, country, ...), used only to fill identity
        when the filing itself does not state it."""
        ...
