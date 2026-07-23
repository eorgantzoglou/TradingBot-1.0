"""The contract every filing source implements.

Harvest is day-indexed because that is the shape all four free primary sources
already have: EDINET's list endpoint takes a date and nothing else, SEC
publishes daily index files, Companies House publishes a daily accounts ZIP,
and filings.xbrl.org can be filtered on processed-date. Day-indexed harvest is
also what makes the archive replayable -- "re-run 2026-07-14" is a meaningful
instruction.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class DocumentRef:
    """A document we know exists but have not downloaded yet.

    Listing is cheap and fetching is not, so the archive gets to decide (by
    consulting its manifest) whether a ref is worth fetching at all.
    """

    source: str
    doc_id: str
    """Stable within a source. SEC accession number, EDINET docID, OAM filing id."""

    filing_date: date | None = None
    """When the issuer filed. Distinct from ingest date, and the one that matters
    for point-in-time correctness -- see PLAN.md section 1.5 on filing lag."""

    form_type: str | None = None
    title: str | None = None
    url: str | None = None
    entity: dict[str, str] = field(default_factory=dict)
    """Whatever identifiers the source gives us: cik, lei, isin, ticker,
    edinet_code, corp_code, company_number. Resolved to a canonical entity later
    via GLEIF/OpenFIGI -- sources must not guess."""

    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RawDocument:
    """A fetched document, ready to archive. Payload is stored verbatim.

    We never normalize on the way in. The whole value of the archive is that it
    holds what was actually published, byte for byte, so that a future parser
    fix can be replayed over history.
    """

    ref: DocumentRef
    payload: bytes
    filename: str
    """Suggested name including extension, e.g. "0000320193-26-000012.zip"."""

    content_type: str | None = None


@runtime_checkable
class Source(Protocol):
    """A primary-source harvester.

    Implementations must be polite (go through `HttpClient`, never bypass its
    limiter), must not raise on a single bad document during listing, and must
    surface auth problems as a clear error rather than an empty result -- an
    empty day and a broken API key look identical otherwise, and the archive
    cannot be backfilled once the publisher's retention window closes.
    """

    name: str
    """Short slug used as the archive partition: "sec", "edinet", "opendart",
    "companies_house", "esef"."""

    def available(self) -> bool:
        """False when required credentials are missing, so `harvest` can skip
        this source with a warning instead of failing the whole run."""
        ...

    def list_documents(self, day: date) -> AsyncIterator[DocumentRef]:
        """Everything this source published on `day`."""
        ...

    async def fetch(self, ref: DocumentRef) -> RawDocument:
        """Download one document's payload."""
        ...
