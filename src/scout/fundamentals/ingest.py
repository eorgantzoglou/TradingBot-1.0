"""Archive -> DuckDB: parse each stored filing, normalize it, store the result.

Reads the archive manifest, routes each document to the parser for its source,
normalizes the facts into a snapshot, and writes both the raw facts (provenance)
and the canonical snapshot (what metrics read). Idempotent: a document already
ingested is skipped unless `reingest=True`, and re-ingesting replaces rather than
duplicates.

Parsing is CPU-bound and independent per document, so it fans out across a
thread pool; the store write is serialized behind it (one DuckDB writer). This
mirrors the harvest shape -- parallel where the work is independent, serial at
the single mutable resource.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date

from scout.config import Config
from scout.data.archive import Archive, StoredDocument
from scout.fundamentals.models import EntityRef
from scout.fundamentals.normalize import normalize_filing
from scout.fundamentals.parse.base import FilingParser
from scout.fundamentals.parse.esef import EsefJsonParser
from scout.fundamentals.parse.sec import SecXbrlParser
from scout.fundamentals.store import FundamentalsStore

logger = logging.getLogger(__name__)


def build_parsers() -> dict[str, FilingParser]:
    """Parsers keyed by archive source slug. EDINET/OpenDART/Companies House
    parsers land here once there are archived filings from them to verify
    against -- building a parser we cannot exercise against real bytes is the
    mistake that shipped the fabricated-fixture SEC bug."""
    return {"sec": SecXbrlParser(), "esef": EsefJsonParser()}


@dataclass(slots=True)
class IngestResult:
    considered: int = 0
    parsed: int = 0
    snapshots: int = 0
    no_xbrl: int = 0
    """Documents that parsed cleanly but carried no financial statements -- an
    8-K, a cover-only filing. Expected, not an error."""

    skipped_existing: int = 0
    unsupported: int = 0
    """No parser for the source, or the parser declined the form type."""

    failed: int = 0
    errors: list[str] = field(default_factory=list)
    warnings_total: int = 0


def _entity_from(document: StoredDocument, entity_name: str | None, entity_id: str) -> EntityRef:
    """Build the entity identity, honest about which identifier scheme applies.

    filings.xbrl.org labels its identifier "lei", but for some jurisdictions
    (Ukraine) it is really an EDRPOU registry number, and a real LEI is 20
    alphanumeric characters. We record what the value actually is rather than
    trusting the label, so later cross-source resolution against GLEIF does not
    mis-join a national id as an LEI.
    """
    if document.source == "sec":
        return EntityRef("sec", entity_id, "cik", name=entity_name,
                         country=document.entity.get("country"))

    raw_id = entity_id or document.entity.get("lei", "")
    is_real_lei = len(raw_id) == 20 and raw_id.isalnum()
    scheme = "lei" if is_real_lei else "national_id"
    return EntityRef(document.source, raw_id, scheme, name=entity_name,
                     country=document.entity.get("country"))


def _parse_filing_date(document: StoredDocument) -> date | None:
    if not document.filing_date:
        return None
    try:
        return date.fromisoformat(document.filing_date)
    except ValueError:
        return None


async def ingest(
    config: Config,
    *,
    sources: list[str] | None = None,
    limit: int | None = None,
    reingest: bool = False,
    concurrency: int | None = None,
) -> IngestResult:
    """Ingest archived filings into the fundamentals store."""
    archive = Archive(config.archive_dir)
    parsers = build_parsers()
    result = IngestResult()

    config.data_dir.mkdir(parents=True, exist_ok=True)
    workers = concurrency or config.concurrency
    semaphore = asyncio.Semaphore(workers)

    with FundamentalsStore(config.db_path) as store:
        store.initialize()

        # Filter the manifest to documents worth parsing before doing any work.
        todo: list[StoredDocument] = []
        for document in archive.iter_manifest():
            if sources and document.source not in sources:
                continue
            parser = parsers.get(document.source)
            if parser is None:
                result.unsupported += 1
                continue
            if not parser.can_parse(
                form_type=document.form_type,
                content_type=document.content_type,
                filename=document.filename,
            ):
                result.unsupported += 1
                continue
            result.considered += 1
            if not reingest and store.has_accession(document.doc_id):
                result.skipped_existing += 1
                continue
            todo.append(document)
            if limit is not None and len(todo) >= limit:
                break

        async def parse_one(document: StoredDocument):  # type: ignore[no-untyped-def]
            parser = parsers[document.source]
            payload = archive.read_payload(document)
            entity_hint = {k: str(v) for k, v in document.entity.items()}
            # Parsing is sync and CPU-heavy; keep the event loop free.
            async with semaphore:
                parsed = await asyncio.to_thread(
                    parser.parse, payload, accession=document.doc_id, entity_hint=entity_hint
                )
            return document, parsed

        # gather preserves order, so each outcome lines up with its todo entry
        # and a failure can be attributed to the right document.
        outcomes = await asyncio.gather(
            *(parse_one(document) for document in todo), return_exceptions=True
        )

        for document, outcome in zip(todo, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                result.failed += 1
                if len(result.errors) < 10:
                    result.errors.append(f"{document.doc_id}: {outcome}")
                logger.warning("ingest: parsing %s failed: %s", document.doc_id, outcome)
                continue

            _document, parsed = outcome
            result.parsed += 1
            store.put_raw_facts(parsed.facts)

            if not parsed.has_xbrl:
                result.no_xbrl += 1
                continue

            entity = _entity_from(document, parsed.entity_name, parsed.entity_id)
            snapshot = normalize_filing(parsed, entity)
            if snapshot is None or not snapshot.facts:
                result.no_xbrl += 1
                continue

            snapshot.filing_date = snapshot.filing_date or _parse_filing_date(document)
            store.put_snapshot(snapshot)
            result.snapshots += 1
            result.warnings_total += len(snapshot.warnings)

    return result
