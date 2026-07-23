"""Daily harvest orchestration: list -> fetch -> archive, across all sources.

The orchestration shape is deliberately the one from PLAN.md section 4.5 --
`asyncio.gather` with a `Semaphore` and `return_exceptions=True` -- because that
is all a bounded fan-out needs. One unreachable source, one malformed document
or one 500 from a publisher must never cost us the rest of the day's harvest:
the publishers purge on 30- and 60-day windows, so a failed run is not something
we can simply repeat later.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from scout.config import Config
from scout.data.archive import Archive
from scout.data.http import HttpClient
from scout.data.sources.base import DocumentRef, Source
from scout.data.sources.companies_house import CompaniesHouseSource
from scout.data.sources.edinet import EdinetSource
from scout.data.sources.esef import EsefSource
from scout.data.sources.opendart import OpenDartSource
from scout.data.sources.sec import SecSource

logger = logging.getLogger(__name__)

ALL_SOURCES = ("sec", "esef", "edinet", "opendart", "companies_house")


@dataclass(slots=True)
class DayResult:
    """What one (source, day) pair produced. Errors are collected, not raised."""

    source: str
    day: date
    listed: int = 0
    stored: int = 0
    revisions: int = 0
    unchanged: int = 0
    """Already held with identical content -- the steady state on a re-run."""

    skipped_known: int = 0
    """Seen before, so never fetched. This is the number that keeps a backfill
    affordable."""

    failed: int = 0
    errors: list[str] = field(default_factory=list)
    skipped_reason: str | None = None
    """Set when the source did not run at all (missing credentials)."""

    @property
    def ok(self) -> bool:
        return not self.errors and self.skipped_reason is None


def build_sources(
    config: Config, http: HttpClient, names: Sequence[str] | None = None
) -> list[Source]:
    """Instantiate the requested sources. Unknown names raise -- a typo in
    `--source` must not silently harvest nothing."""
    wanted = tuple(names) if names else ALL_SOURCES
    unknown = set(wanted) - set(ALL_SOURCES)
    if unknown:
        raise ValueError(
            f"Unknown source(s): {', '.join(sorted(unknown))}. "
            f"Known sources: {', '.join(ALL_SOURCES)}."
        )

    builders: dict[str, Callable[[], Source]] = {
        "sec": lambda: SecSource(http),
        "esef": lambda: EsefSource(http),
        "edinet": lambda: EdinetSource(http, config.credentials),
        "opendart": lambda: OpenDartSource(http, config.credentials),
        "companies_house": lambda: CompaniesHouseSource(http),
    }
    return [builders[name]() for name in wanted]


async def harvest_day(
    source: Source,
    archive: Archive,
    day: date,
    *,
    concurrency: int = 8,
    limit: int | None = None,
    refetch_known: bool = False,
) -> DayResult:
    """Harvest one source for one day.

    By default a document whose id we have already stored is never re-fetched.
    That is what makes a multi-year backfill affordable, and it is safe because
    every source here uses immutable document ids (SEC accession numbers,
    EDINET docIDs, OpenDART rcept_no). Pass `refetch_known=True` to check for
    in-place revisions -- the archive still stores a revision as a new version
    rather than overwriting.
    """
    result = DayResult(source=source.name, day=day)

    if not source.available():
        result.skipped_reason = "missing credentials"
        return result

    refs: list[DocumentRef] = []
    try:
        async for ref in source.list_documents(day):
            result.listed += 1
            if not refetch_known and archive.has_seen(ref.source, ref.doc_id):
                result.skipped_known += 1
                continue
            refs.append(ref)
            if limit is not None and len(refs) >= limit:
                break
    except Exception as exc:
        # Listing failed outright: bad credentials, API change, publisher down.
        # Record it and return -- a partial day is still worth keeping.
        result.errors.append(f"listing failed: {exc}")
        logger.warning("%s: listing %s failed: %s", source.name, day, exc)
        return result

    if not refs:
        return result

    semaphore = asyncio.Semaphore(concurrency)
    # `Archive.put` mutates the in-memory index and appends to the manifest, so
    # it is serialized. Storing inside the worker (rather than collecting every
    # payload and storing at the end) keeps peak memory at concurrency x
    # document size instead of the whole day.
    write_lock = asyncio.Lock()

    async def fetch_and_store(ref: DocumentRef) -> None:
        async with semaphore:
            document = await source.fetch(ref)
        async with write_lock:
            stored = archive.put(document, harvest_day=day)
        if stored is None:
            result.unchanged += 1
            return
        result.stored += 1
        if stored.is_revision:
            result.revisions += 1

    outcomes = await asyncio.gather(
        *(fetch_and_store(ref) for ref in refs), return_exceptions=True
    )
    for ref, outcome in zip(refs, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            result.failed += 1
            message = f"{ref.doc_id}: {outcome}"
            logger.warning("%s: fetch failed for %s", source.name, message)
            # Cap the stored detail: a systemic outage would otherwise produce
            # thousands of identical lines. The count stays exact.
            if len(result.errors) < 10:
                result.errors.append(message)
    return result


async def harvest(
    config: Config,
    *,
    days: Sequence[date],
    sources: Sequence[str] | None = None,
    limit: int | None = None,
    refetch_known: bool = False,
    on_result: Callable[[DayResult], None] | None = None,
) -> list[DayResult]:
    """Harvest every requested source across every requested day.

    Sources run concurrently within a day (they are different hosts, so their
    rate limits are independent), days run in sequence so that interrupting a
    long backfill leaves a clean boundary.
    """
    archive = Archive(config.archive_dir)
    results: list[DayResult] = []

    async with HttpClient(user_agent=config.user_agent) as http:
        source_objects = build_sources(config, http, sources)

        for day in days:
            day_results = await asyncio.gather(
                *(
                    harvest_day(
                        source,
                        archive,
                        day,
                        concurrency=config.concurrency,
                        limit=limit,
                        refetch_known=refetch_known,
                    )
                    for source in source_objects
                ),
                return_exceptions=True,
            )
            for source, outcome in zip(source_objects, day_results, strict=True):
                if isinstance(outcome, BaseException):
                    # harvest_day catches its own errors, so reaching here means
                    # something unanticipated. Keep going; record it honestly.
                    outcome_result = DayResult(source=source.name, day=day)
                    outcome_result.errors.append(f"unhandled: {outcome}")
                    logger.exception("%s: unhandled error on %s", source.name, day)
                else:
                    outcome_result = outcome
                results.append(outcome_result)
                if on_result:
                    on_result(outcome_result)

    return results


def recent_days(count: int, *, ending: date | None = None) -> list[date]:
    """The `count` days ending at `ending` (default yesterday), oldest first.

    Yesterday rather than today because most publishers post on a lag, and a
    same-day harvest reliably finds an empty or partial index.
    """
    last = ending or (date.today() - timedelta(days=1))
    return [last - timedelta(days=offset) for offset in range(count - 1, -1, -1)]
