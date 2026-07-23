"""Command line entry point.

    scout doctor                    check configuration and source availability
    scout harvest --days 3          collect primary filings into the archive
    scout status                    what the archive currently holds
    scout llm-check                 round-trip the whole harness on a synthetic task
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import sys
import uuid
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from scout import __version__
from scout.config import Config, load_config, load_config_for_harvest
from scout.data.archive import Archive
from scout.data.harvest import ALL_SOURCES, DayResult, build_sources, harvest, recent_days
from scout.data.http import HttpClient
from scout.fundamentals.concepts import Concept
from scout.fundamentals.ingest import ingest
from scout.fundamentals.store import FundamentalsStore
from scout.harness.build import build_client
from scout.harness.cost import CostLedger
from scout.harness.errors import HarnessError
from scout.harness.protocol import Message
from scout.harness.structured import complete_structured
from scout.metrics.base import MarketData
from scout.metrics.report import compute_metrics
from scout.output import write_reports
from scout.portfolio.evaluate import DEFAULT_COST_BPS, evaluate, scope_to_vintage
from scout.portfolio.ledger import Ledger, LedgerError
from scout.portfolio.models import Evaluation, Strategy, StrategyScore
from scout.portfolio.pick import PickBatch, run_pick
from scout.research.models import Verdict
from scout.research.pipeline import research_entities
from scout.screen.profile import enrich
from scout.screen.screen import run_screen


def _force_utf8_output() -> None:
    """Make console output encode-safe on non-UTF-8 terminals.

    A Windows console defaults to the locale codepage (cp1253 on a Greek system,
    cp1252 on a Western one), which cannot encode the box-drawing, bullet and
    check/cross glyphs the output uses -- and Python raises UnicodeEncodeError
    mid-render rather than degrading, which crashed `scout research` on exactly
    the memo it had just successfully produced. Forcing UTF-8 with `errors=
    "replace"` means a glyph the terminal genuinely cannot show is substituted,
    never fatal. Done before the Console is built so rich picks up the encoding.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # A redirected or already-detached stream may refuse; leave it as it is.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


app = typer.Typer(
    add_completion=False,
    help="Global deep-research equity scout: harvest primary filings, research them, measure the result.",
)
_force_utf8_output()
console = Console()


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)


def _parse_day(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise typer.BadParameter(f"expected YYYY-MM-DD, got {value!r}") from exc


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"scout {__version__}")


# --------------------------------------------------------------------- doctor


@app.command()
def doctor() -> None:
    """Check configuration and report which sources are usable."""
    try:
        config: Config = load_config_for_harvest()
    except ValueError as exc:
        _fail(str(exc))
        return

    console.print(f"[bold]user agent[/bold]   {config.user_agent}")
    console.print(f"[bold]archive[/bold]      {config.archive_dir.resolve()}")
    console.print(f"[bold]cache[/bold]        {config.cache_dir.resolve()}")

    table = Table("source", "available", "note", title="\nData sources", title_justify="left")
    notes = {
        "sec": "free, public domain, no redistribution limit",
        "esef": "free; EU regulated markets only -- MTFs (AIM, First North) are exempt",
        "edinet": "needs EDINET_API_KEY",
        "opendart": "needs OPENDART_API_KEY",
        "companies_house": "bulk accounts need no key; files purge after 60 days",
    }

    async def check() -> list[tuple[str, bool]]:
        async with HttpClient(user_agent=config.user_agent) as http:
            return [(s.name, s.available()) for s in build_sources(config, http)]

    for name, available in asyncio.run(check()):
        mark = "[green]yes[/green]" if available else "[yellow]no[/yellow]"
        table.add_row(name, mark, notes.get(name, ""))
    console.print(table)

    try:
        llm = load_config().llm
        console.print(f"\n[bold]llm[/bold]          {llm.provider} / {llm.model or '(unset)'}")
    except ValueError:
        console.print("\n[bold]llm[/bold]          not configured (fine -- harvest does not need it)")

    keyed = [name for name in ALL_SOURCES if name in {"edinet", "opendart"}]
    console.print(
        "\n[dim]Sources without credentials are skipped, not failed. "
        f"Free keys unlock: {', '.join(keyed)}.[/dim]"
    )


# -------------------------------------------------------------------- harvest


def _render_results(results: list[DayResult]) -> None:
    table = Table(
        "day", "source", "listed", "new", "rev", "known", "same", "failed", title="Harvest"
    )
    for result in results:
        note = f" [yellow]{result.skipped_reason}[/yellow]" if result.skipped_reason else ""
        table.add_row(
            result.day.isoformat(),
            result.source + note,
            str(result.listed),
            f"[green]{result.stored}[/green]" if result.stored else "0",
            str(result.revisions),
            str(result.skipped_known),
            str(result.unchanged),
            f"[red]{result.failed}[/red]" if result.failed else "0",
        )
    console.print(table)

    problems = [r for r in results if r.errors]
    if not problems:
        return
    console.print("\n[bold red]Errors[/bold red]")
    for result in problems:
        for error in result.errors:
            console.print(f"  {result.source} {result.day}: {error}")


@app.command(name="harvest")
def harvest_cmd(
    days: Annotated[int, typer.Option("--days", help="How many days back to harvest.")] = 1,
    from_: Annotated[str | None, typer.Option("--from", help="Start day, YYYY-MM-DD.")] = None,
    to: Annotated[str | None, typer.Option("--to", help="End day, YYYY-MM-DD.")] = None,
    source: Annotated[
        list[str] | None, typer.Option("--source", "-s", help="Restrict to these sources.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Max documents per source per day (for smoke tests).")
    ] = None,
    refetch: Annotated[
        bool, typer.Option("--refetch", help="Re-fetch documents already held, to catch revisions.")
    ] = False,
) -> None:
    """Collect primary filings into the append-only archive.

    Run this daily. Companies House purges its bulk files after 60 days and
    TDnet after about 30, so days missed here cannot be recovered later.
    """
    try:
        config = load_config_for_harvest()
    except ValueError as exc:
        _fail(str(exc))
        return

    if from_:
        start = _parse_day(from_)
        end = _parse_day(to) if to else start
        if end < start:
            _fail("--to is before --from.")
        span = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    else:
        span = recent_days(max(1, days))

    console.print(
        f"[bold]Harvesting[/bold] {span[0]} to {span[-1]} "
        f"({len(span)} day{'s' if len(span) > 1 else ''}) into {config.archive_dir}"
    )

    results = asyncio.run(
        harvest(
            config,
            days=span,
            sources=source or None,
            limit=limit,
            refetch_known=refetch,
        )
    )
    _render_results(results)

    stored = sum(r.stored for r in results)
    failed = sum(r.failed for r in results)
    console.print(f"\nStored [green]{stored}[/green] new document(s); {failed} fetch failure(s).")
    if failed:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------- status


@app.command()
def status() -> None:
    """Summarize what the archive holds."""
    try:
        config = load_config_for_harvest()
    except ValueError as exc:
        _fail(str(exc))
        return

    archive = Archive(config.archive_dir)
    by_source: Counter[str] = Counter()
    bytes_by_source: Counter[str] = Counter()
    days: set[str] = set()
    revisions = 0
    earliest: str | None = None
    latest: str | None = None

    for record in archive.iter_manifest():
        by_source[record.source] += 1
        bytes_by_source[record.source] += record.size_bytes
        days.add(record.harvest_day)
        revisions += 1 if record.version > 1 else 0
        earliest = record.harvest_day if earliest is None else min(earliest, record.harvest_day)
        latest = record.harvest_day if latest is None else max(latest, record.harvest_day)

    if not by_source:
        console.print(
            f"Archive at [bold]{config.archive_dir.resolve()}[/bold] is empty.\n"
            "Run [bold]scout harvest --days 1[/bold] to start it -- "
            "[yellow]the publishers purge on 30- and 60-day windows, so this cannot be backfilled.[/yellow]"
        )
        return

    table = Table("source", "documents", "size", title=f"Archive: {config.archive_dir.resolve()}")
    for name, count in sorted(by_source.items()):
        table.add_row(name, f"{count:,}", _human_bytes(bytes_by_source[name]))
    table.add_row("[bold]total[/bold]", f"[bold]{sum(by_source.values()):,}[/bold]",
                  f"[bold]{_human_bytes(sum(bytes_by_source.values()))}[/bold]")
    console.print(table)
    console.print(f"\n{len(days)} harvest day(s), {earliest} to {latest}. {revisions} revision(s).")


def _human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.1f} {unit}" if unit != "B" else f"{int(size):,} B"
        size /= 1024
    return f"{size:,.1f} GB"


# --------------------------------------------------------------------- ingest


@app.command(name="ingest")
def ingest_cmd(
    source: Annotated[
        list[str] | None, typer.Option("--source", "-s", help="Restrict to these sources.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Max documents to ingest (smoke test).")
    ] = None,
    reingest: Annotated[
        bool, typer.Option("--reingest", help="Re-parse documents already ingested.")
    ] = False,
) -> None:
    """Parse archived filings into normalized fundamentals in DuckDB.

    Reads the archive, routes each filing to its parser, maps raw XBRL tags to
    the canonical concept vocabulary, and stores both the raw facts (provenance)
    and the normalized snapshot (what the screen and metrics read). Idempotent.
    """
    try:
        config = load_config_for_harvest()
    except ValueError as exc:
        _fail(str(exc))
        return

    console.print(f"[bold]Ingesting[/bold] archive at {config.archive_dir} into {config.db_path}")
    result = asyncio.run(
        ingest(config, sources=source or None, limit=limit, reingest=reingest)
    )

    table = Table("metric", "count", title="Ingest")
    table.add_row("considered", str(result.considered))
    table.add_row("parsed", f"[green]{result.parsed}[/green]")
    table.add_row("snapshots written", f"[green]{result.snapshots}[/green]")
    table.add_row("no financial data", str(result.no_xbrl))
    table.add_row("skipped (already ingested)", str(result.skipped_existing))
    table.add_row("unsupported form/source", str(result.unsupported))
    table.add_row("normalization warnings", str(result.warnings_total))
    table.add_row("failed", f"[red]{result.failed}[/red]" if result.failed else "0")
    console.print(table)

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for error in result.errors:
            console.print(f"  {error}")
    if result.failed:
        raise typer.Exit(code=1)


@app.command()
def fundamentals(
    entity: Annotated[
        str | None, typer.Option("--entity", "-e", help="Entity id (CIK, LEI, ...) to show.")
    ] = None,
) -> None:
    """Show fundamentals coverage, or one entity's latest normalized snapshot."""
    try:
        config = load_config_for_harvest()
    except ValueError as exc:
        _fail(str(exc))
        return

    if not config.db_path.exists():
        _fail(f"No fundamentals database at {config.db_path}. Run `scout ingest` first.")
        return

    with FundamentalsStore(config.db_path, read_only=True) as store:
        if entity is None:
            table = Table("taxonomy", "entities", "snapshots", "concepts",
                          title=f"Fundamentals: {config.db_path}")
            for row in store.coverage():
                table.add_row(row["taxonomy"], str(row["entities"]),
                              str(row["snapshots"]), str(row["concepts"]))
            console.print(table)
            console.print(
                f"\n{store.entity_count()} entities, {store.snapshot_count()} snapshots. "
                "Pass --entity <id> to see one."
            )
            return

        snapshot = store.latest_snapshot(entity)
        if snapshot is None:
            _fail(f"No snapshot for entity {entity!r}.")
            return

        console.print(
            f"[bold]{snapshot.entity.name or snapshot.entity.entity_id}[/bold] "
            f"({snapshot.entity.entity_id}, {snapshot.taxonomy})  "
            f"{snapshot.period_end} {snapshot.fiscal_period or ''} {snapshot.currency or ''}"
        )
        table = Table("concept", "value", "span", "source tag", title="Latest snapshot")
        for concept in Concept:
            fact = snapshot.facts.get(concept)
            if fact is None:
                continue
            span = f"{(fact.period_end - fact.period_start).days}d" if fact.period_start else "instant"
            table.add_row(concept.value, f"{fact.value:,.0f}", span, fact.source_concept)
        console.print(table)
        if snapshot.warnings:
            console.print(f"\n[dim]{len(snapshot.warnings)} normalization warning(s):[/dim]")
            for warning in snapshot.warnings:
                console.print(f"  [dim]- {warning}[/dim]")


# ---------------------------------------------------------------------- enrich


@app.command(name="enrich")
def enrich_cmd(
    reenrich: Annotated[
        bool, typer.Option("--reenrich", help="Refetch profiles already stored.")
    ] = False,
    limit: Annotated[int | None, typer.Option("--limit", help="Max entities to enrich.")] = None,
) -> None:
    """Fetch entity profiles (SIC/sector, exchange, former names, filing history).

    Uses the free SEC submissions API for US filers; other sources get a minimal
    country-only profile until their registries are wired in. The screen uses
    these for sector cohorts and several hard excludes.
    """
    try:
        config = load_config_for_harvest()
    except ValueError as exc:
        _fail(str(exc))
        return

    if not config.db_path.exists():
        _fail(f"No fundamentals database at {config.db_path}. Run `scout ingest` first.")
        return

    result = asyncio.run(enrich(config, reenrich=reenrich, limit=limit))
    table = Table("metric", "count", title="Enrich")
    table.add_row("enriched (SEC)", f"[green]{result.enriched}[/green]")
    table.add_row("minimal (non-SEC)", str(result.minimal))
    table.add_row("skipped (already have)", str(result.skipped_existing))
    table.add_row("failed", f"[red]{result.failed}[/red]" if result.failed else "0")
    console.print(table)
    for error in result.errors[:10]:
        console.print(f"  [dim]{error}[/dim]")


# ---------------------------------------------------------------------- screen


@app.command(name="screen")
def screen_cmd(
    min_cohort: Annotated[
        int, typer.Option("--min-cohort", help="Min peers to z-score a cohort.")
    ] = 3,
    show_excluded: Annotated[
        bool, typer.Option("--show-excluded", help="List excluded names and why.")
    ] = False,
    keep_excluded_sectors: Annotated[
        bool, typer.Option("--keep-financials", help="Keep financials/utilities in the universe.")
    ] = False,
) -> None:
    """Rank the fundamentals universe into a candidate watchlist.

    Deterministic: hard excludes first, then a cheap x quality x safety composite
    ranked within (country x accounting-standard x sector) cohorts. Eyeball this
    before any research runs on top -- if the screen is junk, no LLM fixes it.
    Valuation multiples need prices (not wired in yet), so today names rank on
    quality and safety.
    """
    try:
        config = load_config_for_harvest()
    except ValueError as exc:
        _fail(str(exc))
        return

    if not config.db_path.exists():
        _fail(f"No fundamentals database at {config.db_path}. Run `scout ingest` first.")
        return

    result = run_screen(
        config, min_cohort=min_cohort, include_excluded_sectors=keep_excluded_sectors
    )

    console.print(
        f"[bold]Universe[/bold] {result.universe_size} entities  ·  "
        f"[green]{len(result.ranked)}[/green] ranked  ·  "
        f"[yellow]{len(result.excluded)}[/yellow] excluded"
    )

    if result.ranked:
        table = Table("#", "entity", "name", "cohort", "score", "cheap", "qual", "safe",
                      title="\nRanked candidates")
        for i, cand in enumerate(result.ranked, start=1):
            table.add_row(
                str(i),
                cand.entity_id,
                escape((cand.name or "")[:28]),
                escape(cand.cohort.label()[:32]),
                _score(cand.composite),
                _score(cand.cheap),
                _score(cand.quality),
                _score(cand.safety),
            )
        console.print(table)

    if show_excluded and result.excluded:
        console.print("\n[bold]Excluded[/bold]")
        for exc in result.excluded:
            console.print(f"  {exc.entity_id} {escape((exc.name or '')[:24])}: {escape('; '.join(exc.reasons))}")

    for note in result.notes:
        console.print(f"\n[dim]{escape(note)}[/dim]")


def _score(value: float | None) -> str:
    if value is None:
        return "—"
    color = "green" if value > 0 else "red" if value < 0 else ""
    text = f"{value:+.2f}"
    return f"[{color}]{text}[/{color}]" if color else text


# --------------------------------------------------------------------- metrics


def _fmt_metric(value: float | None, kind: str) -> str:
    if value is None:
        return "—"
    if value == float("inf"):
        return "∞"
    if kind == "pct":
        return f"{value * 100:+.1f}%"
    if kind == "ratio":
        return f"{value:.2f}"
    if kind == "score":
        # Piotroski is an integer 0-9; Altman/Beneish are continuous. Show a
        # whole number cleanly, a continuous score to 2 places.
        return f"{value:.0f}" if value == int(value) else f"{value:.2f}"
    if kind == "flag":
        return "yes" if value else "no"
    if kind == "count":
        return f"{value:.1f}"
    if kind == "currency":
        return f"{value:,.0f}"
    return f"{value:.4f}"


@app.command()
def metrics(
    entity: Annotated[str, typer.Option("--entity", "-e", help="Entity id (CIK, LEI, ...).")],
    price: Annotated[
        float | None, typer.Option("--price", help="Manual price for EV/market-cap metrics.")
    ] = None,
    shares: Annotated[
        float | None, typer.Option("--shares", help="Shares outstanding override.")
    ] = None,
) -> None:
    """Compute all deterministic metrics for one entity.

    Fundamentals-only metrics (GP/A, Altman Z, Piotroski, Beneish, dilution,
    cash runway) need no price. Pass --price to also get the valuation multiples
    (EV/EBIT, P/B, FCF yield); until there is a price feed this is the manual way
    in.
    """
    try:
        config = load_config_for_harvest()
    except ValueError as exc:
        _fail(str(exc))
        return

    if not config.db_path.exists():
        _fail(f"No fundamentals database at {config.db_path}. Run `scout ingest` first.")
        return

    with FundamentalsStore(config.db_path, read_only=True) as store:
        snapshots = store.snapshots_for_entity(entity)

    if not snapshots:
        _fail(f"No snapshots for entity {entity!r}. Run `scout ingest` first.")
        return

    market = MarketData(price=price, shares_outstanding=shares) if price is not None else None
    report = compute_metrics(snapshots, market=market)
    if report is None:
        _fail("Could not compute metrics.")
        return

    header = f"[bold]{entity}[/bold]  {report.period_end} {report.fiscal_period or ''} {report.currency or ''}"
    if not report.has_market_data:
        header += "  [dim](no price — valuation multiples skipped; pass --price)[/dim]"
    if not report.has_annual_pair:
        header += "  [dim](one period only — Piotroski/Beneish/dilution need two annual filings)[/dim]"
    console.print(header)

    # A dim style column for the whole "note" cell, so metric warnings/reasons
    # (which contain raw XBRL tag text and colons) never need to be markup-safe.
    table = Table("metric", "value", "basis", "note", title="Metrics")
    table.columns[3].style = "dim"
    table.columns[3].no_wrap = False
    for name, m in report.metrics.items():
        note = m.reason or ("; ".join(m.warnings[:1]) if m.warnings else "")
        table.add_row(
            name,
            _fmt_metric(m.value, m.kind),
            m.basis if m.ok else "",
            escape(note),
        )
    console.print(table)


# -------------------------------------------------------------------- research


@app.command()
def research(
    entity: Annotated[
        str | None, typer.Option("--entity", "-e", help="Research one entity id.")
    ] = None,
    top: Annotated[
        int, typer.Option("--top", help="Research the top N screened candidates.")
    ] = 5,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass the replay cache.")] = False,
    out: Annotated[
        str | None, typer.Option("--out", help="Directory for saved reports (default: data/reports).")
    ] = None,
    no_save: Annotated[
        bool, typer.Option("--no-save", help="Print only; do not write report files.")
    ] = False,
) -> None:
    """Run the cited LLM research pipeline over screened candidates.

    For each candidate: build an evidence pack from its filings, extract findings
    each anchored to a verbatim quote, verify every citation, run a bull/bear/
    skeptic debate, and write a memo. The LLM can VETO but never promote a name,
    and the verdict is decided in code. Needs a model configured in .env.

    Each memo is saved as Markdown (to read) and JSON (to diff/automate) under
    data/reports/<date>/, unless --no-save is passed.
    """
    try:
        config = load_config()
    except ValueError as exc:
        _fail(str(exc))
        return

    if not config.db_path.exists():
        _fail(f"No fundamentals database at {config.db_path}. Run `scout ingest` first.")
        return

    if entity:
        entity_ids = [entity]
    else:
        screen_result = run_screen(config)
        entity_ids = [c.entity_id for c in screen_result.ranked[:top]]
        if not entity_ids:
            _fail("The screen produced no ranked candidates to research.")
            return

    client = build_client(config, use_cache=not no_cache, prompt_version="research/1")
    run_id = f"{date.today().isoformat()}-{uuid.uuid4().hex[:8]}"
    console.print(
        f"[bold]Researching[/bold] {len(entity_ids)} candidate(s) with {client.model} ...\n"
    )

    try:
        # Thread the configured reasoning effort through: a hybrid-thinking local
        # model (Qwen3.x, DeepSeek-R1) defaults to thinking ON, which for the
        # extraction/debate calls burns the whole output budget on reasoning and
        # returns an empty answer. REASONING_EFFORT=none is what turns that off.
        reports = asyncio.run(
            research_entities(config, entity_ids, client=client, effort=config.llm.effort)  # type: ignore[arg-type]
        )
    except HarnessError as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        return

    if not reports:
        _fail("No candidate could be researched (missing filings or metrics).")
        return

    for report in reports:
        _render_memo(report)

    if not no_save:
        out_dir = Path(out) if out else config.reports_dir
        written = write_reports(reports, out_dir, run_id=run_id, model=client.model)
        memo_count = sum(1 for p in written if p.suffix == ".md")
        console.print(
            f"[dim]Saved {memo_count} memo(s) to {out_dir}"
            f"{' (with JSON sidecars)' if memo_count else ''}.[/dim]"
        )

    vetoed = sum(1 for r in reports if r.vetoed)
    console.print(
        f"\n[bold]Summary[/bold]: {len(reports)} researched, "
        f"[red]{vetoed} vetoed[/red], {len(reports) - vetoed} survived."
    )


def _render_memo(report) -> None:  # type: ignore[no-untyped-def]
    memo = report.memo
    verdict_style = "red" if memo.verdict == Verdict.VETO else "green"
    verdict_text = "VETO" if memo.verdict == Verdict.VETO else "no veto"
    console.print(
        f"[bold]{escape(report.name or report.entity_id)}[/bold] "
        f"({report.entity_id})  [{verdict_style}]{verdict_text}[/{verdict_style}]"
    )
    console.print(f"  {escape(memo.headline)}")
    console.print(f"  [dim]{escape(memo.thesis)}[/dim]")

    if memo.veto_reasons:
        for reason in memo.veto_reasons:
            console.print(f"  [red]✗ {escape(reason)}[/red]")

    if report.verified_findings:
        console.print(f"  [bold]findings[/bold] ({len(report.verified_findings)} cited):")
        for f in report.verified_findings[:6]:
            console.print(f"    [{_sev_color(f.severity.value)}]{f.severity.value}[/] "
                          f"{f.category.value}: {escape(f.claim)}")
    if report.dropped_citations:
        console.print(
            f"  [dim yellow]{len(report.dropped_citations)} finding(s) dropped — "
            f"citation not found in the filing[/dim yellow]"
        )
    for warning in report.warnings:
        console.print(f"  [dim]! {escape(warning)}[/dim]")
    console.print()


def _sev_color(severity: str) -> str:
    return {"critical": "red", "high": "yellow", "medium": "cyan", "low": "dim"}.get(severity, "")


# ----------------------------------------------------------------- pick / score


def _load_prices(inline: list[str] | None, prices_file: str | None) -> dict[str, float]:
    """Assemble a manual price map from a JSON file and/or inline ID=VALUE pairs.

    There is no price feed yet (PLAN.md 1.3 deferral), so prices are supplied by
    hand. Inline pairs override the file, so a quick correction does not mean
    editing the file. A malformed entry fails loud rather than being dropped --
    a silently ignored price becomes a silently ungradeable pick, and a non-finite
    or non-positive one would push inf/nan straight into the score. Keys are
    stripped consistently (file and inline) so " AAPL " and "AAPL" match.
    """
    prices: dict[str, float] = {}
    if prices_file:
        path = Path(prices_file)
        if not path.exists():
            _fail(f"prices file not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _fail(f"could not read prices file {path}: {exc}")
        if not isinstance(raw, dict):
            _fail(f"prices file {path} must be a JSON object of entity_id -> price.")
        for entity_id, value in raw.items():  # type: ignore[union-attr]
            prices[str(entity_id).strip()] = _price_value(value, f"{entity_id!r} in {path}")

    for pair in inline or []:
        if "=" not in pair:
            _fail(f"--price expects ENTITY=VALUE, got {pair!r}.")
        entity_id, _, value = pair.partition("=")
        prices[entity_id.strip()] = _price_value(value, f"--price {entity_id.strip()!r}")
    return prices


def _price_value(value: object, where: str) -> float:
    """Coerce and validate one price: a finite, positive number. `bool` is
    rejected outright -- JSON `true` would otherwise coerce to 1.0 and read as a
    real quote."""
    if isinstance(value, bool):
        _fail(f"price for {where} must be a number, not a boolean: {value!r}")
    try:
        price = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        _fail(f"price for {where} is not a number: {value!r}")
    if not math.isfinite(price) or price <= 0:
        _fail(f"price for {where} must be finite and positive, got {price!r}")
    return price


@app.command(name="pick")
def pick_cmd(
    price: Annotated[
        list[str] | None,
        typer.Option("--price", help="Manual reference price, ENTITY=VALUE (repeatable)."),
    ] = None,
    prices_file: Annotated[
        str | None, typer.Option("--prices", help="JSON file of entity_id -> reference price.")
    ] = None,
    as_of: Annotated[
        str | None, typer.Option("--as-of", help="Pick date, YYYY-MM-DD (default: today).")
    ] = None,
    top: Annotated[int, typer.Option("--top", help="Names in the screen/agent books.")] = 20,
    with_research: Annotated[
        bool, typer.Option("--research", help="Also run the LLM pipeline and record the agent book.")
    ] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass the replay cache.")] = False,
) -> None:
    """Pre-register today's paper picks for each strategy into the ledger.

    Forward paper trading is the only credible evidence this project can produce
    (PLAN.md 1.3), and it only counts if the pick is recorded BEFORE the outcome
    is known -- so run this the day the screen works and every rebalance after.
    Deterministic by default; pass --research to also record the agent's book,
    the one strategy the baselines exist to judge.
    """
    # --research needs a model; a plain pick does not (like harvest).
    try:
        config = load_config() if with_research else load_config_for_harvest()
    except ValueError as exc:
        _fail(str(exc))
        return

    if not config.db_path.exists():
        _fail(f"No fundamentals database at {config.db_path}. Run `scout ingest` first.")
        return

    pick_day = _parse_day(as_of) if as_of else date.today()
    prices = _load_prices(price, prices_file)

    research_client = None
    if with_research:
        research_client = build_client(config, use_cache=not no_cache, prompt_version="research/1")

    try:
        batch = run_pick(
            config,
            as_of=pick_day,
            prices=prices,
            top=top,
            research_client=research_client,
            # Same reason as `scout research`: without the configured effort a
            # hybrid-thinking local model thinks by default and returns no answer.
            effort=config.llm.effort,  # type: ignore[arg-type]
        )
    except (HarnessError, LedgerError) as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        return

    _render_pick(batch, config.ledger_path)


def _render_pick(batch: PickBatch, ledger_path: Path) -> None:
    console.print(
        f"[bold]Picked[/bold] {batch.as_of}  ·  universe {batch.universe_size}  ·  "
        f"run [dim]{batch.run_id}[/dim]"
    )
    table = Table("strategy", "picks", "priced", "note", title="\nPre-registered books")
    table.columns[3].style = "dim"
    for sb in batch.strategies:
        priced = f"[green]{sb.n_priced}[/green]" if sb.n_priced else "0"
        table.add_row(sb.strategy.value, str(sb.n_picks), priced, escape(sb.note or ""))
    console.print(table)
    console.print(
        f"\nWrote [green]{batch.total_written}[/green] pick(s) to {ledger_path}."
    )
    for note in batch.notes:
        console.print(f"[dim]{escape(note)}[/dim]")


@app.command(name="score")
def score_cmd(
    price: Annotated[
        list[str] | None,
        typer.Option("--price", help="Forward (exit) price, ENTITY=VALUE (repeatable)."),
    ] = None,
    prices_file: Annotated[
        str | None, typer.Option("--prices", help="JSON file of entity_id -> forward price.")
    ] = None,
    as_of: Annotated[
        str | None, typer.Option("--as-of", help="Exit date, YYYY-MM-DD (default: today).")
    ] = None,
    vintage: Annotated[
        str | None,
        typer.Option("--vintage", help="Grade only picks with this pick date, YYYY-MM-DD."),
    ] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Grade only picks from this pick run.")
    ] = None,
    cost_bps: Annotated[
        float, typer.Option("--cost-bps", help="Round-trip cost in basis points.")
    ] = DEFAULT_COST_BPS,
) -> None:
    """Grade the pre-registered picks against forward prices, honestly.

    Reports each strategy's full return distribution (not a lone hit rate), the
    agent measured against the three dumb baselines, and -- above all -- a loud
    warning when the sample is too small to mean anything. If the agent does not
    beat the EV/EBIT decile, this says so: the LLM layer is then cost, not signal.
    """
    try:
        config = load_config_for_harvest()
    except ValueError as exc:
        _fail(str(exc))
        return

    ledger = Ledger(config.ledger_path)
    if not ledger.exists():
        _fail(
            f"No ledger at {config.ledger_path}. Run `scout pick` first to pre-register "
            "picks; scoring needs a book that was recorded before the outcome."
        )
        return

    if cost_bps < 0:
        _fail(f"--cost-bps must be >= 0 (a cost cannot pay you), got {cost_bps}.")
        return

    try:
        picks = ledger.read()
    except LedgerError as exc:
        _fail(str(exc))
        return
    if not picks:
        _fail("The ledger is empty. Run `scout pick` to pre-register a book first.")
        return

    # Scope to one comparable book: grading picks from several dates against a
    # single forward-price snapshot would mix holding periods and mislead.
    scoped, scope_note = scope_to_vintage(
        picks,
        vintage=_parse_day(vintage) if vintage else None,
        run_id=run_id,
    )
    if not scoped:
        _fail(scope_note or "no picks matched the requested vintage/run.")
        return

    exit_day = _parse_day(as_of) if as_of else date.today()
    forward_prices = _load_prices(price, prices_file)
    if not forward_prices:
        _fail(
            "No forward prices supplied. Pass --prices <file> or --price ENTITY=VALUE so "
            "the picks can be graded against what price actually did."
        )
        return

    result = evaluate(scoped, forward_prices, as_of_exit=exit_day, cost_bps=cost_bps)
    _render_score(result, scope_note=scope_note)


def _render_score(result: Evaluation, *, scope_note: str | None = None) -> None:
    console.print(
        f"[bold]Scored[/bold] as of {result.as_of_exit}  ·  {result.total_scored} graded pick(s)  "
        f"·  [dim]{result.cost_bps:.0f} bps round-trip cost[/dim]"
    )
    if scope_note:
        console.print(f"[yellow]{escape(scope_note)}[/yellow]")

    table = Table(
        "strategy", "book", "graded", "return", "median", "p10", "p90", "hit",
        title="\nForward returns by strategy",
    )
    for strategy in Strategy:
        score = result.scores.get(strategy)
        if score is None:
            continue
        _add_score_row(table, score)
    console.print(table)

    # The agent-vs-baseline verdicts only mean something once an agent book was
    # recorded (scout pick --research). Without one, the baselines still stand on
    # their own above -- so skip an all-"no agent book" block rather than print it.
    if Strategy.AGENT in result.scores and result.comparisons:
        console.print("\n[bold]Agent vs baselines[/bold]  [dim](mean · median)[/dim]")
        for comp in result.comparisons:
            console.print(
                f"  vs {comp.baseline.value:<16} {_delta_cell(comp.delta)} · "
                f"{_delta_cell(comp.median_delta)}  [dim]{escape(comp.verdict)}[/dim]"
            )

    for score in result.scores.values():
        for note in score.notes:
            console.print(f"[dim]{score.strategy.value}: {escape(note)}[/dim]")

    # The most important thing this command prints, so it goes last and loud.
    for warning in result.warnings:
        console.print(f"\n[bold yellow]{escape(warning)}[/bold yellow]")


def _delta_cell(value: float | None) -> str:
    if value is None:
        return "—"
    style = "green" if value > 0 else "red" if value < 0 else ""
    text = f"{value:+.1%}"
    return f"[{style}]{text}[/{style}]" if style else text


def _add_score_row(table: Table, score: StrategyScore) -> None:
    dist = score.distribution
    if dist is None:
        table.add_row(score.strategy.value, str(score.n_picks), "0", "—", "—", "—", "—", "—")
        return
    table.add_row(
        score.strategy.value,
        str(score.n_picks),
        str(score.n_scored),
        _pct(score.portfolio_return),
        _pct(dist.median),
        _pct(dist.p10),
        _pct(dist.p90),
        f"{dist.hit_rate:.0%}",
    )


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    color = "green" if value > 0 else "red" if value < 0 else ""
    text = f"{value:+.1%}"
    return f"[{color}]{text}[/{color}]" if color else text


# ------------------------------------------------------------------ llm-check


class _Verdict(BaseModel):
    """Deliberately exercises the parts of the ladder that break.

    `confidence` carries a range constraint that Anthropic's JSON Schema subset
    drops on the wire -- so this model also proves the Pydantic validator still
    catches it locally. See `structured.schema_for`.
    """

    headline: str = Field(description="One sentence summarizing the filing.")
    red_flags: list[str] = Field(default_factory=list, description="Concerns, if any.")
    confidence: float = Field(ge=0.0, le=1.0, description="0 to 1.")


_SAMPLE = """\
Registrant: a shell corporation with no revenue. During the period the company
issued 480,000,000 new common shares (up from 42,000,000), completed a 1-for-50
reverse split, changed its name and ticker, and entered a convertible note whose
conversion price is 40% of the lowest closing price over the preceding 20
trading days. The auditor was dismissed and replaced. Cash on hand is $61,000
against a quarterly operating burn of $390,000."""


@app.command()
def llm_check(
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass the replay cache.")] = False,
) -> None:
    """Round-trip the whole harness on a synthetic filing. No data sources needed.

    Exercises provider wiring, the structured-output ladder, reasoning
    normalization, schema validation and cost accounting in one call -- so when
    something breaks later, this tells you whether the harness or the pipeline
    is at fault.
    """
    try:
        config = load_config()
    except ValueError as exc:
        _fail(str(exc))
        return

    client = build_client(config, use_cache=not no_cache, prompt_version="llm-check/1")
    console.print(f"[bold]{client.name}[/bold] / {client.model}")
    console.print(f"capabilities: {client.capabilities}")

    ledger = CostLedger()
    messages = [
        Message(
            role="system",
            content=(
                "You are a forensic analyst reading a filing excerpt. Report only what the "
                "text supports. Do not compute or invent figures that are not present."
            ),
        ),
        Message(role="user", content=_SAMPLE),
    ]

    async def run() -> None:
        with ledger.stage("llm-check"):
            result = await complete_structured(
                client,
                messages,
                _Verdict,
                effort=config.llm.effort,  # type: ignore[arg-type]
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                cache_prefix_upto=0,
            )
            ledger.record(result.response)

        console.print(
            f"\n[green]OK[/green]  mode={result.mode_used.value}  "
            f"attempts={result.attempts}  repairs={result.repairs}"
        )
        if result.response.reasoning:
            console.print(f"[dim]reasoning: {len(result.response.reasoning)} chars (separated)[/dim]")
        console.print(f"\n[bold]headline[/bold]   {result.value.headline}")
        console.print(f"[bold]confidence[/bold] {result.value.confidence}")
        for flag in result.value.red_flags:
            console.print(f"  - {flag}")
        console.print(f"\n{ledger.render()}")

    try:
        asyncio.run(run())
    except HarnessError as exc:
        _fail(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    app()
