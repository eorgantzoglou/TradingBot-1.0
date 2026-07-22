"""Command line entry point.

    scout doctor                    check configuration and source availability
    scout harvest --days 3          collect primary filings into the archive
    scout status                    what the archive currently holds
    scout llm-check                 round-trip the whole harness on a synthetic task
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Annotated

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from scout import __version__
from scout.config import Config, load_config, load_config_for_harvest
from scout.data.archive import Archive
from scout.data.harvest import ALL_SOURCES, DayResult, build_sources, harvest, recent_days
from scout.data.http import HttpClient
from scout.harness.build import build_client
from scout.harness.cost import CostLedger
from scout.harness.errors import HarnessError
from scout.harness.protocol import Message
from scout.harness.structured import complete_structured

app = typer.Typer(
    add_completion=False,
    help="Global deep-research equity scout: harvest primary filings, research them, measure the result.",
)
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
