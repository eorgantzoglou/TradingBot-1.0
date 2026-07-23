"""The agent's toolbox: existing code, exposed as callable tools.

This is where the fusion happens. Each tool is a thin async wrapper over a piece
of the deterministic stack -- the metrics layer, the screen, the research
pipeline, the filing archive -- so the LLM can *drive* while every number it gets
is still code-computed and every disciplined analysis still runs the
citation-verify + code-veto path. The agent chooses what to look at; the tools
decide what is true.

Tools come in three families:

  deterministic financial   get_fundamentals, compute_metrics, screen,
                            check_excludes -- numbers in code (design rule 1).
  disciplined analysis      deep_analyze -- runs the full research pipeline
                            (evidence -> extract -> verify -> debate -> VETO) and
                            returns its cited, code-decided verdict.
  discovery + evidence      search_filings, read_filing (SEC), web_search,
                            fetch_url -- how the agent finds things and gathers
                            quotable text, which the brief's claims verify against.

Every tool returns a `ToolResult`: a short prose observation for the model, plus
(for the evidence tools) the FULL source text stashed for later verification.
Nothing here judges; judgement is the model's and the guardrails' job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from scout.agent.models import Tool, ToolResult
from scout.config import Config
from scout.data.http import HttpClient
from scout.data.sources import sec_fulltext, web
from scout.fundamentals.store import FundamentalsStore
from scout.harness.protocol import Effort, LLMClient
from scout.metrics.base import MarketData
from scout.metrics.report import MetricReport, compute_metrics
from scout.research.evidence import _format_metrics, _html_to_text
from scout.research.pipeline import research_entities
from scout.screen.excludes import ScreenInput, evaluate_excludes
from scout.screen.models import Decision
from scout.screen.screen import run_screen

logger = logging.getLogger(__name__)

_SEC_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"
_FILING_EXCERPT_CHARS = 2500


@dataclass(slots=True)
class ToolContext:
    """Shared resources the tools close over, opened for the run's duration."""

    config: Config
    http: HttpClient
    client: LLMClient
    store: FundamentalsStore | None
    web_provider: web.WebSearchProvider | None
    effort: Effort | None = None


def build_tools(ctx: ToolContext) -> list[Tool]:
    """Assemble the tool list for one run. Which tools are present depends on
    what is configured -- no db means no metric tools, no web provider means no
    web search -- so the agent is never offered a capability that would only
    error."""
    tools: list[Tool] = [
        Tool("search_filings", "Search SEC EDGAR full-text for filings by keyword.",
             "query (string), optional forms (e.g. '10-K,8-K')", _search_filings(ctx)),
        Tool("read_filing", "Fetch and read one SEC filing's text (stores it for citing).",
             "cik (string), accession (string), doc (primary document filename)", _read_filing(ctx)),
    ]

    if ctx.store is not None:
        tools += [
            Tool("get_fundamentals", "Show one entity's latest normalized financial facts.",
                 "entity_id (CIK or LEI)", _get_fundamentals(ctx)),
            Tool("compute_metrics", "Compute all deterministic metrics for one entity.",
                 "entity_id, optional price (number) for valuation multiples", _compute_metrics(ctx)),
            Tool("check_excludes", "Run the hard red-flag excludes on one entity.",
                 "entity_id", _check_excludes(ctx)),
            Tool("screen", "Rank the ingested universe into a candidate watchlist (no LLM).",
                 "optional top (int, default 10)", _screen(ctx)),
            Tool("deep_analyze",
                 "Run the full disciplined research pipeline on one entity: cited findings, "
                 "bull/bear/skeptic debate, and a CODE-DECIDED veto you cannot override.",
                 "entity_id", _deep_analyze(ctx)),
        ]

    if ctx.web_provider is not None:
        tools.append(
            Tool("web_search",
                 "Search the web for news/articles. NOTE: microcap news is often paid "
                 "promotion -- treat enthusiasm as a red flag, not a reason to buy.",
                 "query (string), optional limit (int)", _web_search(ctx))
        )
    tools.append(
        Tool("fetch_url", "Fetch a web page and read its text (stores it for citing).",
             "url (string)", _fetch_url(ctx))
    )
    return tools


# --------------------------------------------------------------------------- #
# Discovery + evidence
# --------------------------------------------------------------------------- #


def _search_filings(ctx: ToolContext):  # type: ignore[no-untyped-def]
    async def run(inp: dict) -> ToolResult:
        query = str(inp.get("query", "")).strip()
        if not query:
            return ToolResult.error("search_filings needs a non-empty 'query'.")
        forms = _split_forms(inp.get("forms"))
        hits = await sec_fulltext.search_filings(ctx.http, query, forms=forms, limit=10)
        if not hits:
            return ToolResult(output=f"No filings found for {query!r}.")
        lines = [f"{len(hits)} filing(s) for {query!r}:"]
        for h in hits:
            lines.append(
                f"- {h.company} | {h.form} filed {h.filed} | cik={h.cik} "
                f"accession={h.accession} doc={h.doc}"
            )
        lines.append("Use read_filing with a cik/accession/doc to read one.")
        return ToolResult(output="\n".join(lines))

    return run


def _read_filing(ctx: ToolContext):  # type: ignore[no-untyped-def]
    async def run(inp: dict) -> ToolResult:
        cik = str(inp.get("cik", "")).strip()
        accession = str(inp.get("accession", "")).strip()
        doc = str(inp.get("doc", "")).strip()
        if not (cik and accession and doc):
            return ToolResult.error("read_filing needs 'cik', 'accession' and 'doc'.")
        acc_nodash = accession.replace("-", "")
        url = f"{_SEC_ARCHIVE}/{int(cik)}/{acc_nodash}/{doc}"
        try:
            raw = await ctx.http.get_bytes(url)
        except Exception as exc:  # surfaced to the agent as an observation
            return ToolResult.error(f"could not fetch {url}: {exc}")
        text = _html_to_text(raw.decode("latin-1", errors="replace"))
        if len(text) < 100:
            return ToolResult.error(f"no readable text at {url} (not an HTML filing document?).")
        excerpt = text[:_FILING_EXCERPT_CHARS]
        note = "" if len(text) <= _FILING_EXCERPT_CHARS else " [excerpt; full text stored for citing]"
        return ToolResult(
            output=f"Filing {accession} ({len(text):,} chars){note}:\n{excerpt}",
            filing_texts={accession: text},
        )

    return run


def _web_search(ctx: ToolContext):  # type: ignore[no-untyped-def]
    async def run(inp: dict) -> ToolResult:
        assert ctx.web_provider is not None
        query = str(inp.get("query", "")).strip()
        if not query:
            return ToolResult.error("web_search needs a non-empty 'query'.")
        limit = _as_int(inp.get("limit"), default=5)
        results = await ctx.web_provider.search(ctx.http, query, limit=limit)
        if not results:
            return ToolResult(output=f"No web results for {query!r}.")
        pages: dict[str, str] = {}
        lines = [f"{len(results)} web result(s) for {query!r}:"]
        for r in results:
            lines.append(f"- {r.title} <{r.url}>\n  {r.snippet[:300]}")
            if r.content:
                pages[r.url] = r.content  # stored so a quote from it can be verified
        lines.append("Use fetch_url to read a full page before quoting it.")
        return ToolResult(output="\n".join(lines), web_pages=pages)

    return run


def _fetch_url(ctx: ToolContext):  # type: ignore[no-untyped-def]
    async def run(inp: dict) -> ToolResult:
        url = str(inp.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            return ToolResult.error("fetch_url needs an absolute http(s) 'url'.")
        text = await web.fetch_url(ctx.http, url)
        if not text:
            return ToolResult.error(f"no readable text at {url}.")
        return ToolResult(
            output=f"{url} ({len(text):,} chars):\n{text[:_FILING_EXCERPT_CHARS]}",
            web_pages={url: text},
        )

    return run


# --------------------------------------------------------------------------- #
# Deterministic financial tools (numbers stay in code)
# --------------------------------------------------------------------------- #


def _get_fundamentals(ctx: ToolContext):  # type: ignore[no-untyped-def]
    async def run(inp: dict) -> ToolResult:
        entity_id = _entity(inp)
        if not entity_id:
            return ToolResult.error("get_fundamentals needs an 'entity_id'.")
        assert ctx.store is not None
        snap = ctx.store.latest_snapshot(entity_id)
        if snap is None:
            return ToolResult.error(_not_ingested(entity_id))
        lines = [
            f"{snap.entity.name or entity_id} ({entity_id}, {snap.taxonomy}) "
            f"{snap.period_end} {snap.fiscal_period or ''} {snap.currency or ''}"
        ]
        for concept, fact in snap.facts.items():
            lines.append(f"  {concept.value} = {fact.value:,.0f}")
        return ToolResult(output="\n".join(lines))

    return run


def _compute_metrics(ctx: ToolContext):  # type: ignore[no-untyped-def]
    async def run(inp: dict) -> ToolResult:
        entity_id = _entity(inp)
        if not entity_id:
            return ToolResult.error("compute_metrics needs an 'entity_id'.")
        assert ctx.store is not None
        snapshots = ctx.store.snapshots_for_entity(entity_id)
        if not snapshots:
            return ToolResult.error(_not_ingested(entity_id))
        price = _as_float(inp.get("price"))
        market = MarketData(price=price) if price is not None else None
        report = compute_metrics(snapshots, market=market)
        if report is None:
            return ToolResult.error(f"could not compute metrics for {entity_id}.")
        block = _format_metrics(report)
        return ToolResult(output=block, metrics_block=block)

    return run


def _check_excludes(ctx: ToolContext):  # type: ignore[no-untyped-def]
    async def run(inp: dict) -> ToolResult:
        entity_id = _entity(inp)
        if not entity_id:
            return ToolResult.error("check_excludes needs an 'entity_id'.")
        assert ctx.store is not None
        snapshots = ctx.store.snapshots_for_entity(entity_id)
        if not snapshots:
            return ToolResult.error(_not_ingested(entity_id))
        report = _safe_report(snapshots)
        checks = evaluate_excludes(
            ScreenInput(entity_id=entity_id, profile=None, snapshots=snapshots, report=report)
        )
        lines = [f"Hard excludes for {entity_id}:"]
        for c in checks:
            mark = {Decision.EXCLUDE: "EXCLUDE", Decision.PASS: "pass",
                    Decision.INSUFFICIENT: "insufficient-data"}[c.decision]
            lines.append(f"  [{mark}] {c.rule}: {c.reason}")
        return ToolResult(output="\n".join(lines))

    return run


def _screen(ctx: ToolContext):  # type: ignore[no-untyped-def]
    async def run(inp: dict) -> ToolResult:
        top = _as_int(inp.get("top"), default=10)
        result = run_screen(ctx.config)
        if not result.ranked:
            return ToolResult(output="The screen produced no ranked candidates.")
        lines = [f"Top {min(top, len(result.ranked))} of {len(result.ranked)} ranked candidates:"]
        for i, c in enumerate(result.ranked[:top], start=1):
            score = f"{c.composite:+.2f}" if c.composite is not None else "unranked"
            lines.append(f"  {i}. {c.name or c.entity_id} ({c.entity_id}) score={score} [{c.cohort.label()}]")
        return ToolResult(output="\n".join(lines))

    return run


def _deep_analyze(ctx: ToolContext):  # type: ignore[no-untyped-def]
    async def run(inp: dict) -> ToolResult:
        entity_id = _entity(inp)
        if not entity_id:
            return ToolResult.error("deep_analyze needs an 'entity_id'.")
        reports = await research_entities(
            ctx.config, [entity_id], client=ctx.client, effort=ctx.effort
        )
        if not reports:
            return ToolResult.error(
                f"{_not_ingested(entity_id)} deep_analyze needs the entity's filings ingested."
            )
        report = reports[0]
        lines = [
            f"deep_analyze({entity_id}) verdict: {report.memo.verdict.value.upper()}",
            f"  {report.memo.headline}",
        ]
        for reason in report.memo.veto_reasons:
            lines.append(f"  VETO: {reason}")
        for f in report.verified_findings:
            lines.append(f"  [{f.severity.value}] {f.category.value}: {f.claim}")
        if report.dropped_citations:
            lines.append(f"  ({len(report.dropped_citations)} finding(s) dropped -- unverifiable)")
        return ToolResult(output="\n".join(lines))

    return run


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _entity(inp: dict) -> str:
    return str(inp.get("entity_id") or inp.get("entity") or inp.get("cik") or "").strip()


def _not_ingested(entity_id: str) -> str:
    return (
        f"entity {entity_id} has no ingested fundamentals. Only entities harvested and "
        "ingested (`scout harvest`/`scout ingest`) have computable numbers; you can still "
        "read its filings with read_filing."
    )


def _safe_report(snapshots) -> MetricReport | None:  # type: ignore[no-untyped-def]
    try:
        return compute_metrics(snapshots)
    except Exception as exc:  # excludes still run on a None report
        logger.warning("metrics failed in check_excludes: %s", exc)
        return None


def _split_forms(value: object) -> list[str] | None:
    if not value:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [f.strip() for f in str(value).split(",") if f.strip()]


def _as_int(value: object, *, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
