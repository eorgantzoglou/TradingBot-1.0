"""Compose the final brief, then enforce the guardrails in code.

The loop lets the model roam; this is where the leash pulls tight. Three steps,
in order:

  1. COMPOSE. One structured call turns everything the agent gathered into an
     `AgentBrief` -- headline, thesis, recommendation, findings.
  2. VERIFY. Every finding's quote is checked against the exact text the tools
     fetched (filings by accession, web pages by URL), using the same matcher as
     the research pipeline (`verify.quote_in_text`). A finding whose quote is not
     really there is DROPPED -- the model cannot cite what it did not fetch.
  3. VETO IN CODE. If the subject is an ingested entity, the disciplined pipeline
     runs and its `decide_verdict` owns the verdict. The agent's recommendation is
     advisory prose; a code VETO overrides it and cannot be argued away.

So the agent gains the discovery and web reach the old pipeline lacked, without
gaining the authority to invent a number, cite a phantom source, or wave a fraud
through.
"""

from __future__ import annotations

import logging

from scout.agent.models import AgentBrief, AgentRun, BriefFinding, FinishedBrief
from scout.agent.prompts import build_compose_messages
from scout.config import Config
from scout.harness.cost import CostLedger
from scout.harness.protocol import Effort, LLMClient
from scout.harness.structured import complete_structured
from scout.research.models import Verdict
from scout.research.pipeline import research_entities
from scout.research.verify import quote_in_text

logger = logging.getLogger(__name__)


async def compose_brief(
    client: LLMClient,
    run: AgentRun,
    *,
    subject: str,
    entity_id: str | None,
    config: Config,
    model: str,
    effort: Effort | None = None,
    ledger: CostLedger | None = None,
) -> FinishedBrief:
    """Turn a finished agent run into the guardrail-checked brief."""
    messages = build_compose_messages(
        run.goal, _digest(run), _evidence_catalog(run), run.metrics_block
    )
    result = await complete_structured(
        client, messages, AgentBrief, effort=effort, temperature=0.2, max_tokens=1500
    )
    if ledger is not None:
        ledger.record(result.response)
    brief = result.value

    verified, dropped = _verify_findings(brief.findings, run)
    verdict, veto_reasons, notes = await _authoritative_verdict(
        config, entity_id, client, effort, ledger
    )

    warnings = list(run.evidence.warnings) if run.evidence else []
    warnings.extend(notes)
    if not run.finished:
        warnings.append(f"the agent {run.stop_reason} -- the brief rests on partial evidence.")
    if dropped:
        warnings.append(
            f"{len(dropped)} finding(s) dropped: the quote was not found in any source the "
            "agent actually fetched (the model cited something it did not read)."
        )

    return FinishedBrief(
        entity_id=entity_id,
        subject=subject,
        model=model,
        headline=brief.headline,
        thesis=brief.thesis,
        recommendation=brief.recommendation,
        verdict=verdict,
        veto_reasons=veto_reasons,
        verified_findings=verified,
        dropped_findings=dropped,
        metrics_block=run.metrics_block,
        steps=run.steps,
        warnings=warnings,
    )


def _verify_findings(
    findings: list[BriefFinding], run: AgentRun
) -> tuple[list[BriefFinding], list[tuple[BriefFinding, str]]]:
    """Drop any finding whose quote is not in a source the agent fetched.

    A quote is accepted if it appears in its cited source, OR in any other fetched
    source (the model sometimes attributes a real quote to the wrong id) -- but a
    quote found nowhere is a fabrication and is dropped."""
    sources: dict[str, str] = {}
    if run.evidence is not None:
        sources.update(run.evidence.texts_by_accession)
    sources.update(run.web_sources)

    verified: list[BriefFinding] = []
    dropped: list[tuple[BriefFinding, str]] = []
    for finding in findings:
        cited = sources.get(finding.source)
        if cited is not None and quote_in_text(finding.quoted_span, cited):
            verified.append(finding)
        elif any(quote_in_text(finding.quoted_span, text) for text in sources.values()):
            verified.append(finding)  # real quote, possibly mis-attributed source
        else:
            dropped.append((finding, "quote not found in any fetched source"))
    return verified, dropped


async def _authoritative_verdict(
    config: Config,
    entity_id: str | None,
    client: LLMClient,
    effort: Effort | None,
    ledger: CostLedger | None,
) -> tuple[Verdict, list[str], list[str]]:
    """The code-owned verdict. For an ingested entity, run the disciplined pipeline
    and take ITS `decide_verdict` result -- the agent cannot override it. Otherwise
    there is no checkable disqualifier, so NO_VETO with an honest note.

    Returns (verdict, veto_reasons, notes)."""
    if not entity_id or not config.db_path.exists():
        return Verdict.NO_VETO, [], [
            "no code-decided veto ran: the subject is not an ingested entity, so the verdict "
            "reflects only the absence of a checkable disqualifier, not a clean bill of health."
        ]

    reports = await research_entities(config, [entity_id], client=client, effort=effort)
    if not reports:
        return Verdict.NO_VETO, [], [
            f"the disciplined veto could not run for {entity_id} (no ingested filings); the "
            "verdict reflects only the absence of a checkable disqualifier."
        ]
    memo = reports[0].memo
    return memo.verdict, list(memo.veto_reasons), []


def _digest(run: AgentRun) -> str:
    """A compact record of what the agent did, for the compose prompt."""
    if not run.steps:
        return "(no tools were called)"
    lines = []
    for i, step in enumerate(run.steps, start=1):
        head = step.observation.splitlines()[0] if step.observation else ""
        status = "" if step.ok else " [failed]"
        lines.append(f"{i}. {step.tool}({step.tool_input}){status} -> {head[:140]}")
    return "\n".join(lines)


def _evidence_catalog(run: AgentRun) -> str:
    """The valid source ids the model may cite, so it quotes real things."""
    lines: list[str] = []
    if run.evidence is not None:
        for accession, text in run.evidence.texts_by_accession.items():
            lines.append(f"- {accession}: SEC filing ({len(text):,} chars)")
    for url, text in run.web_sources.items():
        lines.append(f"- {url}: web page ({len(text):,} chars)")
    return "\n".join(lines) or "(no sources fetched -- you have no basis for any finding)"
