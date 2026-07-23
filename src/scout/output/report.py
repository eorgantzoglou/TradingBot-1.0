"""Write a ResearchReport to Markdown and JSON.

One report becomes two files under `reports/<YYYY-MM-DD>/`:

    <entity_id>-<slug>.md      the human memo -- verdict, thesis, cited findings
    <entity_id>-<slug>.json    the full structured report -- for diffing/automation

plus a `_run-<run_id>.json` index per `scout research` invocation, so a batch is
one thing to scan. The Markdown carries the quoted spans and their accessions
inline, because the whole premise of the tool is that every claim is traceable to
the filing text -- a memo without its citations would throw away the guarantee.

Reports are derived artifacts, not the pre-registration record the ledger is, so
re-running for the same entity on the same day overwrites its files. The
immutable history lives in the ledger and the archive; these are the current read.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scout.agent.models import BriefFinding, FinishedBrief
from scout.research.models import Verdict
from scout.research.pipeline import ResearchReport


def write_reports(
    reports: list[ResearchReport],
    out_dir: Path,
    *,
    run_id: str,
    model: str,
    generated: datetime | None = None,
) -> list[Path]:
    """Write every report as Markdown + JSON, plus a run index. Returns the paths.

    `generated` is injectable so tests are deterministic; it defaults to now.
    """
    generated = generated or datetime.now(UTC)
    day_dir = out_dir / generated.date().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    index_entries: list[dict[str, Any]] = []

    for report in reports:
        base = f"{report.entity_id}-{_slug(report.name or report.entity_id)}"
        md_path = day_dir / f"{base}.md"
        json_path = day_dir / f"{base}.json"

        md_path.write_text(
            render_markdown(report, run_id=run_id, model=model, generated=generated),
            encoding="utf-8",
        )
        json_path.write_text(
            json.dumps(
                report_to_dict(report, run_id=run_id, model=model, generated=generated),
                indent=2,
            ),
            encoding="utf-8",
        )
        written += [md_path, json_path]
        index_entries.append(
            {
                "entity_id": report.entity_id,
                "name": report.name,
                "verdict": report.memo.verdict.value,
                "vetoed": report.vetoed,
                "findings": len(report.verified_findings),
                "markdown": md_path.name,
                "json": json_path.name,
            }
        )

    index_path = day_dir / f"_run-{run_id}.json"
    index_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "generated": generated.isoformat(),
                "model": model,
                "researched": len(reports),
                "vetoed": sum(1 for r in reports if r.vetoed),
                "entries": index_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written.append(index_path)
    return written


def render_markdown(
    report: ResearchReport, *, run_id: str, model: str, generated: datetime
) -> str:
    """The human-readable memo. Every finding keeps its quote and accession."""
    memo = report.memo
    veto = memo.verdict == Verdict.VETO
    lines: list[str] = []

    lines.append(f"# {report.name or report.entity_id}")
    lines.append("")
    lines.append(f"**Entity:** `{report.entity_id}`  ")
    lines.append(f"**Verdict:** {'🚫 VETO' if veto else '✅ no veto'}  ")
    lines.append(f"**Model:** {model}  ")
    lines.append(f"**Generated:** {generated.isoformat(timespec='seconds')}  ")
    lines.append(f"**Run:** `{run_id}`")
    lines.append("")
    lines.append("> *Not investment advice. The LLM can veto a screened candidate; "
                 "it can never recommend buying one. Every number is computed in code.*")
    lines.append("")

    lines.append(f"## Headline\n\n{memo.headline}\n")
    lines.append(f"## Thesis\n\n{memo.thesis}\n")

    if veto:
        lines.append("## Veto reasons\n")
        for reason in memo.veto_reasons:
            lines.append(f"- ❌ {reason}")
        lines.append("")

    lines.append(f"## Cited findings ({len(report.verified_findings)})\n")
    if report.verified_findings:
        for finding in report.verified_findings:
            lines.append(f"### [{finding.severity.value}] {finding.category.value}")
            lines.append("")
            lines.append(finding.claim)
            lines.append("")
            lines.append(f"> {finding.quoted_span}")
            lines.append(f">\n> — filing `{finding.source_accession}`")
            lines.append("")
    else:
        lines.append("_None survived citation verification._\n")

    lines.append("## Debate\n")
    lines.append("**Bull**")
    lines.extend(f"- {point}" for point in report.bull.points)
    lines.append("")
    lines.append("**Bear**")
    lines.extend(f"- {point}" for point in report.bear.points)
    lines.append("")
    lines.append(f"**Skeptic** — {'disqualifying' if report.skeptic.disqualifying else 'not disqualifying'}")
    lines.append("")
    lines.append(report.skeptic.reasoning)
    if report.skeptic.refuted_claims:
        lines.append("")
        lines.append("Refuted claims:")
        lines.extend(f"- {claim}" for claim in report.skeptic.refuted_claims)
    lines.append("")

    if report.dropped_citations or report.warnings:
        lines.append("## Data quality\n")
        if report.dropped_citations:
            lines.append(
                f"- {len(report.dropped_citations)} finding(s) dropped: the quote was not "
                f"found in the cited filing (fabrication rate {report.fabrication_rate:.0%})."
            )
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def report_to_dict(
    report: ResearchReport, *, run_id: str, model: str, generated: datetime
) -> dict[str, Any]:
    """The full report as JSON-safe data. Pydantic parts dumped in json mode so
    enums become their string values."""
    return {
        "entity_id": report.entity_id,
        "name": report.name,
        "run_id": run_id,
        "model": model,
        "generated": generated.isoformat(),
        "verdict": report.memo.verdict.value,
        "vetoed": report.vetoed,
        "memo": report.memo.model_dump(mode="json"),
        "verified_findings": [f.model_dump(mode="json") for f in report.verified_findings],
        "dropped_citations": [
            {"finding": f.model_dump(mode="json"), "reason": reason}
            for f, reason in report.dropped_citations
        ],
        "fabrication_rate": report.fabrication_rate,
        "bull": report.bull.model_dump(mode="json"),
        "bear": report.bear.model_dump(mode="json"),
        "skeptic": report.skeptic.model_dump(mode="json"),
        "warnings": list(report.warnings),
    }


# --------------------------------------------------------------------------- #
# The agent's investigation brief (scout investigate)
# --------------------------------------------------------------------------- #


def write_brief(
    brief: FinishedBrief,
    out_dir: Path,
    *,
    run_id: str,
    generated: datetime | None = None,
) -> list[Path]:
    """Write one investigation brief as Markdown + JSON. Returns the paths."""
    generated = generated or datetime.now(UTC)
    day_dir = out_dir / generated.date().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)

    base = f"{brief.entity_id or 'investigation'}-{_slug(brief.subject)}"
    md_path = day_dir / f"{base}.md"
    json_path = day_dir / f"{base}.json"
    md_path.write_text(render_brief(brief, run_id=run_id, generated=generated), encoding="utf-8")
    json_path.write_text(
        json.dumps(brief_to_dict(brief, run_id=run_id, generated=generated), indent=2),
        encoding="utf-8",
    )
    return [md_path, json_path]


def render_brief(brief: FinishedBrief, *, run_id: str, generated: datetime) -> str:
    """The human-readable investigation brief. Findings keep quote + source."""
    veto = brief.verdict == Verdict.VETO
    lines: list[str] = [f"# {brief.subject}", ""]
    if brief.entity_id:
        lines.append(f"**Entity:** `{brief.entity_id}`  ")
    lines.append(f"**Verdict:** {'🚫 VETO' if veto else '✅ no veto'}  ")
    lines.append(f"**Model:** {brief.model}  ")
    lines.append(f"**Generated:** {generated.isoformat(timespec='seconds')}  ")
    lines.append(f"**Run:** `{run_id}`")
    lines.append("")
    lines.append("> *Not investment advice. The agent may surface a candidate; the veto is "
                 "decided in code and every number is code-computed.*")
    lines.append("")
    lines.append(f"## Headline\n\n{brief.headline}\n")
    lines.append(f"## Thesis\n\n{brief.thesis}\n")
    lines.append(f"## Recommendation\n\n{brief.recommendation}\n")

    if veto:
        lines.append("## Veto reasons (code-decided)\n")
        lines.extend(f"- ❌ {reason}" for reason in brief.veto_reasons)
        lines.append("")

    lines.append(f"## Findings ({len(brief.verified_findings)} verified)\n")
    if brief.verified_findings:
        for finding in brief.verified_findings:
            lines.append(f"### {'🚩 ' if finding.is_red_flag else ''}{finding.claim}")
            lines.append("")
            lines.append(f"> {finding.quoted_span}")
            lines.append(f">\n> — source `{finding.source}`")
            lines.append("")
    else:
        lines.append("_No finding survived citation verification._\n")

    if brief.metrics_block:
        lines.append(f"## Code-computed metrics\n\n```\n{brief.metrics_block}\n```\n")

    lines.append("## What the agent did\n")
    if brief.steps:
        for i, step in enumerate(brief.steps, start=1):
            flag = "" if step.ok else " (failed)"
            lines.append(f"{i}. **{step.tool}**{flag} — {step.thought}")
    else:
        lines.append("_No tools were called._")
    lines.append("")

    if brief.dropped_findings or brief.warnings:
        lines.append("## Data quality\n")
        for finding, reason in brief.dropped_findings:
            lines.append(f"- dropped: \"{finding.claim}\" — {reason}")
        for warning in brief.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def brief_to_dict(brief: FinishedBrief, *, run_id: str, generated: datetime) -> dict[str, Any]:
    """The brief as JSON-safe data."""
    return {
        "subject": brief.subject,
        "entity_id": brief.entity_id,
        "run_id": run_id,
        "model": brief.model,
        "generated": generated.isoformat(),
        "verdict": brief.verdict.value,
        "vetoed": brief.vetoed,
        "headline": brief.headline,
        "thesis": brief.thesis,
        "recommendation": brief.recommendation,
        "veto_reasons": list(brief.veto_reasons),
        "verified_findings": [_finding_dict(f) for f in brief.verified_findings],
        "dropped_findings": [
            {"finding": _finding_dict(f), "reason": reason}
            for f, reason in brief.dropped_findings
        ],
        "metrics_block": brief.metrics_block,
        "steps": [
            {"tool": s.tool, "tool_input": s.tool_input, "thought": s.thought, "ok": s.ok}
            for s in brief.steps
        ],
        "warnings": list(brief.warnings),
    }


def _finding_dict(finding: BriefFinding) -> dict[str, Any]:
    return {
        "claim": finding.claim,
        "quoted_span": finding.quoted_span,
        "source": finding.source,
        "is_red_flag": finding.is_red_flag,
    }


_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    """A filesystem-safe slug: lowercase, alphanumerics joined by single hyphens.

    Bounded in length so a pathological company name cannot produce a path the OS
    rejects; empty input (a nameless entity) falls back to a constant.
    """
    slug = _SLUG_STRIP.sub("-", text.lower()).strip("-")
    return slug[:60] or "entity"
