"""Saving a ResearchReport to Markdown + JSON.

The memo must keep its citations: the whole premise is that every claim traces to
the filing, so the saved Markdown carries the quoted span and its accession, and
the JSON round-trips the full structured report.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from scout.output import render_markdown, report_to_dict, write_reports
from scout.research.models import (
    AnalystView,
    Finding,
    FindingCategory,
    ResearchMemo,
    Severity,
    SkepticVerdict,
    Stance,
    Verdict,
)
from scout.research.pipeline import ResearchReport

GENERATED = datetime(2026, 7, 23, 12, 0, 0, tzinfo=UTC)
QUOTE = "substantial doubt about the Company's ability to continue as a going concern"


def _report(*, vetoed: bool = True) -> ResearchReport:
    finding = Finding(
        category=FindingCategory.GOING_CONCERN,
        claim="Going-concern doubt disclosed.",
        quoted_span=QUOTE,
        source_accession="acc-123",
        severity=Severity.CRITICAL,
    )
    memo = ResearchMemo(
        entity_id="790273",
        headline="A shell company with going-concern doubt.",
        thesis="Cheap on paper but uninvestable.",
        verdict=Verdict.VETO if vetoed else Verdict.NO_VETO,
        veto_reasons=["Going concern with no funding path."] if vetoed else [],
    )
    return ResearchReport(
        entity_id="790273",
        name="CONECTISYS CORP",
        memo=memo,
        verified_findings=[finding],
        dropped_citations=[(finding, "quote not found")],
        bull=AnalystView(stance=Stance.BULL, points=["Trades below net cash."]),
        bear=AnalystView(stance=Stance.BEAR, points=["No operations."]),
        skeptic=SkepticVerdict(refuted_claims=[], disqualifying=True, reasoning="Runway too short."),
        warnings=["high citation-failure rate"],
    )


def test_write_reports_creates_markdown_json_and_index(tmp_path):
    written = write_reports(
        [_report()], tmp_path, run_id="run-1", model="deepseek-v4-flash", generated=GENERATED
    )
    day_dir = tmp_path / "2026-07-23"
    md = day_dir / "790273-conectisys-corp.md"
    js = day_dir / "790273-conectisys-corp.json"
    index = day_dir / "_run-run-1.json"

    assert md.exists() and js.exists() and index.exists()
    assert set(written) == {md, js, index}

    index_data = json.loads(index.read_text(encoding="utf-8"))
    assert index_data["researched"] == 1
    assert index_data["vetoed"] == 1
    assert index_data["entries"][0]["entity_id"] == "790273"


def test_markdown_keeps_the_citation_and_verdict():
    md = render_markdown(_report(), run_id="run-1", model="deepseek-v4-flash", generated=GENERATED)
    assert "VETO" in md
    assert QUOTE in md               # the quoted span survives
    assert "acc-123" in md           # traced to its filing
    assert "Going concern with no funding path." in md
    assert "deepseek-v4-flash" in md


def test_json_round_trips_the_structured_report():
    data = report_to_dict(_report(), run_id="run-1", model="m", generated=GENERATED)
    assert data["verdict"] == "veto"
    assert data["vetoed"] is True
    assert data["verified_findings"][0]["quoted_span"] == QUOTE
    assert data["verified_findings"][0]["severity"] == "critical"  # enum -> value
    assert data["dropped_citations"][0]["reason"] == "quote not found"
    assert data["fabrication_rate"] == 0.5  # 1 verified, 1 dropped
    # The whole thing must be JSON-serializable (no enums/datetimes leaking).
    json.dumps(data)


def test_no_veto_report_renders_without_a_veto_section():
    md = render_markdown(
        _report(vetoed=False), run_id="r", model="m", generated=GENERATED
    )
    assert "no veto" in md
    assert "Veto reasons" not in md


def test_slug_is_filesystem_safe(tmp_path):
    report = _report()
    report.name = "Ácmé, Inc. / Holdings!!!"
    written = write_reports([report], tmp_path, run_id="r", model="m", generated=GENERATED)
    md = next(p for p in written if p.suffix == ".md")
    # Only lowercase alphanumerics and single hyphens in the slug portion.
    slug = md.stem.split("-", 1)[1]
    assert slug and all(c.islower() or c.isdigit() or c == "-" for c in slug)
    assert "--" not in slug
