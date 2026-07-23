"""The brief composer's guardrails, and tool gating. No network.

The two properties that make the agent safe despite driving itself: a finding
whose quote is not in a source it actually fetched is DROPPED, and the verdict is
code-owned (NO_VETO with an honest note when there is no ingested entity to check
against, never a fabricated all-clear).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from scout.agent.brief import compose_brief
from scout.agent.models import AgentRun, FinishedBrief
from scout.agent.tools import ToolContext, build_tools
from scout.config import Config
from scout.output import render_brief, write_brief
from scout.research.evidence import EvidencePack
from scout.research.models import Verdict

GENERATED = datetime(2026, 7, 23, 12, 0, 0, tzinfo=UTC)
REAL_QUOTE = "substantial doubt about its ability to continue as a going concern"


def _brief_json(findings) -> str:
    return json.dumps(
        {"headline": "A shell.", "thesis": "Risky.", "recommendation": "Avoid.", "findings": findings}
    )


async def test_brief_drops_a_quote_not_in_any_fetched_source(make_client, all_modes, tmp_path):
    run = AgentRun(goal="g", evidence=EvidencePack(entity_id="790273"), finished=True)
    run.evidence.texts_by_accession["acc-1"] = (
        f"The auditor noted {REAL_QUOTE}, a serious concern for the company."
    )
    script = _brief_json(
        [
            {"claim": "real", "quoted_span": REAL_QUOTE, "source": "acc-1", "is_red_flag": True},
            {"claim": "made up", "quoted_span": "the company guarantees a tripling of revenue",
             "source": "acc-1", "is_red_flag": False},
        ]
    )
    client = make_client(script=[script], capabilities=all_modes)
    config = Config(user_agent="scout/0.1 t@example.com", data_dir=tmp_path)

    brief = await compose_brief(client, run, subject="Co", entity_id=None, config=config, model="m")

    # The verifiable finding survives; the fabricated quote is dropped.
    assert [f.claim for f in brief.verified_findings] == ["real"]
    assert len(brief.dropped_findings) == 1
    assert brief.dropped_findings[0][0].claim == "made up"


async def test_verdict_is_no_veto_with_a_note_when_no_entity_to_check(make_client, all_modes, tmp_path):
    run = AgentRun(goal="g", evidence=EvidencePack(entity_id="x"), finished=True)
    client = make_client(script=[_brief_json([])], capabilities=all_modes)
    config = Config(user_agent="scout/0.1 t@example.com", data_dir=tmp_path)  # no db

    brief = await compose_brief(client, run, subject="Theme", entity_id=None, config=config, model="m")

    # No ingested entity => no code veto could run, and the brief says so rather
    # than implying a clean bill of health.
    assert brief.verdict == Verdict.NO_VETO
    assert any("not an ingested entity" in w for w in brief.warnings)


async def test_partial_run_is_flagged(make_client, all_modes, tmp_path):
    run = AgentRun(goal="g", evidence=EvidencePack(entity_id="x"), finished=False,
                   stop_reason="reached the 3-step limit before finishing.")
    client = make_client(script=[_brief_json([])], capabilities=all_modes)
    config = Config(user_agent="scout/0.1 t@example.com", data_dir=tmp_path)

    brief = await compose_brief(client, run, subject="Co", entity_id=None, config=config, model="m")
    assert any("partial evidence" in w for w in brief.warnings)


def test_tools_are_gated_by_available_resources():
    # No store and no web provider: only the always-available filing/fetch tools.
    ctx = ToolContext(config=None, http=None, client=None, store=None, web_provider=None)  # type: ignore[arg-type]
    names = {t.name for t in build_tools(ctx)}
    assert names == {"search_filings", "read_filing", "fetch_url"}
    assert "compute_metrics" not in names  # no store -> no metric tools
    assert "web_search" not in names       # no provider -> no web search

    # A store present unlocks the deterministic + disciplined tools.
    ctx_with_store = ToolContext(
        config=None, http=None, client=None, store=object(), web_provider=None  # type: ignore[arg-type]
    )
    with_store = {t.name for t in build_tools(ctx_with_store)}
    assert {"get_fundamentals", "compute_metrics", "screen", "deep_analyze"} <= with_store


def test_render_and_write_brief_keep_citations(tmp_path):
    from scout.agent.models import BriefFinding

    brief = FinishedBrief(
        entity_id="790273",
        subject="CONECTISYS CORP",
        model="deepseek-v4-flash",
        headline="A shell with going-concern doubt.",
        thesis="Uninvestable.",
        recommendation="Avoid.",
        verdict=Verdict.VETO,
        veto_reasons=["Going concern with no funding path."],
        verified_findings=[
            BriefFinding(claim="Going concern.", quoted_span=REAL_QUOTE,
                         source="acc-1", is_red_flag=True)
        ],
    )
    md = render_brief(brief, run_id="r1", generated=GENERATED)
    assert "VETO" in md and REAL_QUOTE in md and "acc-1" in md and "🚩" in md

    written = write_brief(brief, tmp_path, run_id="r1", generated=GENERATED)
    assert (tmp_path / "2026-07-23" / "790273-conectisys-corp.md") in written
    data = json.loads((tmp_path / "2026-07-23" / "790273-conectisys-corp.json").read_text("utf-8"))
    assert data["verdict"] == "veto"
    assert data["verified_findings"][0]["quoted_span"] == REAL_QUOTE
