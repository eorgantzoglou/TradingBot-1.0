"""End-to-end pipeline test with a scripted FakeClient.

Proves the wiring and, most importantly, the trust guarantee: a fabricated
citation the extractor emits is dropped BEFORE the debate and the memo, and the
code-decided veto cannot be overridden by the model. No network.
"""

from __future__ import annotations

import json

from scout.research.evidence import EvidencePack
from scout.research.pipeline import research_candidate

# Five LLM calls per candidate, in order: extract, bull, bear, skeptic, memo.


def _pack() -> EvidencePack:
    pack = EvidencePack(entity_id="42")
    pack.texts_by_accession["acc-1"] = (
        "The financial statements raise substantial doubt about the Company's "
        "ability to continue as a going concern. The Company entered a convertible "
        "note with a variable conversion price equal to the lowest closing price."
    )
    pack.metrics_block = "cash_runway_months = 3 [annual]"
    return pack


def _extract_json(*, real_quote: bool) -> str:
    quote = (
        "substantial doubt about the Company's ability to continue as a going concern"
        if real_quote
        else "the company will certainly triple its revenue next quarter guaranteed"
    )
    return json.dumps(
        {
            "findings": [
                {
                    "category": "going_concern",
                    "claim": "Going-concern doubt disclosed.",
                    "quoted_span": quote,
                    "source_accession": "acc-1",
                    "severity": "critical",
                }
            ],
            "notes": "",
        }
    )


def _bull() -> str:
    return json.dumps({"stance": "bull", "points": ["Cheap on assets", "Insider ownership"]})


def _bear() -> str:
    return json.dumps({"stance": "bear", "points": ["Going concern", "Toxic convertible"]})


def _skeptic(disqualifying: bool) -> str:
    return json.dumps(
        {"refuted_claims": [], "disqualifying": disqualifying, "reasoning": "Runway too short."}
    )


def _memo(verdict: str) -> str:
    return json.dumps(
        {
            "entity_id": "SHOULD_BE_OVERWRITTEN",
            "headline": "A distressed micro-cap.",
            "thesis": "Cheap but burning cash with a toxic convertible.",
            "verdict": verdict,
            "veto_reasons": ["model-invented reason"],
        }
    )


async def test_pipeline_verifies_citations_and_vetoes(make_client, all_modes):
    # Extractor returns a REAL quote -> the critical finding survives verification
    # -> decide_verdict vetoes on the critical severity, regardless of the model's
    # memo claiming no_veto.
    client = make_client(
        script=[
            _extract_json(real_quote=True),
            _bull(),
            _bear(),
            _skeptic(disqualifying=False),
            _memo("no_veto"),  # the model tries to wave it through...
        ],
        capabilities=all_modes,
    )

    report = await research_candidate(_pack(), client=client, name="Distressed Co")

    # The critical finding verified, so the code-owned gate vetoes.
    assert report.vetoed
    assert report.memo.verdict.value == "veto"
    # entity_id is forced from the pack, not the model's placeholder.
    assert report.memo.entity_id == "42"
    assert len(report.verified_findings) == 1
    assert not report.dropped_citations


async def test_pipeline_drops_fabricated_citation(make_client, all_modes):
    # Extractor's quote is NOT in the filing -> the finding is dropped before the
    # debate, so no critical finding reaches the gate. Skeptic says fine -> NO_VETO.
    client = make_client(
        script=[
            _extract_json(real_quote=False),
            _bull(),
            _bear(),
            _skeptic(disqualifying=False),
            _memo("no_veto"),
        ],
        capabilities=all_modes,
    )

    report = await research_candidate(_pack(), client=client, name="Clean Co")

    assert not report.vetoed
    assert len(report.verified_findings) == 0
    assert len(report.dropped_citations) == 1
    # The fabricated finding never influenced the outcome.
    assert report.fabrication_rate == 1.0


async def test_skeptic_can_veto_without_critical_finding(make_client, all_modes):
    client = make_client(
        script=[
            _extract_json(real_quote=False),  # dropped, no critical finding survives
            _bull(),
            _bear(),
            _skeptic(disqualifying=True),  # ...but the skeptic disqualifies
            _memo("no_veto"),
        ],
        capabilities=all_modes,
    )

    report = await research_candidate(_pack(), client=client, name="Doubtful Co")

    assert report.vetoed
    assert any("Runway" in r for r in report.memo.veto_reasons)
