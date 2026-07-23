"""Tests for the memo stage: the deterministic veto gate and prose synthesis.

The load-bearing test is `test_code_overrules_model_verdict` -- it proves a model
that returns NO_VETO cannot escape a code-decided VETO. Everything else pins the
pure gate (`decide_verdict`) exhaustively, because that function IS the thing
that removes a fraud from the shortlist.
"""

from __future__ import annotations

import json

from scout.harness.protocol import Capabilities
from scout.research.evidence import EvidencePack
from scout.research.memo import decide_verdict, write_memo
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
from scout.research.prompts.memo import build_memo_messages

# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _finding(severity: Severity, claim: str = "A confirmed red flag.") -> Finding:
    return Finding(
        category=FindingCategory.GOING_CONCERN,
        claim=claim,
        quoted_span="substantial doubt about its ability to continue as a going concern",
        source_accession="0001-24",
        severity=severity,
    )


def _skeptic(*, disqualifying: bool, reasoning: str = "Weak but not fatal.") -> SkepticVerdict:
    return SkepticVerdict(disqualifying=disqualifying, reasoning=reasoning)


def _pack() -> EvidencePack:
    return EvidencePack(
        entity_id="US-0000320193",
        metrics_block="Period: 2024-12-31 (FY), currency USD\n  ev_to_ebit = -3.2 [computed]",
    )


def _bull() -> AnalystView:
    return AnalystView(
        stance=Stance.BULL,
        points=["Trades below net cash.", "No debt on the balance sheet."],
    )


def _bear() -> AnalystView:
    return AnalystView(
        stance=Stance.BEAR,
        points=["Revenue is shrinking.", "Cash burn accelerating."],
    )


def _memo_json(entity_id: str = "MODEL-SAYS-THIS", verdict: str = "no_veto") -> str:
    return json.dumps(
        {
            "entity_id": entity_id,
            "headline": "A cash-rich microcap trading below liquidation value.",
            "thesis": "The screen flags deep value; the main risk is cash burn.",
            "verdict": verdict,
            "veto_reasons": ["model invented reason"],
        }
    )


# --------------------------------------------------------------------------
# decide_verdict: the pure gate, exhaustively
# --------------------------------------------------------------------------


def test_critical_finding_vetoes():
    verdict, reasons = decide_verdict(
        [_finding(Severity.CRITICAL, claim="Toxic convertible confirmed.")],
        _skeptic(disqualifying=False),
    )
    assert verdict is Verdict.VETO
    assert "Toxic convertible confirmed." in reasons


def test_skeptic_disqualifying_vetoes_without_critical_finding():
    verdict, reasons = decide_verdict(
        [_finding(Severity.MEDIUM)],
        _skeptic(disqualifying=True, reasoning="Going concern with no runway."),
    )
    assert verdict is Verdict.VETO
    assert "Going concern with no runway." in reasons


def test_neither_gate_is_no_veto_with_empty_reasons():
    verdict, reasons = decide_verdict(
        [_finding(Severity.HIGH)],
        _skeptic(disqualifying=False),
    )
    assert verdict is Verdict.NO_VETO
    assert reasons == []


def test_both_gates_include_both_reasons():
    verdict, reasons = decide_verdict(
        [_finding(Severity.CRITICAL, claim="Auditor resigned citing fraud.")],
        _skeptic(disqualifying=True, reasoning="Un-investable regardless of cheapness."),
    )
    assert verdict is Verdict.VETO
    assert "Auditor resigned citing fraud." in reasons
    assert "Un-investable regardless of cheapness." in reasons


def test_empty_findings_and_not_disqualifying_is_no_veto():
    verdict, reasons = decide_verdict([], _skeptic(disqualifying=False))
    assert verdict is Verdict.NO_VETO
    assert reasons == []


# --------------------------------------------------------------------------
# build_memo_messages: pure prompt builder
# --------------------------------------------------------------------------


def test_prompt_includes_all_inputs_and_decided_verdict():
    pack = _pack()
    finding = _finding(Severity.CRITICAL, claim="Toxic convertible confirmed.")
    messages = build_memo_messages(
        pack,
        [finding],
        _bull(),
        _bear(),
        _skeptic(disqualifying=True, reasoning="No runway remains."),
        Verdict.VETO,
        ["Toxic convertible confirmed.", "No runway remains."],
    )
    blob = "\n".join(m.content for m in messages)

    assert pack.metrics_block in blob  # authoritative numbers
    assert "Toxic convertible confirmed." in blob  # a finding
    assert "Trades below net cash." in blob  # bull
    assert "Revenue is shrinking." in blob  # bear
    assert "No runway remains." in blob  # skeptic
    assert "veto" in blob.lower()  # the decided verdict


# --------------------------------------------------------------------------
# write_memo: FakeClient scripted, no network
# --------------------------------------------------------------------------


async def test_entity_id_forced_from_pack(make_client):
    client = make_client([_memo_json(entity_id="MODEL-SAYS-THIS")])

    memo = await write_memo(
        client,
        _pack(),
        [_finding(Severity.LOW)],
        _bull(),
        _bear(),
        _skeptic(disqualifying=False),
    )

    assert isinstance(memo, ResearchMemo)
    assert memo.entity_id == "US-0000320193"  # pack wins, not the model's JSON


async def test_code_overrules_model_verdict(make_client):
    # Model says NO_VETO, code says VETO (a CRITICAL finding). Code must win.
    client = make_client([_memo_json(verdict="no_veto")])

    memo = await write_memo(
        client,
        _pack(),
        [_finding(Severity.CRITICAL, claim="Going-concern with no runway.")],
        _bull(),
        _bear(),
        _skeptic(disqualifying=False),
    )

    assert memo.verdict is Verdict.VETO
    assert memo.veto_reasons == ["Going-concern with no runway."]
    assert "model invented reason" not in memo.veto_reasons


async def test_prose_comes_through_from_model(make_client):
    client = make_client([_memo_json()])

    memo = await write_memo(
        client,
        _pack(),
        [],
        _bull(),
        _bear(),
        _skeptic(disqualifying=False),
    )

    assert memo.headline == "A cash-rich microcap trading below liquidation value."
    assert memo.thesis == "The screen flags deep value; the main risk is cash burn."


async def test_capabilities_are_respected(make_client):
    # A json-object-only provider still lands a valid memo (JSON_OBJECT rung).
    client = make_client([_memo_json()], capabilities=Capabilities(json_object=True))

    memo = await write_memo(
        client,
        _pack(),
        [],
        _bull(),
        _bear(),
        _skeptic(disqualifying=False),
    )

    assert isinstance(memo, ResearchMemo)
