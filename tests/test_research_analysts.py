"""Tests for the bull/bear/skeptic debate stage.

No network: every response is scripted through FakeClient (tests/conftest.py). The
things worth pinning here are that each side comes back on the right stance, that
the skeptic can veto in BOTH directions, and -- the load-bearing one -- that the
skeptic actually SEES the bull case, since a skeptic reviewing nothing is just a
second opinion, not a refutation.
"""

from __future__ import annotations

import json

from scout.harness.protocol import Message
from scout.research.analysts import argue, run_debate, skeptic_review
from scout.research.evidence import EvidencePack, Excerpt
from scout.research.models import (
    AnalystView,
    Finding,
    FindingCategory,
    Severity,
    Stance,
)
from scout.research.prompts.analysts import (
    build_analyst_messages,
    build_skeptic_messages,
)

# --------------------------------------------------------------------------
# Fixtures: a small anonymised pack and one verified finding
# --------------------------------------------------------------------------

_METRICS_BLOCK = "Period: 2024-12-31 (FY), currency USD\n  ncav_to_mktcap = 1.8 [computed]"

_FINDING = Finding(
    category=FindingCategory.GOING_CONCERN,
    claim="The auditor raised substantial doubt about the company's ability to continue.",
    quoted_span="substantial doubt about its ability to continue as a going concern",
    source_accession="0001-24-000001",
    severity=Severity.HIGH,
)


def _pack() -> EvidencePack:
    return EvidencePack(
        entity_id="ENT-1",
        excerpts=[
            Excerpt(
                source_accession="0001-24-000001",
                category="going_concern",
                text="There is substantial doubt about its ability to continue as a going concern.",
                char_start=42,
            )
        ],
        texts_by_accession={
            "0001-24-000001": "There is substantial doubt about its ability to continue as a going concern."
        },
        metrics_block=_METRICS_BLOCK,
    )


# Scripted structured responses (the FakeClient returns these as response text).
_BULL_JSON = json.dumps(
    {
        "stance": "bull",
        "points": [
            "Trades at 1.8x net current asset value per the metrics block.",
            "No toxic convertible appears in the verified findings.",
        ],
    }
)
_BEAR_JSON = json.dumps(
    {
        "stance": "bear",
        "points": [
            "Auditor flagged substantial doubt about going concern.",
            "Balance-sheet cheapness means nothing if the company cannot fund operations.",
        ],
    }
)
_SKEPTIC_DISQUALIFYING = json.dumps(
    {
        "refuted_claims": ["The 'no toxic convertible' point ignores the confirmed going-concern finding."],
        "disqualifying": True,
        "reasoning": "A confirmed going-concern doubt with no described funding path is fatal for a microcap.",
    }
)
_SKEPTIC_CLEARED = json.dumps(
    {
        "refuted_claims": [],
        "disqualifying": False,
        "reasoning": "The findings show no confirmed disqualifying red flag; the bull points are supported.",
    }
)


# --------------------------------------------------------------------------
# argue()
# --------------------------------------------------------------------------


async def test_argue_bull_returns_bull_view_with_points(make_client):
    client = make_client([_BULL_JSON])

    view = await argue(client, Stance.BULL, _pack(), [_FINDING])

    assert isinstance(view, AnalystView)
    assert view.stance is Stance.BULL
    assert view.points  # non-empty


async def test_argue_bear_returns_bear_view(make_client):
    client = make_client([_BEAR_JSON])

    view = await argue(client, Stance.BEAR, _pack(), [_FINDING])

    assert view.stance is Stance.BEAR
    assert view.points


# --------------------------------------------------------------------------
# skeptic_review() -- both directions
# --------------------------------------------------------------------------


async def test_skeptic_can_disqualify(make_client):
    client = make_client([_SKEPTIC_DISQUALIFYING])
    bull = AnalystView.model_validate(json.loads(_BULL_JSON))

    verdict = await skeptic_review(client, _pack(), [_FINDING], bull)

    assert verdict.disqualifying is True
    assert verdict.refuted_claims


async def test_skeptic_can_clear(make_client):
    client = make_client([_SKEPTIC_CLEARED])
    bull = AnalystView.model_validate(json.loads(_BULL_JSON))

    verdict = await skeptic_review(client, _pack(), [_FINDING], bull)

    assert verdict.disqualifying is False


# --------------------------------------------------------------------------
# run_debate() -- orchestration
# --------------------------------------------------------------------------


async def test_run_debate_runs_all_three_on_the_right_stances(make_client):
    # Distinct clients with distinct scripts prove each side ran and that the
    # right response came back on the right stance.
    bull_client = make_client([_BULL_JSON])
    bear_client = make_client([_BEAR_JSON])
    skeptic_client = make_client([_SKEPTIC_DISQUALIFYING])

    bull, bear, skeptic = await run_debate(
        bull_client, bear_client, skeptic_client, _pack(), [_FINDING]
    )

    assert bull.stance is Stance.BULL
    assert bear.stance is Stance.BEAR
    assert skeptic.disqualifying is True

    # Each client was called exactly once -- bull and bear did run.
    assert len(bull_client.calls) == 1
    assert len(bear_client.calls) == 1
    assert len(skeptic_client.calls) == 1


async def test_run_debate_skeptic_sees_the_bull_case(make_client):
    # A bull point with a distinctive marker we can look for in the skeptic's prompt.
    marker = "DISTINCTIVE-BULL-MARKER-42"
    bull_json = json.dumps({"stance": "bull", "points": [f"{marker} cheap on NCAV."]})
    bull_client = make_client([bull_json])
    bear_client = make_client([_BEAR_JSON])
    skeptic_client = make_client([_SKEPTIC_CLEARED])

    await run_debate(bull_client, bear_client, skeptic_client, _pack(), [_FINDING])

    # The skeptic's single call must carry the bull marker in its messages.
    skeptic_prompt = "\n".join(m.content for m in skeptic_client.calls[0]["messages"])
    assert marker in skeptic_prompt


# --------------------------------------------------------------------------
# Prompt builders (pure functions)
# --------------------------------------------------------------------------


def test_analyst_messages_include_metrics_finding_and_anon_note():
    messages = build_analyst_messages(Stance.BULL, _pack(), [_FINDING])
    assert all(isinstance(m, Message) for m in messages)

    text = "\n".join(m.content for m in messages)
    assert _METRICS_BLOCK in text  # the metrics block, verbatim
    assert _FINDING.quoted_span in text  # a verified finding
    assert "ANONYMISED" in text or "[COMPANY]" in text  # anonymisation note

    # Stable instructions first, variable evidence last.
    assert messages[0].role == "system"
    assert messages[-1].role == "user"


def test_skeptic_messages_include_bull_case_and_caution():
    bull = AnalystView.model_validate(json.loads(_BULL_JSON))
    messages = build_skeptic_messages(_pack(), [_FINDING], bull)

    text = "\n".join(m.content for m in messages)
    # The bull's actual points must be present for the review to be adversarial.
    for point in bull.points:
        assert point in text
    # The metrics and a finding travel with the skeptic too.
    assert _METRICS_BLOCK in text
    assert _FINDING.quoted_span in text
    # Caution-by-default must be stated explicitly (design requirement).
    assert "VETO" in text.upper()


def test_analyst_messages_reflect_stance():
    bull_text = build_analyst_messages(Stance.BULL, _pack(), [_FINDING])[0].content
    bear_text = build_analyst_messages(Stance.BEAR, _pack(), [_FINDING])[0].content
    assert "BULL" in bull_text
    assert "BEAR" in bear_text
    assert bull_text != bear_text
