"""Tests for the LLM extraction stage.

No network: every LLM call goes through conftest's scriptable FakeClient (via the
`make_client` fixture), so we assert on exactly the JSON we scripted and on the
exact prompt we sent. The load-bearing checks are (a) that a scripted
ExtractionResult round-trips through the harness into a validated object, (b) that
the prompt actually carries the metrics block, the accession label, the
anonymisation note and the filing text, and (c) that build_user_prompt shows a
short filing whole but falls back to excerpts for a long one.
"""

from __future__ import annotations

import json

from scout.research.evidence import EvidencePack, Excerpt
from scout.research.extract import extract_findings
from scout.research.models import ExtractionResult, FindingCategory, Severity
from scout.research.prompts.extract import build_user_prompt
from scout.research.verify import verify_findings

# A real sentence the model can quote verbatim; it must be a substring of the
# short filing's text so verify.py would confirm it.
GOING_CONCERN_SENTENCE = (
    "These conditions raise substantial doubt about the Company's ability to "
    "continue as a going concern."
)

SHORT_ACCESSION = "0001111-24-000001"
LONG_ACCESSION = "0002222-24-000009"

_METRICS_BLOCK = "Period: 2023-12-31 (FY), currency USD\n  net_cash = -1.2e6 [reported]\n  ev = 3.4e6 [computed]"

# Distinct markers let us prove which branch build_user_prompt took without
# asserting on huge strings: one appears only in full text, one only in an excerpt.
_FULLTEXT_MARKER = "ZZZ_ONLY_IN_FULL_TEXT_ZZZ"
_EXCERPT_MARKER = "QQQ_ONLY_IN_EXCERPT_QQQ"


def _short_pack() -> EvidencePack:
    """One short, anonymised filing shown whole (well under the 12k cap)."""
    text = (
        "Item 1A. Risk Factors. [COMPANY] has incurred recurring losses. "
        f"{GOING_CONCERN_SENTENCE} "
        "The Company entered into a convertible note with a variable conversion "
        "price equal to 60% of the lowest trading price over the prior 20 days."
    )
    return EvidencePack(
        entity_id="test-entity",
        texts_by_accession={SHORT_ACCESSION: text},
        excerpts=[
            Excerpt(
                source_accession=SHORT_ACCESSION,
                category="going_concern",
                text=GOING_CONCERN_SENTENCE,
                char_start=0,
            )
        ],
        metrics_block=_METRICS_BLOCK,
    )


def _long_pack() -> EvidencePack:
    """One filing whose full text is over the cap, so excerpts are used instead."""
    long_text = ("boilerplate padding sentence. " * 500) + _FULLTEXT_MARKER + (" tail padding. " * 100)
    assert len(long_text) > 12_000  # guard: the branch under test requires this
    return EvidencePack(
        entity_id="test-entity",
        texts_by_accession={LONG_ACCESSION: long_text},
        excerpts=[
            Excerpt(
                source_accession=LONG_ACCESSION,
                category="going_concern",
                text=f"{_EXCERPT_MARKER} substantial doubt about the ability to continue as a going concern.",
                char_start=100,
            )
        ],
        metrics_block=_METRICS_BLOCK,
    )


# --------------------------------------------------------------------------
# The extractor itself
# --------------------------------------------------------------------------


async def test_extract_findings_returns_expected(make_client):
    pack = _short_pack()
    response = json.dumps(
        {
            "findings": [
                {
                    "category": "going_concern",
                    "claim": "The company discloses substantial doubt about continuing as a going concern.",
                    "quoted_span": GOING_CONCERN_SENTENCE,
                    "source_accession": SHORT_ACCESSION,
                    "severity": "critical",
                }
            ],
            "notes": "",
        }
    )
    client = make_client([response])

    result = await extract_findings(client, pack)

    assert isinstance(result, ExtractionResult)
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.category is FindingCategory.GOING_CONCERN
    assert finding.severity is Severity.CRITICAL
    assert finding.source_accession == SHORT_ACCESSION
    assert finding.quoted_span == GOING_CONCERN_SENTENCE

    # The whole point of the verbatim rule: this citation verifies against the
    # exact text the model was shown.
    verification = verify_findings(result.findings, pack)
    assert verification.verified_count == 1
    assert verification.dropped_count == 0


async def test_prompt_carries_metrics_evidence_and_accession(make_client):
    """Inspect what the FakeClient actually received (it records every call)."""
    pack = _short_pack()
    empty = json.dumps({"findings": [], "notes": ""})
    client = make_client([empty])

    await extract_findings(client, pack)

    call = client.calls[0]
    combined = "\n".join(message.content for message in call["messages"])
    assert pack.metrics_block in combined
    assert SHORT_ACCESSION in combined
    assert GOING_CONCERN_SENTENCE in combined
    # The stable system prefix is marked for prompt caching.
    assert call["cache_prefix_upto"] == 0
    # Temperature defaults to 0.0 for a deterministic forensic read.
    assert call["temperature"] == 0.0


async def test_empty_findings_validates_clean_company(make_client):
    pack = _short_pack()
    client = make_client([json.dumps({"findings": [], "notes": "Nothing of concern."})])

    result = await extract_findings(client, pack)

    assert result.findings == []
    assert result.notes == "Nothing of concern."


# --------------------------------------------------------------------------
# The prompt builder as a pure function
# --------------------------------------------------------------------------


def test_build_user_prompt_labels_and_anonymisation():
    prompt = build_user_prompt(_short_pack())

    assert "COMPUTED METRICS (authoritative — do not recompute)" in prompt
    assert _METRICS_BLOCK in prompt
    assert f"=== FILING {SHORT_ACCESSION} ===" in prompt
    # The anonymisation note must be present so the model does not speculate.
    assert "anonymised" in prompt
    assert "[COMPANY]" in prompt


def test_short_filing_is_shown_whole():
    prompt = build_user_prompt(_short_pack())

    # A short filing is included verbatim, so the full sentence is present.
    assert GOING_CONCERN_SENTENCE in prompt


def test_long_filing_falls_back_to_excerpts():
    prompt = build_user_prompt(_long_pack())

    # Excerpt text is shown; the over-cap full text is not.
    assert _EXCERPT_MARKER in prompt
    assert _FULLTEXT_MARKER not in prompt
