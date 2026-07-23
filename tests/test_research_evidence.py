"""Evidence extraction and citation verification, against real archived filings.

These two modules carry no LLM, so they are golden-tested against real bytes:
the ConectiSys microcap 10-Q, which contains a genuine going-concern disclosure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scout.research.evidence import (
    _anonymize,
    _html_to_text,
    build_evidence_pack,
    extract_primary_text,
)
from scout.research.models import Finding, FindingCategory, Severity
from scout.research.verify import verify_finding, verify_findings

ARCHIVE = Path("data/archive")


def _conectisys_bytes() -> tuple[str, bytes] | None:
    manifest = ARCHIVE / "manifest.jsonl"
    if not manifest.exists():
        return None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["source"] == "sec" and row["doc_id"].startswith("0001683168"):
            return row["doc_id"], (ARCHIVE / row["path"]).read_bytes()
    return None


@pytest.fixture(scope="module")
def conectisys():
    data = _conectisys_bytes()
    if data is None:
        pytest.skip("ConectiSys filing not in archive")
    return data


# --------------------------------------------------------------------- unit


def test_anonymize_respects_word_boundaries():
    # The regression: ticker "CONC" must NOT redact the "conc" inside "concern".
    text = "the company as a going concern; CONC traded flat"
    out = _anonymize(text, {"[COMPANY]": "ACME", "[TICKER]": "CONC"})
    assert "concern" in out
    assert "[TICKER]" in out
    assert " CONC " not in out


def test_html_to_text_strips_tags_and_entities():
    html = "<html><body><p>Going&#160;concern &amp; doubt</p><script>x=1</script></body></html>"
    text = _html_to_text(html)
    assert text == "Going concern & doubt"
    assert "script" not in text  # <script> body removed, not just tags


def test_extract_primary_text_empty_bytes_no_crash():
    assert extract_primary_text(b"") == ""


# --------------------------------------------------------------- golden (real)


def test_extract_real_narrative(conectisys):
    _accession, raw = conectisys
    text = extract_primary_text(raw)
    assert len(text) > 5000
    assert "going concern" in text.lower()


def test_build_pack_selects_red_flags_and_anonymizes(conectisys):
    accession, raw = conectisys
    pack = build_evidence_pack(
        "790273", [(accession, raw)], company_name="CONECTISYS", tickers=("CONC",)
    )
    assert pack.has_narrative
    assert accession in pack.texts_by_accession
    # Real going-concern disclosure is present and selected as an excerpt.
    categories = {e.category for e in pack.excerpts}
    assert "going_concern" in categories
    # Anonymised: the company name is gone, "concern" survives intact.
    full = pack.texts_by_accession[accession]
    assert "CONECTISYS" not in full.upper()
    assert "concern" in full.lower()


def test_real_quote_verifies_fabrication_dropped(conectisys):
    accession, raw = conectisys
    pack = build_evidence_pack("790273", [(accession, raw)], company_name="CONECTISYS")
    gc_excerpt = next(e for e in pack.excerpts if e.category == "going_concern")

    real = Finding(
        category=FindingCategory.GOING_CONCERN,
        claim="Going concern doubt.",
        quoted_span=gc_excerpt.text[:60],
        source_accession=accession,
        severity=Severity.CRITICAL,
    )
    ok, _ = verify_finding(real, pack)
    assert ok

    fake = Finding(
        category=FindingCategory.DILUTION,
        claim="Issued a billion toxic shares.",
        quoted_span="the company issued one billion toxic death-spiral shares overnight",
        source_accession=accession,
        severity=Severity.CRITICAL,
    )
    ok2, reason = verify_finding(fake, pack)
    assert not ok2
    assert "not found" in reason


# --------------------------------------------------------------- verify unit


def _pack_with_text(accession: str, text: str):
    from scout.research.evidence import EvidencePack

    pack = EvidencePack(entity_id="x")
    pack.texts_by_accession[accession] = text
    return pack


def test_verify_whitespace_normalisation():
    pack = _pack_with_text("acc-1", "the  company   reported   substantial doubt today")
    finding = Finding(
        category=FindingCategory.GOING_CONCERN, claim="doubt",
        quoted_span="reported substantial doubt", source_accession="acc-1", severity=Severity.HIGH,
    )
    ok, reason = verify_finding(finding, pack)
    assert ok
    assert "normal" in reason


def test_verify_rejects_short_quote():
    pack = _pack_with_text("acc-1", "in default on its notes")
    finding = Finding(
        category=FindingCategory.OTHER, claim="x", quoted_span="default",
        source_accession="acc-1", severity=Severity.LOW,
    )
    ok, reason = verify_finding(finding, pack)
    assert not ok
    assert "too short" in reason


def test_verify_wrong_accession_flagged():
    pack = _pack_with_text("acc-1", "there is substantial doubt about the entity")
    finding = Finding(
        category=FindingCategory.GOING_CONCERN, claim="doubt",
        quoted_span="substantial doubt about the entity", source_accession="acc-2",
        severity=Severity.HIGH,
    )
    ok, reason = verify_finding(finding, pack)
    assert not ok
    assert "wrong" in reason or "not in" in reason


def test_verify_findings_partitions():
    pack = _pack_with_text("acc-1", "the registrant disclosed a going concern qualification")
    good = Finding(category=FindingCategory.GOING_CONCERN, claim="a",
                   quoted_span="disclosed a going concern qualification",
                   source_accession="acc-1", severity=Severity.HIGH)
    bad = Finding(category=FindingCategory.DILUTION, claim="b",
                  quoted_span="issued five hundred million dilutive warrants",
                  source_accession="acc-1", severity=Severity.HIGH)
    result = verify_findings([good, bad], pack)
    assert result.verified_count == 1
    assert result.dropped_count == 1
    assert result.fabrication_rate == 0.5
