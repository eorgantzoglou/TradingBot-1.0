"""Tests for the SEC XBRL parser -- offline, against a real archived filing.

The point of parsing from the archive is that it works with no network from the
exact bytes SEC disseminated, so the load-bearing tests use the real 13MB 3M
10-Q sitting in `data/archive`. They're guarded by a skipif so the suite still
runs on a checkout without the archive, but on a machine that has it they MUST
pass -- a green run that silently skipped the only real-data proof would be worse
than a red one. A session-scoped fixture parses the 13MB file once and shares the
result across every real-data test.

The synthetic tests (concept splitting, the no-XBRL path, can_parse) need no
archive and run everywhere.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scout.fundamentals.concepts import PeriodType
from scout.fundamentals.models import RawFact
from scout.fundamentals.parse.base import ParsedFiling
from scout.fundamentals.parse.sec import SecXbrlParser

# The real archived 3M 10-Q. Resolved relative to the repo root (this file lives
# in tests/), so it works regardless of the pytest invocation directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_FILING_PATH = (
    _REPO_ROOT
    / "data"
    / "archive"
    / "sec"
    / "2026"
    / "07"
    / "21"
    / "0000066740-26-000246"
    / "v1"
    / "0000066740-26-000246.txt"
)
_ACCESSION = "0000066740-26-000246"
_ENTITY_HINT = {"cik": "66740"}

_needs_archive = pytest.mark.skipif(
    not _FILING_PATH.exists(),
    reason=f"real SEC archive not present at {_FILING_PATH}",
)


@pytest.fixture(scope="session")
def parsed_3m() -> ParsedFiling:
    """Parse the real 3M 10-Q exactly once for the whole session (it's 13MB)."""
    payload = _FILING_PATH.read_bytes()
    return SecXbrlParser().parse(payload, accession=_ACCESSION, entity_hint=_ENTITY_HINT)


def _facts_named(filing: ParsedFiling, taxonomy: str, local_name: str) -> list[RawFact]:
    return [
        f for f in filing.facts if f.taxonomy == taxonomy and f.local_name == local_name
    ]


# -- real-data tests ---------------------------------------------------------


@_needs_archive
class TestRealFiling:
    def test_headline_identity_and_volume(self, parsed_3m: ParsedFiling) -> None:
        assert parsed_3m.source == "sec"
        assert parsed_3m.entity_id == "66740"
        assert parsed_3m.taxonomy == "us-gaap"
        assert parsed_3m.entity_name is not None and "3M" in parsed_3m.entity_name
        assert parsed_3m.period_of_report == date(2026, 6, 30)
        assert parsed_3m.filing_date == date(2026, 7, 21)
        # 3M's 10-Q reconstructs to ~1279 numeric+text facts; after dropping the
        # ~100 non-numeric ones we still keep well over a thousand.
        assert len(parsed_3m.facts) > 1000
        assert parsed_3m.has_xbrl is True

    def test_revenues_has_a_non_dimensioned_duration_fact(self, parsed_3m: ParsedFiling) -> None:
        revenues = _facts_named(parsed_3m, "us-gaap", "Revenues")
        assert revenues, "expected us-gaap:Revenues facts in a 3M 10-Q"

        consolidated = [
            f
            for f in revenues
            if not f.is_dimensioned and f.period_type is PeriodType.DURATION
        ]
        assert consolidated, "expected at least one consolidated (non-dimensioned) Revenues"

        fact = consolidated[0]
        assert fact.period_start is not None  # durations carry a start
        assert fact.period_end is not None
        assert fact.unit == "USD"
        assert fact.accession == _ACCESSION

    def test_assets_known_instant_value(self, parsed_3m: ParsedFiling) -> None:
        assets = [
            f
            for f in _facts_named(parsed_3m, "us-gaap", "Assets")
            if not f.is_dimensioned and f.period_type is PeriodType.INSTANT
        ]
        # The consolidated total assets at the period end is a known-good number.
        values = {f.value for f in assets}
        assert 34924000000.0 in values

        current = next(f for f in assets if f.value == 34924000000.0)
        assert current.period_type is PeriodType.INSTANT
        assert current.period_start is None  # instants have no start
        assert current.period_end == date(2026, 6, 30)
        assert current.unit == "USD"

    def test_net_income_present(self, parsed_3m: ParsedFiling) -> None:
        net_income = _facts_named(parsed_3m, "us-gaap", "NetIncomeLoss")
        assert net_income, "expected us-gaap:NetIncomeLoss in a 3M 10-Q"
        assert any(f.period_type is PeriodType.DURATION for f in net_income)

    def test_taxonomy_and_local_name_split_correctly(self, parsed_3m: ParsedFiling) -> None:
        # Every fact from this filing carries a prefixed concept, so none should
        # land in the empty-taxonomy fallback, and the prefix must be stripped
        # out of the local name.
        assert all(f.taxonomy for f in parsed_3m.facts)
        assert all(":" not in f.local_name for f in parsed_3m.facts)
        # concept_key round-trips the split.
        rev = _facts_named(parsed_3m, "us-gaap", "Revenues")[0]
        assert rev.concept_key == "us-gaap:Revenues"

    def test_all_kept_facts_are_numeric(self, parsed_3m: ParsedFiling) -> None:
        # Non-numeric facts (text blocks, dates) must have been dropped: every
        # kept RawFact.value is a real float.
        assert all(isinstance(f.value, float) for f in parsed_3m.facts)

    def test_non_numeric_facts_were_skipped_and_warned(self, parsed_3m: ParsedFiling) -> None:
        # The filing contains text facts; the parser must report that it dropped
        # some, so the gap between "facts filed" and "facts kept" is auditable.
        assert any("non-numeric" in w for w in parsed_3m.warnings)

    def test_dimensioned_facts_are_kept(self, parsed_3m: ParsedFiling) -> None:
        # The normalizer needs segment/product breakdowns, so they must survive.
        assert any(f.is_dimensioned for f in parsed_3m.facts)


# -- synthetic tests (no archive needed) -------------------------------------


class TestCanParse:
    @pytest.mark.parametrize("form", ["10-Q", "10-K", "20-F", "40-F", "10-K/A", "10-q"])
    def test_accepts_financial_statement_forms(self, form: str) -> None:
        parser = SecXbrlParser()
        assert parser.can_parse(form_type=form, content_type=None, filename="x.txt") is True

    @pytest.mark.parametrize("form", ["4", "3", "5", "S-1", "424B5", "25", "15-12B", "8-K"])
    def test_rejects_non_fundamental_forms(self, form: str) -> None:
        parser = SecXbrlParser()
        assert parser.can_parse(form_type=form, content_type=None, filename="x.txt") is False

    def test_none_form_is_allowed_for_parse_to_decide(self) -> None:
        parser = SecXbrlParser()
        assert parser.can_parse(form_type=None, content_type=None, filename="x.txt") is True


class TestNoXbrl:
    def test_plain_text_payload_returns_empty_with_warning_and_does_not_raise(self) -> None:
        parser = SecXbrlParser()
        payload = (
            b"<SEC-DOCUMENT>0000000000-00-000000.txt\n"
            b"<SEC-HEADER>plain text filing, no xbrl here</SEC-HEADER>\n"
            b"Just some narrative text, nothing machine-readable.\n"
            b"</SEC-DOCUMENT>"
        )
        result = parser.parse(payload, accession="0000000000-00-000000", entity_hint={"cik": "12345"})

        assert isinstance(result, ParsedFiling)
        assert result.source == "sec"
        assert result.facts == []
        assert result.has_xbrl is False
        # Identity still falls back to the harvest hint even with no XBRL.
        assert result.entity_id == "12345"
        assert result.warnings
        assert any("no XBRL" in w for w in result.warnings)

    def test_garbage_bytes_do_not_raise(self) -> None:
        parser = SecXbrlParser()
        result = parser.parse(b"\x00\x01\x02not a filing at all", accession="acc-1", entity_hint={})
        assert result.facts == []
        assert result.warnings


class TestParserProtocol:
    def test_conforms_to_filing_parser(self) -> None:
        from scout.fundamentals.parse.base import FilingParser

        assert isinstance(SecXbrlParser(), FilingParser)

    def test_source_tag(self) -> None:
        assert SecXbrlParser.source == "sec"
