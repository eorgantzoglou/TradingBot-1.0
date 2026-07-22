"""Tests for scout.data.sources.sec. No network calls -- respx mocks httpx.

The fixture below is captured from a real EDGAR daily index (see
https://www.sec.gov/Archives/edgar/daily-index/2026/QTR3/form.20260721.idx),
not hand-rolled, because two real bugs (a missing "edgar/" URL segment and a
dashed-vs-undashed Date Filed column) shipped past a fabricated fixture that
didn't share those quirks and were only caught by a live smoke test. Building
it line-by-line from the captured shape keeps this file the thing that would
have caught them.
"""

from __future__ import annotations

import logging
from datetime import date

import httpx
import pytest
import respx

from scout.data.http import HttpClient
from scout.data.sources import sec

# Real daily-index rows are laid out with 2+ spaces between columns (never a
# fixed width -- company-name length varies), the Date Filed column is
# undashed YYYYMMDD, and rows carry significant trailing whitespace after the
# File Name column. Form types can themselves contain spaces ("1-A POS",
# "NT 10-K", "X-17A-5" has none but is included as a form NOT in
# DEFAULT_FORM_TYPES), which is exactly where a naive split-on-whitespace
# parser breaks. Built as a list of lines (rather than one triple-quoted
# block) so the trailing whitespace on data rows survives linting -- it lives
# inside the string literal, not at the end of a physical source line.
_IDX_LINES = [
    # -- 4-line description block, verbatim shape SEC always ships --
    "Description:           Daily Index of EDGAR Dissemination Feed by Form Type",
    "Last Data Received:    Jul 21, 2026",
    "Comments:              webmaster@sec.gov",
    "Anonymous FTP:         ftp://ftp.sec.gov/edgar/",
    # -- blank lines before the header --
    "",
    "",
    "",
    "",
    # -- header WRAPS across two lines --
    "Form Type   Company Name                                                  CIK",
    "      Date Filed  File Name",
    # -- dashed separator, on the line after BOTH header lines --
    "-" * 143,
    # -- data rows --
    # Not in DEFAULT_FORM_TYPES; plain form type, no space.
    "1-A              OBSIDIAN PRIME INC                                            2011021     20260721    edgar/data/2011021/0002011021-26-000007.txt                                                ",
    # Not in DEFAULT_FORM_TYPES; form type contains a space -- proves the
    # 2+-space column separator survives a form type that itself has a
    # single embedded space.
    "1-A POS          Energea Portfolio 2 LP                                        1811470     20260721    edgar/data/1811470/0001811470-26-000014.txt                                                ",
    # Not in DEFAULT_FORM_TYPES; proves filtering excludes real non-target
    # forms, not just ones convenient to fabricate.
    "X-17A-5          MIZUHO SECURITIES USA LLC                                     812291      20260721    edgar/data/812291/0000812291-26-000011.txt                                                ",
    # Genuinely malformed: no column structure at all.
    "this line is just garbage and does not match the columns at all",
    # In DEFAULT_FORM_TYPES.
    "10-K             GLOBAL WIDGETS CORP                                           1234567     20260721    edgar/data/1234567/0001234567-26-000012.txt                                                ",
    # In DEFAULT_FORM_TYPES; form type contains a space -- proves a kept
    # form still parses correctly when its type has an embedded space.
    "NT 10-K          LATE FILER HOLDINGS INC                                       7654321     20260721    edgar/data/7654321/0007654321-26-000005.txt                                                ",
    # In DEFAULT_FORM_TYPES.
    "4                DELTA INSIDER JANE                                            3334444     20260721    edgar/data/3334444/0003334444-26-000099.txt                                                ",
    # In DEFAULT_FORM_TYPES.
    "8-K              BETA HOLDINGS PLC                                             9999999     20260721    edgar/data/9999999/0009999999-26-000045.txt                                                ",
]
_IDX_FIXTURE = "\n".join(_IDX_LINES) + "\n"

_DAY = date(2026, 7, 21)  # a Tuesday -- the day the fixture above was captured for


def _client() -> HttpClient:
    return HttpClient(user_agent="scout-test/0.1 test@example.com")


def _idx_url(day: date) -> str:
    # Mirrors sec.py's own URL construction exactly. Do not "simplify" this
    # back to a hand-written string with the "edgar/" segment left out --
    # that omission is precisely the bug this suite exists to catch (see
    # TestDailyIndexUrl below), and a hand-rolled duplicate here would let it
    # regress silently the next time both copies drift out of sync.
    quarter = (day.month - 1) // 3 + 1
    return f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{quarter}/form.{day:%Y%m%d}.idx"


async def _collect(source: sec.SecSource, day: date) -> list:
    return [ref async for ref in source.list_documents(day)]


class TestNormalizeAccession:
    def test_already_dashed(self):
        assert sec.normalize_accession("0001234567-26-000012") == "0001234567-26-000012"

    def test_undashed_gets_dashes_inserted(self):
        assert sec.normalize_accession("000123456726000012") == "0001234567-26-000012"

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="not a recognizable SEC accession number"):
            sec.normalize_accession("not-an-accession")


class TestDailyIndexUrl:
    """Regression test for the missing-'edgar/'-segment bug: it produced a
    404 on every weekday that was indistinguishable from a quiet weekend
    until a live smoke test caught it."""

    @pytest.mark.asyncio
    async def test_requested_url_is_under_archives_edgar_daily_index(self):
        requested_urls: list[str] = []

        def _capture(request: httpx.Request) -> httpx.Response:
            requested_urls.append(str(request.url))
            return httpx.Response(200, content=_IDX_FIXTURE.encode("latin-1"))

        async with respx.mock:
            respx.get(url__regex=r".*/form\.\d{8}\.idx$").mock(side_effect=_capture)
            async with _client() as http:
                source = sec.SecSource(http)
                await _collect(source, _DAY)

        assert len(requested_urls) == 1
        assert "/Archives/edgar/daily-index/" in requested_urls[0]


class TestWeekdayVsWeekend404:
    """A 404 is a normal empty day on weekends/holidays, but on a weekday it
    means our URL or SEC's layout is wrong -- the two must not look the
    same, or the failure mode is silent and the history is unrecoverable."""

    @pytest.mark.asyncio
    async def test_weekday_404_logs_warning_and_returns_no_refs(self, caplog):
        day = date(2026, 7, 21)  # Tuesday
        async with respx.mock:
            respx.get(_idx_url(day)).mock(return_value=httpx.Response(404))
            async with _client() as http:
                source = sec.SecSource(http)
                with caplog.at_level(logging.WARNING, logger="scout.data.sources.sec"):
                    refs = await _collect(source, day)

        assert refs == []
        assert any("no daily index" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_weekend_404_does_not_warn(self, caplog):
        day = date(2026, 7, 18)  # Saturday
        assert day.weekday() >= 5
        async with respx.mock:
            respx.get(_idx_url(day)).mock(return_value=httpx.Response(404))
            async with _client() as http:
                source = sec.SecSource(http)
                with caplog.at_level(logging.WARNING, logger="scout.data.sources.sec"):
                    refs = await _collect(source, day)

        assert refs == []
        assert caplog.records == []


class TestListDocuments:
    @pytest.mark.asyncio
    async def test_empty_day_returns_no_refs_and_does_not_raise(self):
        day = date(2026, 7, 18)  # a Saturday
        async with respx.mock:
            respx.get(_idx_url(day)).mock(return_value=httpx.Response(404))
            async with _client() as http:
                source = sec.SecSource(http)
                refs = await _collect(source, day)
        assert refs == []

    @pytest.mark.asyncio
    async def test_form_type_filtering(self):
        async with respx.mock:
            respx.get(_idx_url(_DAY)).mock(
                return_value=httpx.Response(200, content=_IDX_FIXTURE.encode("latin-1"))
            )
            async with _client() as http:
                source = sec.SecSource(http)
                refs = await _collect(source, _DAY)

        form_types = {ref.form_type for ref in refs}
        # 1-A, 1-A POS and X-17A-5 are not in DEFAULT_FORM_TYPES and must be
        # filtered out; the space-containing form types must still parse.
        assert "1-A" not in form_types
        assert "1-A POS" not in form_types
        assert "X-17A-5" not in form_types
        assert form_types == {"10-K", "NT 10-K", "4", "8-K"}
        assert len(refs) == 4

    @pytest.mark.asyncio
    async def test_custom_form_types_restricts_further(self):
        async with respx.mock:
            respx.get(_idx_url(_DAY)).mock(
                return_value=httpx.Response(200, content=_IDX_FIXTURE.encode("latin-1"))
            )
            async with _client() as http:
                source = sec.SecSource(http, form_types=frozenset({"10-K"}))
                refs = await _collect(source, _DAY)

        assert [ref.form_type for ref in refs] == ["10-K"]

    @pytest.mark.asyncio
    async def test_malformed_line_is_skipped_and_surfaced_not_raised(self, caplog):
        async with respx.mock:
            respx.get(_idx_url(_DAY)).mock(
                return_value=httpx.Response(200, content=_IDX_FIXTURE.encode("latin-1"))
            )
            async with _client() as http:
                source = sec.SecSource(http)
                with caplog.at_level(logging.WARNING, logger="scout.data.sources.sec"):
                    refs = await _collect(source, _DAY)

        # 7 of the 8 data rows are well-formed despite the garbage line
        # between them -- one bad line does not take the rest of the day
        # down. 4 remain after form-type filtering (10-K, NT 10-K, 4, 8-K).
        assert len(refs) == 4
        assert any("skipped 1 malformed" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_undashed_date_filed_parses_correctly(self):
        async with respx.mock:
            respx.get(_idx_url(_DAY)).mock(
                return_value=httpx.Response(200, content=_IDX_FIXTURE.encode("latin-1"))
            )
            async with _client() as http:
                source = sec.SecSource(http)
                refs = await _collect(source, _DAY)

        assert refs  # sanity: filtering above didn't eat everything
        assert all(ref.filing_date == date(2026, 7, 21) for ref in refs)

    @pytest.mark.asyncio
    async def test_doc_id_and_fields_populated_from_idx_row(self):
        async with respx.mock:
            respx.get(_idx_url(_DAY)).mock(
                return_value=httpx.Response(200, content=_IDX_FIXTURE.encode("latin-1"))
            )
            async with _client() as http:
                source = sec.SecSource(http)
                refs = await _collect(source, _DAY)

        by_form = {ref.form_type: ref for ref in refs}
        ten_k = by_form["10-K"]
        assert ten_k.source == "sec"
        assert ten_k.doc_id == "0001234567-26-000012"
        assert ten_k.filing_date == date(2026, 7, 21)
        assert ten_k.title == "GLOBAL WIDGETS CORP"
        assert ten_k.entity == {"cik": "1234567"}
        assert ten_k.url == "https://www.sec.gov/Archives/edgar/data/1234567/0001234567-26-000012.txt"

        # The space-containing form type must parse just as cleanly.
        nt_10k = by_form["NT 10-K"]
        assert nt_10k.title == "LATE FILER HOLDINGS INC"
        assert nt_10k.entity == {"cik": "7654321"}

    @pytest.mark.asyncio
    async def test_unrecognizable_header_raises_rather_than_returning_empty(self):
        # A day where the format has changed underneath us must not be
        # indistinguishable from a day with no filings.
        async with respx.mock:
            respx.get(_idx_url(_DAY)).mock(
                return_value=httpx.Response(200, content=b"nothing recognizable here at all\n")
            )
            async with _client() as http:
                source = sec.SecSource(http)
                with pytest.raises(ValueError, match="no recognizable header"):
                    await _collect(source, _DAY)


class TestFetch:
    @pytest.mark.asyncio
    async def test_fetch_uses_ref_url_and_returns_verbatim_bytes(self):
        from scout.data.sources.base import DocumentRef

        ref = DocumentRef(
            source="sec",
            doc_id="0001234567-26-000012",
            filing_date=date(2026, 7, 21),
            form_type="10-K",
            title="GLOBAL WIDGETS CORP",
            url="https://www.sec.gov/Archives/edgar/data/1234567/0001234567-26-000012.txt",
            entity={"cik": "1234567"},
        )
        payload = b"SEC-DOCUMENT>filing text here\n"
        async with respx.mock:
            respx.get(ref.url).mock(return_value=httpx.Response(200, content=payload))
            async with _client() as http:
                source = sec.SecSource(http)
                doc = await source.fetch(ref)

        assert doc.payload == payload
        assert doc.filename == "0001234567-26-000012.txt"
        assert doc.ref is ref

    @pytest.mark.asyncio
    async def test_fetch_falls_back_to_constructed_url_when_ref_has_none(self):
        from scout.data.sources.base import DocumentRef

        ref = DocumentRef(
            source="sec",
            doc_id="0001234567-26-000012",
            entity={"cik": "1234567"},
        )
        expected_url = (
            "https://www.sec.gov/Archives/edgar/data/1234567/"
            "000123456726000012/0001234567-26-000012.txt"
        )
        payload = b"fallback content"
        async with respx.mock:
            respx.get(expected_url).mock(return_value=httpx.Response(200, content=payload))
            async with _client() as http:
                source = sec.SecSource(http)
                doc = await source.fetch(ref)

        assert doc.payload == payload


class TestFetchCompanyTickers:
    @pytest.mark.asyncio
    async def test_flattens_the_index_keyed_object(self):
        body = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 1234567, "ticker": "ACME", "title": "Acme Widgets Inc"},
        }
        async with respx.mock:
            respx.get("https://www.sec.gov/files/company_tickers.json").mock(
                return_value=httpx.Response(200, json=body)
            )
            async with _client() as http:
                rows = await sec.fetch_company_tickers(http)

        assert rows == [
            {"cik": "320193", "ticker": "AAPL", "title": "Apple Inc."},
            {"cik": "1234567", "ticker": "ACME", "title": "Acme Widgets Inc"},
        ]
