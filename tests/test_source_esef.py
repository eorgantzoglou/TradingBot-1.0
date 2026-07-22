"""Tests for scout.data.sources.esef. No network calls -- respx mocks httpx."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from scout.data.http import HttpClient
from scout.data.sources import esef

_FILINGS_URL = f"{esef._API}/filings"

# Two-page JSON:API fixture modelled on a real filings.xbrl.org response
# (verified against the live API): sorted -processed, entity relationships
# resolved via "included", relative json_url/package_url/report_url paths.
#
# Page 1: two filings processed on the target day (2026-07-20).
#   - 3001 is a complete record -> exercises the happy path and json_url
#     preference over package_url/report_url.
#   - 3002 is missing country, period_end, relationships and json_url
#     entirely -> exercises "missing fields degrade to None", including
#     falling back from json_url to package_url.
# Page 2: one filing processed the day before the target -> must NOT be
# yielded, and its presence proves the day-boundary stop actually fires
# rather than the loop just running out of pages.
_PAGE_1 = {
    "data": [
        {
            "type": "filing",
            "id": "3001",
            "attributes": {
                "country": "GB",
                "period_end": "2025-09-30",
                "processed": "2026-07-20 12:00:00.000000",
                "date_added": "2026-07-20 11:00:00.000000",
                "json_url": "/2138007CEIRKZMNI2979/2025-09-30/ESEF/GB/0/2138007CEIRKZMNI2979-2025-09-30-T01.json",
                "package_url": "/2138007CEIRKZMNI2979/2025-09-30/ESEF/GB/0/2138007CEIRKZMNI2979-2025-09-30.zip",
                "report_url": "/2138007CEIRKZMNI2979/2025-09-30/ESEF/GB/0/2138007CEIRKZMNI2979-2025-09-30-T01.xhtml",
                "error_count": 0,
            },
            "relationships": {"entity": {"data": {"type": "entity", "id": "9"}}},
        },
        {
            "type": "filing",
            "id": "3002",
            "attributes": {
                "processed": "2026-07-20 08:00:00.000000",
                "package_url": "/00000000000000000000/2025-06-30/ESEF/XX/0/report.zip",
            },
        },
    ],
    "included": [
        {"type": "entity", "id": "9", "attributes": {"name": "GRAINGER PLC", "identifier": "2138007CEIRKZMNI2979"}}
    ],
    "links": {
        "self": f"{_FILINGS_URL}?page[number]=1",
        "next": f"{_FILINGS_URL}?page[number]=2",
    },
    "meta": {"count": 3},
}

_PAGE_2 = {
    "data": [
        {
            "type": "filing",
            "id": "3003",
            "attributes": {
                "country": "SE",
                "period_end": "2025-12-31",
                "processed": "2026-07-19 23:00:00.000000",
                "json_url": "/OLDLEI0000000000001/2025-12-31/ESEF/SE/0/report.json",
            },
            "relationships": {"entity": {"data": {"type": "entity", "id": "5"}}},
        }
    ],
    "included": [{"type": "entity", "id": "5", "attributes": {"name": "OLD FILER AB", "identifier": "OLDLEI0000000000001"}}],
    "links": {"self": f"{_FILINGS_URL}?page[number]=2", "next": None},
}


def _client() -> HttpClient:
    return HttpClient(user_agent="scout-test/0.1 test@example.com")


def _paged_response(request: httpx.Request) -> httpx.Response:
    page = request.url.params.get("page[number]", "1")
    if page == "1":
        return httpx.Response(200, json=_PAGE_1)
    if page == "2":
        return httpx.Response(200, json=_PAGE_2)
    return httpx.Response(200, json={"data": [], "links": {}})


async def _collect(source: esef.EsefSource, day: date) -> list:
    return [ref async for ref in source.list_documents(day)]


class TestListDocuments:
    @pytest.mark.asyncio
    async def test_pagination_is_followed_and_stops_past_the_target_day(self):
        day = date(2026, 7, 20)
        async with respx.mock:
            route = respx.get(_FILINGS_URL).mock(side_effect=_paged_response)
            async with _client() as http:
                source = esef.EsefSource(http, page_size=2)
                refs = await _collect(source, day)

        assert route.call_count == 2  # page 2 was actually requested
        assert [ref.doc_id for ref in refs] == ["3001", "3002"]  # 3003 (older day) excluded

    @pytest.mark.asyncio
    async def test_complete_record_populates_all_fields_and_prefers_json(self):
        day = date(2026, 7, 20)
        async with respx.mock:
            respx.get(_FILINGS_URL).mock(side_effect=_paged_response)
            async with _client() as http:
                source = esef.EsefSource(http, page_size=2)
                refs = await _collect(source, day)

        ref = next(r for r in refs if r.doc_id == "3001")
        assert ref.source == "esef"
        assert ref.filing_date == date(2025, 9, 30)
        assert ref.title == "GRAINGER PLC"
        assert ref.entity == {"lei": "2138007CEIRKZMNI2979", "country": "GB"}
        assert ref.url == (
            "https://filings.xbrl.org/2138007CEIRKZMNI2979/2025-09-30/ESEF/GB/0/"
            "2138007CEIRKZMNI2979-2025-09-30-T01.json"
        )
        assert ref.meta["download_kind"] == "xbrl-json"

    @pytest.mark.asyncio
    async def test_missing_optional_fields_degrade_to_none_not_raise(self):
        day = date(2026, 7, 20)
        async with respx.mock:
            respx.get(_FILINGS_URL).mock(side_effect=_paged_response)
            async with _client() as http:
                source = esef.EsefSource(http, page_size=2)
                refs = await _collect(source, day)

        ref = next(r for r in refs if r.doc_id == "3002")
        assert ref.filing_date is None  # no period_end
        assert ref.title is None  # no relationships -> no entity name
        assert ref.entity == {}  # no lei, no country
        # Falls back to package_url since json_url is absent for this record.
        assert ref.url == "https://filings.xbrl.org/00000000000000000000/2025-06-30/ESEF/XX/0/report.zip"
        assert ref.meta["download_kind"] == "zip"

    @pytest.mark.asyncio
    async def test_country_filter_is_sent_when_configured(self):
        day = date(2026, 7, 20)
        async with respx.mock:
            respx.get(_FILINGS_URL).mock(side_effect=_paged_response)
            async with _client() as http:
                source = esef.EsefSource(http, country="GB", page_size=2)
                await _collect(source, day)
            request = respx.calls[0].request
            assert request.url.params.get("filter[country]") == "GB"

    @pytest.mark.asyncio
    async def test_unexpected_response_shape_raises(self):
        day = date(2026, 7, 20)
        async with respx.mock:
            respx.get(_FILINGS_URL).mock(return_value=httpx.Response(200, json={"unexpected": True}))
            async with _client() as http:
                source = esef.EsefSource(http)
                with pytest.raises(ValueError, match=r"unexpected filings\.xbrl\.org response shape"):
                    await _collect(source, day)


class TestFetch:
    @pytest.mark.asyncio
    async def test_fetch_downloads_the_preferred_url(self):
        from scout.data.sources.base import DocumentRef

        ref = DocumentRef(
            source="esef",
            doc_id="3001",
            url="https://filings.xbrl.org/2138007CEIRKZMNI2979/2025-09-30/report.json",
            meta={"download_kind": "xbrl-json"},
        )
        payload = b'{"facts": []}'
        async with respx.mock:
            respx.get(ref.url).mock(return_value=httpx.Response(200, content=payload))
            async with _client() as http:
                source = esef.EsefSource(http)
                doc = await source.fetch(ref)

        assert doc.payload == payload
        assert doc.filename == "3001.json"
        assert doc.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_fetch_raises_a_clear_error_when_no_url(self):
        from scout.data.sources.base import DocumentRef

        ref = DocumentRef(source="esef", doc_id="3999")
        async with _client() as http:
            source = esef.EsefSource(http)
            with pytest.raises(ValueError, match="no download URL"):
                await source.fetch(ref)


class TestHelpers:
    def test_pick_download_prefers_json_over_zip_over_html(self):
        url, kind = esef._pick_download(
            {
                "json_url": "/a.json",
                "package_url": "/a.zip",
                "report_url": "/a.xhtml",
            }
        )
        assert kind == "xbrl-json"
        assert url == "https://filings.xbrl.org/a.json"

    def test_pick_download_falls_back_to_zip_then_html_then_none(self):
        assert esef._pick_download({"package_url": "/a.zip"}) == ("https://filings.xbrl.org/a.zip", "zip")
        assert esef._pick_download({"report_url": "/a.xhtml"}) == (
            "https://filings.xbrl.org/a.xhtml",
            "report_html",
        )
        assert esef._pick_download({}) == (None, None)

    def test_parse_date_degrades_to_none_on_garbage(self):
        assert esef._parse_date("2026-07-20") == date(2026, 7, 20)
        assert esef._parse_date("not-a-date") is None
        assert esef._parse_date(None) is None
        assert esef._parse_date(12345) is None
