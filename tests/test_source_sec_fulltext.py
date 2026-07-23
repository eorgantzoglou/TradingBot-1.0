"""Tests for scout.data.sources.sec_fulltext. No network -- respx mocks httpx.

The fixture below is shaped after a live capture from
`https://efts.sec.gov/LATEST/search-index?q=...` (Elasticsearch-style envelope:
`hits.total.value` plus a `hits.hits` list where each hit's `_id` packs the
dashed accession and the primary document into one "<accession>:<doc>" string,
and `_source` carries `ciks`/`display_names`/`root_forms` as arrays). The
one-off odd hits (missing `_source`, empty arrays) are here because the parser's
whole job is to degrade those to empty strings rather than raise.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from scout.data.http import HttpClient
from scout.data.sources import sec_fulltext

# Two well-formed hits plus deliberately awkward ones. The forms differ so the
# `forms` filter and root_forms parsing are both exercised.
_RESPONSE = {
    "took": 12,
    "hits": {
        "total": {"value": 2, "relation": "eq"},
        "hits": [
            {
                "_index": "edgar_file",
                "_id": "0001683168-26-005674:conecti.htm",
                "_source": {
                    "ciks": ["0000790273"],
                    "display_names": ["CONECTISYS CORP (CE)"],
                    "file_date": "2026-01-15",
                    "root_forms": ["10-K"],
                    "file_type": "10-K",
                    "form": "10-K/A",  # root_forms, not this, is what we surface
                },
            },
            {
                "_index": "edgar_file",
                "_id": "0009999999-26-000045:beta8k.htm",
                "_source": {
                    "ciks": ["0009999999"],
                    "display_names": ["BETA HOLDINGS PLC"],
                    "file_date": "2026-02-01",
                    "root_forms": ["8-K"],
                    "file_type": "8-K",
                },
            },
        ],
    },
}


def _client() -> HttpClient:
    return HttpClient(user_agent="scout/0.1 test@example.com")


class TestSearchFilings:
    @pytest.mark.asyncio
    async def test_parses_hits_into_filing_hits(self):
        async with respx.mock:
            respx.get(url__regex=r"https://efts\.sec\.gov/LATEST/search-index.*").mock(
                return_value=httpx.Response(200, json=_RESPONSE)
            )
            async with _client() as http:
                hits = await sec_fulltext.search_filings(http, "artificial intelligence")

        assert len(hits) == 2
        first = hits[0]
        # Accession is the part of `_id` before ':', the doc is the part after.
        assert first.accession == "0001683168-26-005674"
        assert first.doc == "conecti.htm"
        assert first.cik == "0000790273"
        assert first.company == "CONECTISYS CORP (CE)"
        # root_forms wins over the more specific `form` ("10-K/A").
        assert first.form == "10-K"
        assert first.filed == "2026-01-15"

        assert hits[1].accession == "0009999999-26-000045"
        assert hits[1].form == "8-K"

    @pytest.mark.asyncio
    async def test_forms_filter_is_sent_in_request_url(self):
        requested_urls: list[str] = []

        def _capture(request: httpx.Request) -> httpx.Response:
            requested_urls.append(str(request.url))
            return httpx.Response(200, json=_RESPONSE)

        async with respx.mock:
            respx.get(url__regex=r"https://efts\.sec\.gov/.*").mock(side_effect=_capture)
            async with _client() as http:
                await sec_fulltext.search_filings(http, "widgets", forms=["10-K", "8-K"])

        assert len(requested_urls) == 1
        # urlencode renders the comma as %2C; the two forms must both be present.
        assert "forms=10-K%2C8-K" in requested_urls[0]

    @pytest.mark.asyncio
    async def test_no_forms_filter_omits_the_param(self):
        requested_urls: list[str] = []

        def _capture(request: httpx.Request) -> httpx.Response:
            requested_urls.append(str(request.url))
            return httpx.Response(200, json=_RESPONSE)

        async with respx.mock:
            respx.get(url__regex=r"https://efts\.sec\.gov/.*").mock(side_effect=_capture)
            async with _client() as http:
                await sec_fulltext.search_filings(http, "widgets")

        assert len(requested_urls) == 1
        assert "forms=" not in requested_urls[0]
        assert "q=widgets" in requested_urls[0]

    @pytest.mark.asyncio
    async def test_limit_truncates_hits(self):
        async with respx.mock:
            respx.get(url__regex=r"https://efts\.sec\.gov/.*").mock(
                return_value=httpx.Response(200, json=_RESPONSE)
            )
            async with _client() as http:
                hits = await sec_fulltext.search_filings(http, "anything", limit=1)

        assert len(hits) == 1
        assert hits[0].accession == "0001683168-26-005674"

    @pytest.mark.asyncio
    async def test_zero_total_returns_empty(self):
        body = {"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}}
        async with respx.mock:
            respx.get(url__regex=r"https://efts\.sec\.gov/.*").mock(
                return_value=httpx.Response(200, json=body)
            )
            async with _client() as http:
                hits = await sec_fulltext.search_filings(http, "nonexistent phrase")

        assert hits == []

    @pytest.mark.asyncio
    async def test_malformed_body_returns_empty_not_raises(self):
        # A 200 with a shape EDGAR has never sent must degrade to [], not crash
        # the discovery step.
        for body in ([], {"unexpected": "shape"}, {"hits": "not-an-object"}, {"hits": {}}):
            async with respx.mock:
                respx.get(url__regex=r"https://efts\.sec\.gov/.*").mock(
                    return_value=httpx.Response(200, json=body)
                )
                async with _client() as http:
                    hits = await sec_fulltext.search_filings(http, "q")
            assert hits == []

    @pytest.mark.asyncio
    async def test_hit_with_missing_source_and_empty_arrays_degrades_to_empty_strings(self):
        body = {
            "hits": {
                "total": {"value": 2, "relation": "eq"},
                "hits": [
                    # No `_source` at all -- everything but the id must be "".
                    {"_id": "0001111111-26-000001:only.htm"},
                    # Empty arrays and a non-string id part.
                    {
                        "_id": "0002222222-26-000002:",
                        "_source": {"ciks": [], "display_names": [], "root_forms": []},
                    },
                ],
            }
        }
        async with respx.mock:
            respx.get(url__regex=r"https://efts\.sec\.gov/.*").mock(
                return_value=httpx.Response(200, json=body)
            )
            async with _client() as http:
                hits = await sec_fulltext.search_filings(http, "q")

        assert len(hits) == 2
        assert hits[0].accession == "0001111111-26-000001"
        assert hits[0].doc == "only.htm"
        assert hits[0].cik == ""
        assert hits[0].company == ""
        assert hits[0].form == ""
        assert hits[0].filed == ""

        assert hits[1].accession == "0002222222-26-000002"
        assert hits[1].doc == ""
        assert hits[1].cik == ""
