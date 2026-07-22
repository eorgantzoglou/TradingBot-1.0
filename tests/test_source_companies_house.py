from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx
from httpx import Response

from scout.data.http import HttpClient
from scout.data.sources.base import DocumentRef
from scout.data.sources.companies_house import (
    CompaniesHouseSource,
    fetch_company_profile,
    fetch_filing_history,
)

USER_AGENT = "scout-test test@example.com"
BULK_URL = "http://download.companieshouse.gov.uk/Accounts_Bulk_Data-2026-07-21.zip"


def test_available_is_always_true_no_key_needed():
    http = HttpClient(user_agent=USER_AGENT)
    # The bulk path needs no credentials at all -- available() must not gate on them.
    assert CompaniesHouseSource(http).available() is True


@respx.mock
async def test_list_documents_normal_day():
    http = HttpClient(user_agent=USER_AGENT)
    source = CompaniesHouseSource(http)
    route = respx.head(BULK_URL).mock(return_value=Response(200))

    refs = [ref async for ref in source.list_documents(date(2026, 7, 21))]

    assert route.called
    assert len(refs) == 1
    ref = refs[0]
    assert ref.doc_id == "accounts-bulk-2026-07-21"
    assert ref.filing_date == date(2026, 7, 21)
    assert ref.url == BULK_URL
    await http.aclose()


@respx.mock
async def test_list_documents_missing_file_is_not_an_error():
    """Sunday/Monday/holiday gaps are normal -- a 404 must not raise."""
    http = HttpClient(user_agent=USER_AGENT)
    source = CompaniesHouseSource(http)
    respx.head("http://download.companieshouse.gov.uk/Accounts_Bulk_Data-2026-07-19.zip").mock(
        return_value=Response(404)
    )

    refs = [ref async for ref in source.list_documents(date(2026, 7, 19))]

    assert refs == []
    await http.aclose()


@respx.mock
async def test_list_documents_server_error_raises():
    """A genuine outage is not the same thing as a normal missing-day 404."""
    http = HttpClient(user_agent=USER_AGENT)
    source = CompaniesHouseSource(http)
    respx.head(BULK_URL).mock(return_value=Response(503))

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in source.list_documents(date(2026, 7, 21)):
            pass
    await http.aclose()


@respx.mock
async def test_fetch_returns_zip_payload():
    http = HttpClient(user_agent=USER_AGENT)
    source = CompaniesHouseSource(http)
    ref = DocumentRef(
        source="companies_house",
        doc_id="accounts-bulk-2026-07-21",
        url=BULK_URL,
        meta={"filename": "Accounts_Bulk_Data-2026-07-21.zip"},
    )
    respx.get(BULK_URL).mock(return_value=Response(200, content=b"bulk-zip-bytes"))

    doc = await source.fetch(ref)

    assert doc.payload == b"bulk-zip-bytes"
    assert doc.filename == "Accounts_Bulk_Data-2026-07-21.zip"
    assert doc.content_type == "application/zip"
    await http.aclose()


@respx.mock
async def test_fetch_company_profile_uses_basic_auth():
    http = HttpClient(user_agent=USER_AGENT)
    route = respx.get("https://api.company-information.service.gov.uk/company/00000006").mock(
        return_value=Response(200, json={"company_number": "00000006", "company_name": "Test"})
    )

    profile = await fetch_company_profile(http, "test-key", "00000006")

    assert route.called
    request = route.calls.last.request
    assert request.headers["Authorization"].startswith("Basic ")
    assert profile["company_number"] == "00000006"
    await http.aclose()


@respx.mock
async def test_fetch_filing_history_uses_basic_auth():
    http = HttpClient(user_agent=USER_AGENT)
    route = respx.get(
        "https://api.company-information.service.gov.uk/company/00000006/filing-history"
    ).mock(return_value=Response(200, json={"items": []}))

    history = await fetch_filing_history(http, "test-key", "00000006")

    assert route.called
    assert history["items"] == []
    await http.aclose()
