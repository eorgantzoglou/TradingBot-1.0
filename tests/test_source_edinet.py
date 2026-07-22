from __future__ import annotations

from datetime import date

import pytest
import respx
from httpx import Response

from scout.config import SourceCredentials
from scout.data.http import HttpClient
from scout.data.sources.base import DocumentRef
from scout.data.sources.edinet import EdinetSource

USER_AGENT = "scout-test test@example.com"
LIST_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"


def _creds(key: str | None = "edinet-test-key") -> SourceCredentials:
    return SourceCredentials(edinet_key=key)


def test_available_true_with_key():
    http = HttpClient(user_agent=USER_AGENT)
    assert EdinetSource(http, _creds()).available() is True


def test_available_false_without_key():
    http = HttpClient(user_agent=USER_AGENT)
    assert EdinetSource(http, _creds(None)).available() is False


async def test_list_documents_without_key_raises_not_empty():
    # Missing credentials must surface as a clear error, never a silent empty
    # list -- otherwise a broken key and a genuinely empty day are indistinguishable.
    http = HttpClient(user_agent=USER_AGENT)
    source = EdinetSource(http, _creds(None))
    with pytest.raises(RuntimeError, match="edinet_key"):
        async for _ in source.list_documents(date(2026, 7, 21)):
            pass
    await http.aclose()


@respx.mock
async def test_list_documents_normal_day_filters_withdrawn_and_missing_xbrl():
    http = HttpClient(user_agent=USER_AGENT)
    source = EdinetSource(http, _creds())
    payload = {
        "metadata": {"status": "200", "message": "OK", "resultset": {"count": 3}},
        "results": [
            {
                "docID": "S100ABCD",
                "edinetCode": "E00001",
                "secCode": "12345",
                "JCN": "1234567890123",
                "filerName": "Test Corp",
                "docTypeCode": "120",
                "submitDateTime": "2026-07-21 15:00",
                "docDescription": "Annual securities report",
                "withdrawalStatus": "0",
                "xbrlFlag": "1",
                "csvFlag": "1",
            },
            {
                # withdrawn -- must be filtered out
                "docID": "S100WITHDRAWN",
                "withdrawalStatus": "1",
                "xbrlFlag": "1",
                "csvFlag": "1",
            },
            {
                # no XBRL and no CSV available -- nothing to fetch, must be filtered out
                "docID": "S100NODATA",
                "withdrawalStatus": "0",
                "xbrlFlag": "0",
                "csvFlag": "0",
            },
        ],
    }
    route = respx.get(LIST_URL).mock(return_value=Response(200, json=payload))

    refs = [ref async for ref in source.list_documents(date(2026, 7, 21))]

    assert route.called
    request = route.calls.last.request
    assert request.url.params["date"] == "2026-07-21"
    assert request.url.params["type"] == "2"
    assert request.url.params["Subscription-Key"] == "edinet-test-key"

    assert len(refs) == 1
    ref = refs[0]
    assert ref.doc_id == "S100ABCD"
    assert ref.filing_date == date(2026, 7, 21)
    assert ref.form_type == "120"
    assert ref.entity == {"edinet_code": "E00001", "sec_code": "12345", "jcn": "1234567890123"}
    assert ref.meta["csv_flag"] == "1"
    # No key ever leaks into a stored ref -- it must never reach the archive's manifest.
    assert "Subscription-Key" not in (ref.url or "")
    await http.aclose()


@respx.mock
async def test_list_documents_empty_day():
    http = HttpClient(user_agent=USER_AGENT)
    source = EdinetSource(http, _creds())
    payload = {"metadata": {"status": "200", "message": "OK", "resultset": {"count": 0}}, "results": []}
    respx.get(LIST_URL).mock(return_value=Response(200, json=payload))

    refs = [ref async for ref in source.list_documents(date(2026, 7, 25))]

    assert refs == []
    await http.aclose()


@respx.mock
async def test_list_documents_raises_on_error_status():
    http = HttpClient(user_agent=USER_AGENT)
    source = EdinetSource(http, _creds())
    payload = {"metadata": {"status": "400", "message": "invalid parameter"}, "results": []}
    respx.get(LIST_URL).mock(return_value=Response(200, json=payload))

    with pytest.raises(RuntimeError, match="400"):
        async for _ in source.list_documents(date(2026, 7, 21)):
            pass
    await http.aclose()


@respx.mock
async def test_fetch_defaults_to_type5_csv():
    http = HttpClient(user_agent=USER_AGENT)
    source = EdinetSource(http, _creds())
    ref = DocumentRef(source="edinet", doc_id="S100ABCD", meta={"csv_flag": "1"})
    route = respx.get("https://api.edinet-fsa.go.jp/api/v2/documents/S100ABCD").mock(
        return_value=Response(200, content=b"csv-zip-bytes", headers={"content-type": "application/octet-stream"})
    )

    doc = await source.fetch(ref)

    assert route.called
    assert route.calls.last.request.url.params["type"] == "5"
    assert doc.payload == b"csv-zip-bytes"
    assert doc.filename == "S100ABCD_type5.zip"
    await http.aclose()


@respx.mock
async def test_fetch_falls_back_to_type1_when_csv_unavailable():
    http = HttpClient(user_agent=USER_AGENT)
    source = EdinetSource(http, _creds())
    ref = DocumentRef(source="edinet", doc_id="S100OLD", meta={"csv_flag": "0"})
    route = respx.get("https://api.edinet-fsa.go.jp/api/v2/documents/S100OLD").mock(
        return_value=Response(200, content=b"xbrl-zip-bytes")
    )

    doc = await source.fetch(ref)

    assert route.calls.last.request.url.params["type"] == "1"
    assert doc.payload == b"xbrl-zip-bytes"
    await http.aclose()


async def test_fetch_without_key_raises():
    http = HttpClient(user_agent=USER_AGENT)
    source = EdinetSource(http, _creds(None))
    ref = DocumentRef(source="edinet", doc_id="S100ABCD")
    with pytest.raises(RuntimeError, match="edinet_key"):
        await source.fetch(ref)
    await http.aclose()
