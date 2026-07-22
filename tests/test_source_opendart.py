from __future__ import annotations

from datetime import date

import pytest
import respx
from httpx import Response

from scout.config import SourceCredentials
from scout.data.http import HttpClient
from scout.data.sources.base import DocumentRef
from scout.data.sources.opendart import OpenDartSource

USER_AGENT = "scout-test test@example.com"
LIST_URL = "https://opendart.fss.or.kr/api/list.json"
DOC_URL = "https://opendart.fss.or.kr/api/document.xml"


def _creds(key: str | None = "opendart-test-key") -> SourceCredentials:
    return SourceCredentials(opendart_key=key)


def _list_body(*, status: str, message: str, items: list[dict], page_no: int = 1, total_page: int = 1) -> dict:
    return {
        "status": status,
        "message": message,
        "page_no": page_no,
        "page_count": 100,
        "total_count": len(items),
        "total_page": total_page,
        "list": items,
    }


def test_available_true_with_key():
    http = HttpClient(user_agent=USER_AGENT)
    assert OpenDartSource(http, _creds()).available() is True


def test_available_false_without_key():
    http = HttpClient(user_agent=USER_AGENT)
    assert OpenDartSource(http, _creds(None)).available() is False


@respx.mock
async def test_list_documents_normal_day():
    http = HttpClient(user_agent=USER_AGENT)
    source = OpenDartSource(http, _creds())
    body = _list_body(
        status="000",
        message="정상",
        items=[
            {
                "corp_code": "00126380",
                "corp_name": "Test Corp",
                "stock_code": "005930",
                "corp_cls": "Y",
                "report_nm": "사업보고서",
                "rcept_no": "20260721000123",
                "flr_nm": "Test Corp",
                "rcept_dt": "20260721",
                "rm": "",
            }
        ],
    )
    route = respx.get(LIST_URL).mock(return_value=Response(200, json=body))

    refs = [ref async for ref in source.list_documents(date(2026, 7, 21))]

    assert route.called
    request = route.calls.last.request
    assert request.url.params["bgn_de"] == "20260721"
    assert request.url.params["end_de"] == "20260721"
    assert request.url.params["crtfc_key"] == "opendart-test-key"

    assert len(refs) == 1
    ref = refs[0]
    assert ref.doc_id == "20260721000123"
    assert ref.filing_date == date(2026, 7, 21)
    assert ref.entity == {"corp_code": "00126380", "stock_code": "005930", "corp_cls": "Y"}
    # No key ever leaks into a stored ref -- it must never reach the archive's manifest.
    assert "crtfc_key" not in (ref.url or "")
    await http.aclose()


@respx.mock
async def test_list_documents_status_013_is_empty_not_error():
    """The whole point: '013' (no data) must yield an empty list, not raise."""
    http = HttpClient(user_agent=USER_AGENT)
    source = OpenDartSource(http, _creds())
    body = {"status": "013", "message": "조회된 데이타가 없습니다"}
    respx.get(LIST_URL).mock(return_value=Response(200, json=body))

    refs = [ref async for ref in source.list_documents(date(2026, 7, 26))]

    assert refs == []
    await http.aclose()


@respx.mock
async def test_list_documents_status_010_raises_clear_error():
    """An unregistered key must NOT look like an empty day."""
    http = HttpClient(user_agent=USER_AGENT)
    source = OpenDartSource(http, _creds())
    body = {"status": "010", "message": "등록되지 않은 키 입니다."}
    respx.get(LIST_URL).mock(return_value=Response(200, json=body))

    with pytest.raises(RuntimeError, match="010"):
        async for _ in source.list_documents(date(2026, 7, 21)):
            pass
    await http.aclose()


@respx.mock
async def test_list_documents_status_020_rate_limit_raises():
    http = HttpClient(user_agent=USER_AGENT)
    source = OpenDartSource(http, _creds())
    body = {"status": "020", "message": "요청 제한을 초과하였습니다."}
    respx.get(LIST_URL).mock(return_value=Response(200, json=body))

    with pytest.raises(RuntimeError, match="020"):
        async for _ in source.list_documents(date(2026, 7, 21)):
            pass
    await http.aclose()


@respx.mock
async def test_list_documents_follows_pagination():
    http = HttpClient(user_agent=USER_AGENT)
    source = OpenDartSource(http, _creds())

    page_1 = _list_body(
        status="000",
        message="정상",
        items=[{"corp_code": "1", "rcept_no": "A1", "rcept_dt": "20260721"}],
        page_no=1,
        total_page=2,
    )
    page_2 = _list_body(
        status="000",
        message="정상",
        items=[{"corp_code": "2", "rcept_no": "A2", "rcept_dt": "20260721"}],
        page_no=2,
        total_page=2,
    )

    def _responder(request):
        page_no = request.url.params["page_no"]
        return Response(200, json=page_1 if page_no == "1" else page_2)

    respx.get(LIST_URL).mock(side_effect=_responder)

    refs = [ref async for ref in source.list_documents(date(2026, 7, 21))]

    assert [ref.doc_id for ref in refs] == ["A1", "A2"]
    await http.aclose()


@respx.mock
async def test_fetch_returns_zip_payload():
    http = HttpClient(user_agent=USER_AGENT)
    source = OpenDartSource(http, _creds())
    ref = DocumentRef(source="opendart", doc_id="20260721000123")
    route = respx.get(DOC_URL).mock(
        return_value=Response(200, content=b"zip-bytes", headers={"content-type": "application/x-zip-compressed"})
    )

    doc = await source.fetch(ref)

    assert route.called
    assert route.calls.last.request.url.params["rcept_no"] == "20260721000123"
    assert route.calls.last.request.url.params["crtfc_key"] == "opendart-test-key"
    assert doc.payload == b"zip-bytes"
    assert doc.filename == "20260721000123.zip"
    await http.aclose()


async def test_fetch_without_key_raises():
    http = HttpClient(user_agent=USER_AGENT)
    source = OpenDartSource(http, _creds(None))
    ref = DocumentRef(source="opendart", doc_id="20260721000123")
    with pytest.raises(RuntimeError, match="opendart_key"):
        await source.fetch(ref)
    await http.aclose()
