"""South Korea -- OpenDART (Financial Supervisory Service).

The one thing to get right in this module: OpenDART signals errors through a
`status` field in the JSON body, never through the HTTP status code -- every
response is HTTP 200 regardless of what happened. `"000"` is success, `"013"`
means "no data for this query" (a normal empty day, e.g. a weekend or public
holiday -- must NOT raise), and every other code is a real problem (`"010"`
unregistered key, `"011"` deactivated key, `"020"` request limit exceeded,
...). Treating any non-"000" status as "just an empty list" would make a bad
API key indistinguishable from a quiet day, which is exactly the failure mode
that silently loses archive history (see the Source protocol docstring).

Responses are in Korean, including `report_nm` (report name) and account
names once a filing's financial statements are parsed downstream -- this
module does not translate anything; a Korean-to-canonical account-name
mapping belongs in the parsing/normalization layer (metrics/normalize.py),
not here. We only carry bytes and identifiers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from typing import Any

from scout.config import SourceCredentials
from scout.data.http import HttpClient
from scout.data.sources.base import DocumentRef, RawDocument

BASE_URL = "https://opendart.fss.or.kr/api/"

STATUS_OK = "000"
STATUS_NO_DATA = "013"
"""Not an error -- a legitimately empty day. Must not raise."""

_PAGE_COUNT = 100


def _parse_rcept_dt(raw: str | None) -> date | None:
    """`rcept_dt` looks like "20260721". A malformed or missing value on one
    record shouldn't kill the whole listing."""
    if not raw or len(raw) != 8:
        return None
    try:
        return date(int(raw[0:4]), int(raw[4:6]), int(raw[6:8]))
    except ValueError:
        return None


class OpenDartSource:
    """Harvests OpenDART's daily filing list and downloads individual filings."""

    name = "opendart"

    def __init__(self, http: HttpClient, credentials: SourceCredentials) -> None:
        self._http = http
        self._key = credentials.opendart_key

    def available(self) -> bool:
        return bool(self._key)

    def _require_key(self) -> str:
        if not self._key:
            raise RuntimeError(
                "OpenDartSource requires SourceCredentials.opendart_key; "
                "call available() before using this source."
            )
        return self._key

    async def list_documents(self, day: date) -> AsyncIterator[DocumentRef]:
        """Everything OpenDART published for `day`, walking `total_page`."""
        key = self._require_key()
        date_str = day.strftime("%Y%m%d")
        page_no = 1

        while True:
            params = {
                "crtfc_key": key,
                "bgn_de": date_str,
                "end_de": date_str,
                "page_no": str(page_no),
                "page_count": str(_PAGE_COUNT),
            }
            data = await self._http.get_json(f"{BASE_URL}list.json", params=params)
            assert isinstance(data, dict)

            status = data.get("status")
            if status == STATUS_NO_DATA:
                return  # genuinely empty day -- not an error
            if status != STATUS_OK:
                # Real error: bad key, rate limit, system maintenance, etc. Must
                # be loud -- see module docstring on why this distinction matters.
                raise RuntimeError(
                    f"OpenDART list.json error {status}: {data.get('message')} "
                    f"(bgn_de={date_str}, page_no={page_no})"
                )

            for item in data.get("list") or []:
                ref = self._to_ref(item)
                if ref is not None:
                    yield ref

            total_page = int(data.get("total_page") or 1)
            if page_no >= total_page:
                return
            page_no += 1

    def _to_ref(self, item: dict[str, Any]) -> DocumentRef | None:
        rcept_no = item.get("rcept_no")
        if not rcept_no:
            return None

        return DocumentRef(
            source=self.name,
            doc_id=rcept_no,
            filing_date=_parse_rcept_dt(item.get("rcept_dt")),
            form_type=item.get("report_nm"),
            title=item.get("report_nm"),
            url=f"{BASE_URL}document.xml?rcept_no={rcept_no}",
            entity={
                "corp_code": item.get("corp_code") or "",
                "stock_code": item.get("stock_code") or "",
                "corp_cls": item.get("corp_cls") or "",
            },
            meta={
                "corp_name": item.get("corp_name") or "",
                "flr_nm": item.get("flr_nm") or "",
                "rm": item.get("rm") or "",
            },
        )

    async def fetch(self, ref: DocumentRef) -> RawDocument:
        """Download one filing. Despite the endpoint name `document.xml`, this
        returns a ZIP archive of XML documents, not a bare XML file."""
        key = self._require_key()
        response = await self._http.get(
            f"{BASE_URL}document.xml", params={"crtfc_key": key, "rcept_no": ref.doc_id}
        )
        response.raise_for_status()

        return RawDocument(
            ref=ref,
            payload=response.content,
            filename=f"{ref.doc_id}.zip",
            content_type=response.headers.get("content-type"),
        )
