"""Japan -- EDINET API v2 (Financial Services Agency).

The single fact that shapes this module: EDINET's `documents.json` listing
endpoint is indexed by date and nothing else -- there is no "list everything
since X" or "list everything for filer Y" mode. Building a universe therefore
means walking every calendar day one at a time and archiving whatever that day
returns. That is slow and there is no shortcut; see PLAN.md sections 1.3 and 2.

`type=5` on the fetch endpoint returns a CSV conversion of the XBRL, which
saves us from parsing XBRL directly for the common case. Not every filing has
one (older or unusual submissions may lack `csvFlag`), so we fall back to the
raw XBRL zip (`type=1`) when that happens.

Content is Japanese throughout -- filer names, document descriptions, and the
CSV cell values once parsed. No translation happens here; the archive stores
raw bytes and a later stage in the pipeline is the LLM's job (PLAN.md 1.1).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date, datetime
from typing import Any

from scout.config import SourceCredentials
from scout.data.http import HttpClient
from scout.data.sources.base import DocumentRef, RawDocument

BASE_URL = "https://api.edinet-fsa.go.jp/api/v2/"

# EDINET's own numbering, not ours -- see the class docstring on `fetch`.
_DOC_TYPE_XBRL_ZIP = "1"
_DOC_TYPE_PDF = "2"
_DOC_TYPE_CSV = "5"

_NOT_WITHDRAWN = "0"
_FLAG_AVAILABLE = "1"


def _parse_submit_date(raw: str | None) -> date | None:
    """`submitDateTime` looks like "2026-07-21 15:00". A malformed or missing
    value shouldn't kill the whole listing -- one bad record is skipped, not
    fatal (see the Source protocol docstring)."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


class EdinetSource:
    """Harvests EDINET's daily document list and downloads individual filings."""

    name = "edinet"

    def __init__(
        self,
        http: HttpClient,
        credentials: SourceCredentials,
        *,
        doc_type: str = _DOC_TYPE_CSV,
    ) -> None:
        self._http = http
        self._key = credentials.edinet_key
        # Configurable so a caller can force the raw XBRL zip instead of the
        # CSV conversion; defaults to CSV because it's the big time-saver.
        self._doc_type = doc_type

    def available(self) -> bool:
        return bool(self._key)

    def _require_key(self) -> str:
        if not self._key:
            # A missing key must fail loudly, not silently return no documents --
            # otherwise a broken key and a genuinely empty day look identical.
            # Callers should check `available()` first to skip this source cleanly.
            raise RuntimeError(
                "EdinetSource requires SourceCredentials.edinet_key; "
                "call available() before using this source."
            )
        return self._key

    async def list_documents(self, day: date) -> AsyncIterator[DocumentRef]:
        """Everything EDINET published for `day`.

        The list endpoint has no filter beyond date, so this is the only call
        this method ever makes per invocation -- no pagination exists here
        (unlike OpenDART and its `page_no`).
        """
        key = self._require_key()
        params = {"date": day.isoformat(), "type": "2", "Subscription-Key": key}
        data = await self._http.get_json(f"{BASE_URL}documents.json", params=params)
        assert isinstance(data, dict)

        metadata = data.get("metadata") or {}
        status = metadata.get("status")
        if status is not None and status != "200":
            # EDINET reports logical errors (bad params, disabled key) inside a
            # 200-OK JSON body rather than via HTTP status -- an HTTP-level
            # raise_for_status() would miss this entirely.
            raise RuntimeError(
                f"EDINET documents.json error {status}: {metadata.get('message')} (date={day})"
            )

        for item in data.get("results") or []:
            ref = self._to_ref(item)
            if ref is not None:
                yield ref

    def _to_ref(self, item: dict[str, Any]) -> DocumentRef | None:
        doc_id = item.get("docID")
        if not doc_id:
            return None

        if item.get("withdrawalStatus", _NOT_WITHDRAWN) != _NOT_WITHDRAWN:
            return None  # withdrawn filing -- not part of the point-in-time record

        xbrl_flag = item.get("xbrlFlag")
        csv_flag = item.get("csvFlag")
        if xbrl_flag != _FLAG_AVAILABLE and csv_flag != _FLAG_AVAILABLE:
            return None  # nothing fetchable: neither raw XBRL nor the CSV conversion exists

        return DocumentRef(
            source=self.name,
            doc_id=doc_id,
            filing_date=_parse_submit_date(item.get("submitDateTime")),
            form_type=item.get("docTypeCode"),
            title=item.get("docDescription"),
            url=f"{BASE_URL}documents/{doc_id}",
            entity={
                "edinet_code": item.get("edinetCode") or "",
                "sec_code": item.get("secCode") or "",
                "jcn": item.get("JCN") or "",
            },
            meta={
                "xbrl_flag": xbrl_flag or "0",
                "csv_flag": csv_flag or "0",
                "filer_name": item.get("filerName") or "",
            },
        )

    async def fetch(self, ref: DocumentRef) -> RawDocument:
        """Download one filing.

        type=1 XBRL zip, type=2 PDF, type=5 CSV conversion of the XBRL. We
        default to type=5 and fall back to type=1 when this particular
        document's `csvFlag` says no CSV was produced for it.
        """
        key = self._require_key()
        doc_type = self._doc_type
        if doc_type == _DOC_TYPE_CSV and ref.meta.get("csv_flag") != _FLAG_AVAILABLE:
            doc_type = _DOC_TYPE_XBRL_ZIP

        url = f"{BASE_URL}documents/{ref.doc_id}"
        response = await self._http.get(url, params={"type": doc_type, "Subscription-Key": key})
        response.raise_for_status()

        ext = "pdf" if doc_type == _DOC_TYPE_PDF else "zip"
        filename = f"{ref.doc_id}_type{doc_type}.{ext}"
        return RawDocument(
            ref=ref,
            payload=response.content,
            filename=filename,
            content_type=response.headers.get("content-type"),
        )
