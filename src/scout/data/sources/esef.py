"""filings.xbrl.org (EU + UK + UA ESEF filings).

Two things that will otherwise bite whoever reads this later:

1. **This is not a complete European picture.** ESEF (mandatory iXBRL for
   annual reports) applies to issuers on EU *regulated* markets only. AIM,
   Euronext Growth, Nasdaq First North and Scale are MTFs and are EXEMPT --
   which is exactly where most European microcaps list (PLAN.md section 1.2).
   Germany and Ireland are absent from filings.xbrl.org entirely, not just
   under-covered. Treat this source as "regulated-market EU/UK/UA", never as
   "Europe".
2. **The licence is unusually good.** The operator states there are currently
   no restrictions on use of the data -- the only source in this project with
   that property besides SEC EDGAR's public-domain status.

The API (`https://filings.xbrl.org/api`) is JSON:API and is undocumented in
places. The operator reserves the right to change or withdraw it, and some
filings fail processing upstream (no package, no xBRL-JSON, sometimes no
period_end). Every field read from a filing record here degrades to None
rather than raising -- an incomplete record is normal, not exceptional.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import date
from typing import Any
from urllib.parse import urljoin

from scout.data.http import HttpClient
from scout.data.sources.base import DocumentRef, RawDocument

logger = logging.getLogger(__name__)

_SITE = "https://filings.xbrl.org"
_API = f"{_SITE}/api"

# Preference order for the download link on a filing. xBRL-JSON is
# pre-normalized and far cheaper to parse downstream than re-deriving facts
# from the iXBRL buried in the ZIP; the rendered report is a last resort kept
# only so a ref is never entirely undownloadable when the others are missing.
_DOWNLOAD_KINDS: tuple[tuple[str, str], ...] = (
    ("json_url", "xbrl-json"),
    ("package_url", "zip"),
    ("report_url", "report_html"),
)
_EXTENSION_BY_KIND = {"xbrl-json": "json", "zip": "zip", "report_html": "xhtml"}
_CONTENT_TYPE_BY_KIND = {
    "xbrl-json": "application/json",
    "zip": "application/zip",
    "report_html": "application/xhtml+xml",
}


class EsefSource:
    """`Source` implementation for filings.xbrl.org."""

    name = "esef"

    def __init__(self, http: HttpClient, *, country: str | None = None, page_size: int = 200) -> None:
        self._http = http
        self._country = country
        """Optional server-side `filter[country]`, e.g. "GB". Left unset by
        default so a day's harvest covers every country filings.xbrl.org
        knows about in one paginated sweep; venue whitelisting happens later,
        in the universe layer, not here."""
        self._page_size = page_size

    def available(self) -> bool:
        return True  # no auth on this API at all, see module docstring

    async def list_documents(self, day: date) -> AsyncIterator[DocumentRef]:
        target = day.isoformat()
        page = 1
        while True:
            params: dict[str, Any] = {
                "sort": "-processed",
                "include": "entity",
                "page[size]": self._page_size,
                "page[number]": page,
            }
            if self._country:
                params["filter[country]"] = self._country

            body = await self._http.get_json(f"{_API}/filings", params=params)
            if not isinstance(body, dict) or not isinstance(body.get("data"), list):
                raise ValueError(
                    f"unexpected filings.xbrl.org response shape on page {page} for {day} "
                    "-- the API is undocumented and may have changed."
                )

            data = body["data"]
            if not data:
                return  # exhausted the whole collection before reaching the target day

            entities = _index_entities(body.get("included"))
            passed_target = False
            for item in data:
                ref, item_date = _parse_filing(item, entities)
                if ref is None:
                    logger.warning("esef: skipping unparseable filing record: %r", item)
                    continue
                if item_date is None:
                    # No processed/date_added timestamp at all -- it can never
                    # be placed on a day, so a day-indexed harvest can never
                    # select it. Surfaced rather than silently dropped.
                    logger.warning("esef: filing %s has no usable date, skipping", ref.doc_id)
                    continue
                if item_date > target:
                    continue  # newer than the target day; keep paging
                if item_date < target:
                    # Sorted -processed (newest first): once we're older than
                    # the target day, every remaining page only gets older.
                    passed_target = True
                    break
                yield ref

            if passed_target:
                return
            if not _has_next_page(body, received=len(data), page_size=self._page_size):
                return
            page += 1

    async def fetch(self, ref: DocumentRef) -> RawDocument:
        if not ref.url:
            raise ValueError(f"esef filing {ref.doc_id} has no download URL (processing may have failed upstream)")
        payload = await self._http.get_bytes(ref.url)
        kind = ref.meta.get("download_kind")
        extension = _EXTENSION_BY_KIND.get(kind, "bin")
        content_type = _CONTENT_TYPE_BY_KIND.get(kind)
        return RawDocument(
            ref=ref,
            payload=payload,
            filename=f"{ref.doc_id}.{extension}",
            content_type=content_type,
        )


def _index_entities(included: object) -> dict[str, dict[str, Any]]:
    if not isinstance(included, list):
        return {}
    return {
        str(item["id"]): (item.get("attributes") or {})
        for item in included
        if isinstance(item, dict) and item.get("type") == "entity" and "id" in item
    }


def _dig(obj: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _pick_download(attrs: dict[str, Any]) -> tuple[str | None, str | None]:
    for field, kind in _DOWNLOAD_KINDS:
        value = attrs.get(field)
        if value:
            return urljoin(_SITE, value), kind
    return None, None


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _date_part(value: object) -> str | None:
    """First 10 chars of a "YYYY-MM-DD HH:MM:SS.ffffff" timestamp, for
    lexical day comparison -- ISO dates sort correctly as plain strings."""
    if not isinstance(value, str) or len(value) < 10:
        return None
    return value[:10]


def _parse_filing(
    item: dict[str, Any], entities: dict[str, dict[str, Any]]
) -> tuple[DocumentRef | None, str | None]:
    filing_id = item.get("id")
    if not filing_id:
        return None, None
    attrs = item.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}

    entity_id = _dig(item, "relationships", "entity", "data", "id")
    entity_attrs = entities.get(str(entity_id), {}) if entity_id is not None else {}

    entity: dict[str, str] = {}
    lei = entity_attrs.get("identifier")
    if lei:
        entity["lei"] = lei
    country = attrs.get("country")
    if country:
        entity["country"] = country

    url, kind = _pick_download(attrs)
    meta: dict[str, Any] = {}
    if kind:
        meta["download_kind"] = kind
    date_added = attrs.get("date_added")
    if date_added:
        meta["date_added"] = date_added
    fxo_id = attrs.get("fxo_id")
    if fxo_id:
        meta["fxo_id"] = fxo_id
    error_count = attrs.get("error_count")
    if error_count is not None:
        meta["error_count"] = error_count

    ref = DocumentRef(
        source="esef",
        doc_id=str(filing_id),
        filing_date=_parse_date(attrs.get("period_end")),
        form_type=None,  # ESEF has no form-type taxonomy the way EDGAR does
        title=entity_attrs.get("name"),
        url=url,
        entity=entity,
        meta=meta,
    )
    processed = attrs.get("processed") or attrs.get("date_added")
    return ref, _date_part(processed)


def _has_next_page(body: dict[str, Any], *, received: int, page_size: int) -> bool:
    links = body.get("links")
    if isinstance(links, dict):
        return bool(links.get("next"))
    # No links object at all: undocumented API, degrade to a size heuristic
    # rather than assuming either "definitely done" or "infinite".
    return received >= page_size
