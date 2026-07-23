"""SEC EDGAR full-text search (US) -- keyword DISCOVERY of filings.

Every other SEC entry point in this project (see `sec.py`) is *listing* by day
or *fetching* a known document. Neither can answer "which filings mention this
phrase?" -- that is what the research agent needs to go from a thesis keyword to
a set of candidate filings without knowing any CIK or accession up front. EDGAR
exposes exactly that as a free, key-less JSON API behind the
https://www.sec.gov/edgar/search/ frontend; this module is a thin, defensive
wrapper over it.

Scope note: full-text search only covers filings from 2001 onward, and it
indexes the primary documents and exhibits, not the raw submission wrapper. It
is a discovery tool, not a system of record -- once a hit points at an
accession, the authoritative bytes still come through `sec.py`'s fetch path.

The endpoint and response shape below were verified live against
`https://efts.sec.gov/LATEST/search-index?q=...` (a 200 returns an
Elasticsearch-style envelope: `{"hits": {"total": {"value": N}, "hits": [...]}}`
with each hit carrying `_id` = "<dashed-accession>:<primary-doc>" and a
`_source` object of arrays). Parsing is written to degrade a malformed-but-200
body to `[]` rather than raise, because a shape drift on SEC's side must not
take the discovery step down -- a network/HTTP error, by contrast, propagates
out of `http.get_json` as it should.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlencode

from scout.data.http import HttpClient

logger = logging.getLogger(__name__)

# The frontend at /edgar/search/ calls this host; it is separately rate-limited
# in http.py's DEFAULT_LIMITS. "LATEST" is EDGAR's own alias for the current
# index version -- there is no dated/pinned variant to prefer.
_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"


@dataclass(frozen=True, slots=True)
class FilingHit:
    accession: str  # dashed form, e.g. "0001683168-26-005674"
    cik: str  # digits as returned by EDGAR; leading zeros not stripped
    company: str  # display name, e.g. "CONECTISYS CORP (CE)"
    form: str  # e.g. "10-K", "8-K"
    filed: str  # ISO date "YYYY-MM-DD"
    doc: str  # primary document filename, from the hit id after the ':'


def _first(value: object) -> str:
    """EDGAR returns `ciks`, `display_names`, `root_forms` as arrays even when a
    filing has exactly one filer/name/form. Take the first element as a string;
    an empty or non-list value degrades to "" rather than raising, so one odd
    hit does not sink the whole page."""
    if isinstance(value, list) and value:
        first = value[0]
        return str(first) if first is not None else ""
    return ""


def _parse_hit(hit: object) -> FilingHit | None:
    """One raw hit -> FilingHit, or None if it is too malformed to use.

    Returning None (rather than raising) is deliberate: the caller filters
    Nones out, so a single unexpected hit shape costs one result, not the query.
    """
    if not isinstance(hit, dict):
        return None

    # `_id` is "<dashed-accession>:<primary-doc>". The split is the whole reason
    # this parse can't be a plain dict lookup -- accession and doc are packed
    # into one field. Missing/odd ids yield empty parts rather than an
    # IndexError.
    raw_id = hit.get("_id")
    accession, _, doc = (raw_id if isinstance(raw_id, str) else "").partition(":")

    source = hit.get("_source")
    if not isinstance(source, dict):
        source = {}

    filed = source.get("file_date")
    return FilingHit(
        accession=accession,
        cik=_first(source.get("ciks")),
        company=_first(source.get("display_names")),
        # `root_forms` is the normalized base form ("10-K" for a "10-K/A"),
        # which is what a keyword-discovery caller wants to filter and group on.
        form=_first(source.get("root_forms")),
        filed=filed if isinstance(filed, str) else "",
        doc=doc,
    )


async def search_filings(
    http: HttpClient,
    query: str,
    *,
    forms: list[str] | None = None,
    limit: int = 10,
) -> list[FilingHit]:
    """Discover filings mentioning `query` via EDGAR full-text search.

    `forms` maps to EDGAR's `&forms=10-K,8-K` filter (base form types). `limit`
    truncates client-side -- EDGAR pages at 10 hits, but we slice regardless so
    the caller's cap is always honoured no matter what the server returns.

    A malformed-but-200 response returns []; HTTP/network errors propagate from
    `http.get_json`.
    """
    params = {"q": query}
    if forms:
        params["forms"] = ",".join(forms)
    url = f"{_SEARCH_URL}?{urlencode(params)}"

    payload = await http.get_json(url)
    if not isinstance(payload, dict):
        logger.warning("sec_fulltext: unexpected non-object response for query %r", query)
        return []

    hits_container = payload.get("hits")
    if not isinstance(hits_container, dict):
        logger.warning("sec_fulltext: response missing 'hits' object for query %r", query)
        return []

    # total.value == 0 is a legitimate no-results answer, not an error. The
    # hits list below would already be empty in that case; checking total is
    # only a cheap guard against iterating a surprising shape.
    total = hits_container.get("total")
    if isinstance(total, dict) and total.get("value") == 0:
        return []

    raw_hits = hits_container.get("hits")
    if not isinstance(raw_hits, list):
        logger.warning("sec_fulltext: 'hits.hits' was not a list for query %r", query)
        return []

    parsed = (_parse_hit(hit) for hit in raw_hits[:limit])
    return [hit for hit in parsed if hit is not None]
