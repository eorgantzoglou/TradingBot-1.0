"""Web / news search behind a provider seam, plus a clean-text URL fetcher.

WHY this is a seam and not a single hard-coded client:

  * The research agent needs to read news and articles *today*, before anyone
    has provisioned an API key -- so the default provider (DuckDuckGo's HTML
    endpoint) needs no credential. But that endpoint is unofficial and ToS-grey
    (PLAN.md section 2 is emphatic that this project treats terms of service as
    contractual, not advisory), so it is explicitly a best-effort default, and a
    keyed, production-grade provider (Tavily) is a drop-in replacement selected
    by config. Callers depend only on the `WebSearchProvider` protocol, so
    swapping one for the other never touches the agent.

  * PROMOTION CAVEAT. Microcap "news" is disproportionately paid promotion --
    stock-promotion newsletters, sponsored press-release wire copy, and pump
    blogs outnumber real reporting for exactly the names this project screens.
    Web search results are therefore LEADS for the research pipeline to verify
    against primary filings, never facts in their own right. Nothing here ranks,
    de-duplicates, or trusts a result; that judgement lives downstream.

The URL fetcher reuses `scout.research.evidence._html_to_text` rather than
reimplementing tag-stripping, so "readable text" means the same thing whether it
came from a filing or a web page.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qs, quote_plus, urlsplit

import httpx

from scout.data.http import HttpClient
from scout.research.evidence import _html_to_text

logger = logging.getLogger(__name__)

_DDG_HTML = "https://html.duckduckgo.com/html/"
_TAVILY_SEARCH = "https://api.tavily.com/search"

# Content types we are willing to strip to text. Anything else (PDF, image,
# JSON) is not prose and `fetch_url` returns "" rather than gibberish.
_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "application/xml", "text/xml")


@dataclass(frozen=True, slots=True)
class WebResult:
    title: str
    url: str
    snippet: str
    content: str = ""
    """Full extracted page text when the provider returns it (Tavily's
    raw_content); "" for providers that only return a snippet (DuckDuckGo)."""


class WebSearchProvider(Protocol):
    name: str

    async def search(self, http: HttpClient, query: str, *, limit: int = 5) -> list[WebResult]:
        ...


# --- DuckDuckGo (keyless default) -------------------------------------------
#
# The result markup is a flat list of anchors: the title/link carries class
# `result__a`, the description carries class `result__snippet`. We parse with
# simple regexes rather than a DOM library because the shape is stable enough
# and a parse failure must degrade to [] (a missing news lead is fine; a crash
# in an unofficial-endpoint parser taking down the agent is not). Both regexes
# tolerate attribute reordering by matching on the class token, not position.
_RESULT_A_RE = re.compile(
    r'<a\s+(?P<attrs>[^>]*?class="[^"]*\bresult__a\b[^"]*"[^>]*?)>(?P<inner>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]*?class="[^"]*\bresult__snippet\b[^"]*"[^>]*?>(?P<inner>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_HREF_RE = re.compile(r'href="(?P<href>[^"]*)"', re.IGNORECASE)


def _unwrap_ddg_url(href: str) -> str:
    """DuckDuckGo often wraps the real link in a redirect of the form
    `//duckduckgo.com/l/?uddg=<url-encoded target>&rut=...`. Return the decoded
    target, or the href unchanged when it is already a direct link.

    The href arrives HTML-escaped (`&amp;` between params), so unescape before
    splitting the query; `parse_qs` then URL-decodes the `uddg` value for us.
    """
    raw = html.unescape(href.strip())
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlsplit(raw)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return target
    return raw


def _parse_ddg_html(page: str, *, limit: int) -> list[WebResult]:
    """Extract up to `limit` results from a DuckDuckGo HTML page. Any parsing
    problem yields [] -- this endpoint is unofficial and best-effort."""
    try:
        titles = list(_RESULT_A_RE.finditer(page))
        snippets = [_html_to_text(m.group("inner")) for m in _SNIPPET_RE.finditer(page)]

        results: list[WebResult] = []
        for i, match in enumerate(titles):
            if len(results) >= limit:
                break
            href_match = _HREF_RE.search(match.group("attrs"))
            if not href_match:
                continue
            url = _unwrap_ddg_url(href_match.group("href"))
            title = _html_to_text(match.group("inner"))
            if not url or not title:
                continue
            # Snippets and titles line up positionally in the DDG markup; fall
            # back to "" if a result happens to have no snippet block.
            snippet = snippets[i] if i < len(snippets) else ""
            results.append(WebResult(title=title, url=url, snippet=snippet, content=""))
        return results
    except Exception:
        # Best-effort endpoint: never let a markup change crash the agent.
        logger.warning("duckduckgo: failed to parse HTML results", exc_info=True)
        return []


class DuckDuckGoProvider:
    """Keyless default provider. Best-effort and ToS-grey (see module docstring);
    fine as a today-it-works default, but Tavily is the production choice."""

    name = "duckduckgo"

    async def search(self, http: HttpClient, query: str, *, limit: int = 5) -> list[WebResult]:
        url = f"{_DDG_HTML}?q={quote_plus(query)}"
        try:
            response = await http.get(url)
        except httpx.HTTPError as exc:
            logger.warning("duckduckgo: request failed for %r: %s", query, exc)
            return []
        if response.status_code != 200:
            logger.warning("duckduckgo: unexpected status %d for %r", response.status_code, query)
            return []
        return _parse_ddg_html(response.text, limit=limit)


# --- Tavily (keyed, production) ---------------------------------------------


class TavilyProvider:
    """Keyed provider. POSTs to Tavily's search API and returns full page text.

    Request/response shape verified against docs.tavily.com (the search
    endpoint): the body accepts `query`, `max_results`, `include_raw_content`,
    and a legacy `api_key` field; each result carries `title`, `url`, `content`
    (a short description) and, when requested, `raw_content` (cleaned full text).
    """

    name = "tavily"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def search(self, http: HttpClient, query: str, *, limit: int = 5) -> list[WebResult]:
        body = {
            "api_key": self._api_key,
            "query": query,
            "max_results": limit,
            "include_raw_content": True,
        }
        response = await http.request("POST", _TAVILY_SEARCH, json=body)
        # A bad key is an auth error, not an empty result set. Surfacing it (per
        # the Source contract) beats silently returning [] and looking like "no
        # news", which would be indistinguishable from a real quiet day.
        response.raise_for_status()
        payload = response.json()
        raw_results = payload.get("results", []) if isinstance(payload, dict) else []

        results: list[WebResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            if len(results) >= limit:
                break
            # Prefer the full cleaned body; fall back to the short description
            # when raw_content was not returned for this result.
            short = item.get("content") or ""
            content = item.get("raw_content") or short
            results.append(
                WebResult(
                    title=item.get("title") or "",
                    url=item.get("url") or "",
                    snippet=short[:300],
                    content=content,
                )
            )
        return results


def build_web_provider(
    provider: str, *, tavily_api_key: str | None = None
) -> WebSearchProvider | None:
    """Return the configured provider, or None if it can't be built.

    "duckduckgo" is the keyless default and always builds. "tavily" builds only
    when a key is supplied; without one we return None (and log) so the caller
    can fall back or skip web search rather than crash. Unknown names -> None.
    """
    key = provider.strip().lower()
    if key == "duckduckgo":
        return DuckDuckGoProvider()
    if key == "tavily":
        if not tavily_api_key:
            logger.warning("web provider 'tavily' selected but no API key provided; not building")
            return None
        return TavilyProvider(tavily_api_key)
    logger.warning("unknown web search provider %r; not building", provider)
    return None


async def fetch_url(http: HttpClient, url: str, *, max_chars: int = 8000) -> str:
    """Fetch a URL and return clean, whitespace-normalised text, truncated to
    `max_chars`. Returns "" on a non-HTML content type, a bad status, or a
    transport failure -- callers treat "" as "nothing readable here"."""
    try:
        response = await http.get(url)
    except httpx.HTTPError as exc:
        logger.warning("fetch_url: request failed for %s: %s", url, exc)
        return ""
    if response.status_code != 200:
        logger.warning("fetch_url: status %d for %s", response.status_code, url)
        return ""

    content_type = response.headers.get("content-type", "").lower()
    if not any(ct in content_type for ct in _HTML_CONTENT_TYPES):
        # Not prose we can strip to text (PDF, image, JSON, binary). Silently
        # returning "" is correct here -- it is a normal "skip this" outcome.
        return ""

    text = _html_to_text(response.text)
    return text[:max_chars]
