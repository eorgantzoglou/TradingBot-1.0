"""Tests for scout.data.sources.web. No network calls -- respx mocks httpx.

The DuckDuckGo fixture mirrors the real HTML endpoint's shape: title/link
anchors carry class `result__a`, snippets carry `result__snippet`, hrefs are
HTML-escaped, and one result link is a `//duckduckgo.com/l/?uddg=...` redirect
wrapper (the other is a direct link) so the unwrap path is actually exercised.
"""

from __future__ import annotations

import httpx
import respx

from scout.data.http import HttpClient
from scout.data.sources import web

# Two results: the first is wrapped in DDG's redirect (uddg=<url-encoded>) and
# has a bolded title; the second is a direct link. Trailing `&amp;rut=...` on
# the wrapper proves the href is unescaped before the query is parsed.
_DDG_FIXTURE = """
<html><body>
  <div class="result results_links_deep web-result">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Falpha-news&amp;rut=deadbeef">
        Alpha <b>Corp</b> soars on earnings
      </a>
    </h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">
      Alpha Corp announced record quarterly earnings today.
    </a>
  </div>
  <div class="result results_links_deep web-result">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="https://example.org/beta-report">
        Beta Holdings files report
      </a>
    </h2>
    <a class="result__snippet" href="https://example.org/beta-report">
      Beta Holdings issues new shares in a registered direct offering.
    </a>
  </div>
</body></html>
"""


def _client() -> HttpClient:
    return HttpClient(user_agent="scout-test/0.1 test@example.com")


class TestDuckDuckGoProvider:
    async def test_parses_two_results_with_unwrapped_urls(self):
        async with respx.mock:
            respx.get(url__regex=r"https://html\.duckduckgo\.com/html/.*").mock(
                return_value=httpx.Response(200, html=_DDG_FIXTURE)
            )
            async with _client() as http:
                results = await web.DuckDuckGoProvider().search(http, "alpha corp")

        assert len(results) == 2

        first, second = results
        # The redirect wrapper is unwrapped and URL-decoded to the real target.
        assert first.url == "https://example.com/alpha-news"
        assert first.title == "Alpha Corp soars on earnings"  # <b> stripped, whitespace normalised
        assert "record quarterly earnings" in first.snippet
        assert first.content == ""  # DuckDuckGo returns no full page text

        # The direct link is returned unchanged.
        assert second.url == "https://example.org/beta-report"
        assert second.title == "Beta Holdings files report"
        assert "registered direct offering" in second.snippet

    async def test_limit_caps_results(self):
        async with respx.mock:
            respx.get(url__regex=r"https://html\.duckduckgo\.com/html/.*").mock(
                return_value=httpx.Response(200, html=_DDG_FIXTURE)
            )
            async with _client() as http:
                results = await web.DuckDuckGoProvider().search(http, "alpha", limit=1)

        assert len(results) == 1
        assert results[0].url == "https://example.com/alpha-news"

    async def test_malformed_html_returns_empty_list(self):
        async with respx.mock:
            respx.get(url__regex=r"https://html\.duckduckgo\.com/html/.*").mock(
                return_value=httpx.Response(200, html="<html><body>no results here</body></html>")
            )
            async with _client() as http:
                results = await web.DuckDuckGoProvider().search(http, "nothing")

        assert results == []

    async def test_non_200_returns_empty_list(self):
        async with respx.mock:
            respx.get(url__regex=r"https://html\.duckduckgo\.com/html/.*").mock(
                return_value=httpx.Response(503)
            )
            async with _client() as http:
                results = await web.DuckDuckGoProvider().search(http, "anything")

        assert results == []


class TestTavilyProvider:
    async def test_maps_results_and_populates_content(self):
        body = {
            "results": [
                {
                    "title": "Alpha Corp Q3 results",
                    "url": "https://news.example.com/alpha",
                    "content": "Alpha Corp beat estimates.",
                    "raw_content": "Alpha Corp beat estimates. " + "full body text " * 40,
                },
                {
                    "title": "Beta note",
                    "url": "https://news.example.com/beta",
                    "content": "Beta short description.",
                    "raw_content": None,  # provider omitted raw content for this one
                },
            ]
        }
        async with respx.mock:
            respx.post("https://api.tavily.com/search").mock(
                return_value=httpx.Response(200, json=body)
            )
            async with _client() as http:
                results = await web.TavilyProvider("k").search(http, "alpha corp")

        assert len(results) == 2

        first, second = results
        assert first.title == "Alpha Corp Q3 results"
        assert first.url == "https://news.example.com/alpha"
        assert first.content.startswith("Alpha Corp beat estimates.")
        assert "full body text" in first.content  # raw_content preferred
        assert first.snippet == "Alpha Corp beat estimates."  # short content field

        # When raw_content is missing, content falls back to the short field.
        assert second.content == "Beta short description."
        assert second.snippet == "Beta short description."

    async def test_snippet_truncated_to_300_chars(self):
        long_content = "x" * 500
        body = {
            "results": [
                {"title": "T", "url": "https://e.com", "content": long_content, "raw_content": "r"}
            ]
        }
        async with respx.mock:
            respx.post("https://api.tavily.com/search").mock(
                return_value=httpx.Response(200, json=body)
            )
            async with _client() as http:
                results = await web.TavilyProvider("k").search(http, "q")

        assert len(results[0].snippet) == 300


class TestBuildWebProvider:
    def test_duckduckgo_is_the_keyless_default(self):
        provider = web.build_web_provider("duckduckgo")
        assert isinstance(provider, web.DuckDuckGoProvider)
        assert provider.name == "duckduckgo"

    def test_tavily_without_key_returns_none(self):
        assert web.build_web_provider("tavily", tavily_api_key=None) is None

    def test_tavily_with_key_returns_provider(self):
        provider = web.build_web_provider("tavily", tavily_api_key="secret")
        assert isinstance(provider, web.TavilyProvider)
        assert provider.name == "tavily"

    def test_unknown_provider_returns_none(self):
        assert web.build_web_provider("bing") is None


class TestFetchUrl:
    async def test_returns_clean_truncated_text(self):
        # _html_to_text drops <script>/<style> bodies but keeps other text, so
        # the head deliberately carries no visible text -- the first prose is
        # the <h1>, which is what we assert on below.
        page = (
            "<html><head>"
            "<style>.x{color:red}</style></head><body>"
            "<h1>Headline</h1>   <p>Some   article   text&nbsp;here.</p>"
            "<script>var a = 1;</script>"
            "</body></html>"
        )
        async with respx.mock:
            respx.get("https://example.com/article").mock(
                return_value=httpx.Response(200, html=page)
            )
            async with _client() as http:
                text = await web.fetch_url(http, "https://example.com/article", max_chars=20)

        assert len(text) == 20
        assert "<" not in text  # tags stripped
        assert text.startswith("Headline")  # whitespace normalised, script/style dropped

    async def test_full_text_when_under_limit(self):
        page = "<html><body><p>Short and clean.</p></body></html>"
        async with respx.mock:
            respx.get("https://example.com/short").mock(
                return_value=httpx.Response(200, html=page)
            )
            async with _client() as http:
                text = await web.fetch_url(http, "https://example.com/short")

        assert text == "Short and clean."

    async def test_non_200_returns_empty(self):
        async with respx.mock:
            respx.get("https://example.com/missing").mock(return_value=httpx.Response(404))
            async with _client() as http:
                text = await web.fetch_url(http, "https://example.com/missing")

        assert text == ""

    async def test_non_html_content_type_returns_empty(self):
        async with respx.mock:
            respx.get("https://example.com/data.json").mock(
                return_value=httpx.Response(200, json={"a": 1})
            )
            async with _client() as http:
                text = await web.fetch_url(http, "https://example.com/data.json")

        assert text == ""
