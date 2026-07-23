"""Rate-limited async HTTP, per host.

Every source we harvest publishes a different limit, and several of them ban
rather than throttle. The limits below are the published ones -- treat them as
contractual, not advisory:

  SEC EDGAR         10 req/s across ALL machines, and a descriptive User-Agent
                    is mandatory (403 without it). SEC blocks "unclassified bots".
  Companies House   600 requests / 5 minutes per key; they reserve the right to
                    ban applications that repeatedly exceed it.
  EDINET (Japan)    guidance is 3-5 seconds between requests. Not a rate, an interval.
  OpenDART (Korea)  ~20,000 requests/day.
  FINRA             1,200 req/min per IP.
  OpenFIGI          25 req/min anonymous, 250 with a key.
  filings.xbrl.org  no documented limit, but the operator reserves the right to
                    impose one or withdraw the API. Be polite.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class RateLimit:
    """Token bucket plus an optional hard floor between requests.

    `min_interval` exists because some publishers (EDINET) specify a gap
    between requests rather than a rate, and a bucket with burst would violate
    that even while respecting the average.
    """

    per_second: float
    burst: int = 1
    min_interval: float = 0.0

    @staticmethod
    def per_minute(n: float, burst: int = 1) -> RateLimit:
        return RateLimit(per_second=n / 60.0, burst=burst)

    @staticmethod
    def every(seconds: float) -> RateLimit:
        return RateLimit(per_second=1.0 / seconds, burst=1, min_interval=seconds)


DEFAULT_LIMITS: dict[str, RateLimit] = {
    "data.sec.gov": RateLimit(per_second=8.0, burst=4),
    "www.sec.gov": RateLimit(per_second=8.0, burst=4),
    "efts.sec.gov": RateLimit(per_second=8.0, burst=4),
    "api.company-information.service.gov.uk": RateLimit.per_minute(110, burst=5),
    "download.companieshouse.gov.uk": RateLimit(per_second=1.0),
    "api.edinet-fsa.go.jp": RateLimit.every(4.0),
    "opendart.fss.or.kr": RateLimit(per_second=2.0, burst=2),
    "filings.xbrl.org": RateLimit(per_second=2.0, burst=2),
    "api.finra.org": RateLimit.per_minute(600, burst=10),
    "api.openfigi.com": RateLimit.per_minute(20, burst=5),
    "api.gleif.org": RateLimit(per_second=2.0, burst=2),
    "data-api.ecb.europa.eu": RateLimit(per_second=2.0, burst=2),
}

_DEFAULT_LIMIT = RateLimit(per_second=2.0, burst=2)


class _Bucket:
    def __init__(self, limit: RateLimit):
        self._limit = limit
        self._tokens = float(limit.burst)
        self._updated = time.monotonic()
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(
                    float(self._limit.burst),
                    self._tokens + (now - self._updated) * self._limit.per_second,
                )
                self._updated = now

                wait = 0.0
                if self._tokens < 1.0:
                    wait = (1.0 - self._tokens) / self._limit.per_second
                if self._limit.min_interval:
                    gap = self._limit.min_interval - (now - self._last_request)
                    wait = max(wait, gap)

                if wait <= 0:
                    self._tokens -= 1.0
                    self._last_request = now
                    return
                await asyncio.sleep(wait)


class HttpClient:
    """Shared async client. One per process; pass it to every source.

    Retries 429 and 5xx with exponential backoff and jitter, honouring
    Retry-After when present. Does NOT retry 4xx other than 429 -- those are
    our bug, not their outage.
    """

    def __init__(
        self,
        *,
        user_agent: str,
        limits: dict[str, RateLimit] | None = None,
        timeout: float = 60.0,
        max_retries: int = 4,
    ):
        if not user_agent or "@" not in user_agent:
            # SEC requires "AppName contact@email"; getting this wrong returns a
            # 403 HTML page rather than a clean error, which is confusing enough
            # to be worth failing loudly here instead.
            raise ValueError(
                "user_agent must include a contact email, e.g. 'scout/0.1 you@example.com'. "
                "SEC EDGAR rejects requests without one."
            )
        self._limits = {**DEFAULT_LIMITS, **(limits or {})}
        self._buckets: dict[str, _Bucket] = {}
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            follow_redirects=True,
        )

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _bucket(self, url: str) -> _Bucket:
        host = urlsplit(url).netloc.lower()
        if host not in self._buckets:
            self._buckets[host] = _Bucket(self._limits.get(host, _DEFAULT_LIMIT))
        return self._buckets[host]

    async def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        bucket = self._bucket(url)
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            await bucket.acquire()
            try:
                response = await self._client.request(method, url, **kwargs)  # type: ignore[arg-type]
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    raise
                await asyncio.sleep(self._backoff(attempt))
                continue

            if response.status_code in RETRY_STATUSES and attempt < self._max_retries:
                await asyncio.sleep(self._retry_delay(response, attempt))
                continue
            return response

        raise RuntimeError(f"unreachable retry state for {url}") from last_exc

    def _backoff(self, attempt: int) -> float:
        return min(30.0, 2.0**attempt) * (0.5 + random.random() / 2)

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(60.0, float(retry_after))
            except ValueError:
                pass
        return self._backoff(attempt)

    async def get(self, url: str, **kwargs: object) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def get_json(self, url: str, **kwargs: object) -> object:
        response = await self.get(url, **kwargs)
        response.raise_for_status()
        return response.json()

    async def get_bytes(self, url: str, **kwargs: object) -> bytes:
        response = await self.get(url, **kwargs)
        response.raise_for_status()
        return response.content
