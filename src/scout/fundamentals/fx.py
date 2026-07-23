"""FX conversion for cross-currency screening, per IAS 21 and ECB reference rates.

Global screening compares companies that report in different currencies.
Applying a single spot rate to every line item of a filing is a common and
silent bug: IAS 21 requires balance-sheet (INSTANT) items to convert at the
period-END rate, while income and cash-flow (DURATION) items convert at the
period-AVERAGE rate over the reporting span. Using one spot rate for both
quietly corrupts multi-year growth and margin series -- a revenue line grown
in a depreciating currency looks like real growth, and a balance sheet dated
mid-quarter looks wrong relative to its own income statement. See PLAN.md
section 1.4/1.5.

Rates come from the European Central Bank (`data-api.ecb.europa.eu`), not from
whichever data vendor sourced the filing. This matters because vendor FX is
frequently undocumented (which rate, which day, bid/mid/ask) and inconsistent
across vendors, which reintroduces exactly the silent-corruption problem this
module exists to prevent. ECB reference rates are free, public, and published
once daily on every TARGET business day.

ECB quotes EUR as the base currency: the "USD" series is USD-per-1-EUR, not
USD-per-1-unit-of-anything-else. That mechanic is intentionally hidden behind
this module's API -- callers ask to convert amount X from currency A to
currency B on/over a date, never touch a raw EUR-denominated rate. EUR is
always the pivot for a cross rate (e.g. JPY -> EUR -> USD).

ECB does not publish a rate on weekends or TARGET holidays. `rate_eur_per`
carries the most recent prior rate forward for such dates -- this is standard
practice (it is what "period-end rate" means when the period end is a
Saturday) and is different from `convert_duration`'s average, which uses only
the business days actually inside the window and refuses to guess if there are
none.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from scout.data.http import HttpClient

log = logging.getLogger(__name__)

BASE_URL = "https://data-api.ecb.europa.eu/service/data/EXR/"

# How far back to widen the query window when looking for a period-end rate,
# to carry forward over weekends and clustered holidays (e.g. the ECB closes
# several consecutive days around Christmas/New Year). Ten calendar days
# comfortably covers every TARGET closure run without risking silently
# reaching back to a stale, economically meaningless rate.
_CARRY_FORWARD_LOOKBACK_DAYS = 10


class FxError(RuntimeError):
    """Raised for anything FX-rate related: an unsupported/unknown currency
    code, or no ECB observations in the requested window.

    Deliberately never swallowed into a fallback spot rate anywhere in this
    module -- a missing rate must be visible to the caller, not guessed at.
    """


def parse_ecb_series(payload: dict[str, Any]) -> dict[date, float]:
    """Pure parser: one ECB EXR `jsondata` response -> {date: rate}.

    SDMX-JSON keeps the reported dates once, in
    `structure.dimensions.observation[0].values` (a list, in query order), and
    each series' `observations` dict maps a *string index* into that list to
    an observation-value array whose first element is the rate:

        {
          "dataSets": [{"series": {"0:0:0:0:0": {"observations": {
              "0": [1.0895, 0, null, null, 0],
              "2": [1.0912, 0, null, null, 0]
          }}}}],
          "structure": {"dimensions": {"observation": [{
              "id": "TIME_PERIOD",
              "values": [{"id": "2026-01-02"}, {"id": "2026-01-03"}, {"id": "2026-01-06"}]
          }]}}
        }

    Index "0" is 2026-01-02, index "2" is 2026-01-06 -- note index "1"
    (2026-01-03) is simply absent from `observations` above, which is exactly
    how ECB represents a day inside the query range with no published
    observation (e.g. a national holiday still inside a mixed window).

    Pulled out as a pure function -- no httpx/respx needed to test it
    thoroughly against a real captured shape -- because getting this parse
    right is the entire point of the module; the async plumbing around it is
    comparatively low-risk.
    """
    data_sets = payload.get("dataSets") or []
    if not data_sets:
        return {}

    series_map = data_sets[0].get("series") or {}
    if not series_map:
        return {}

    # EXR's SP00.A key selects exactly one series per request (one currency,
    # daily, ECB reference, average). We don't assume its dict key, which ECB
    # derives from dimension positions we have no other need to decode.
    (series_body,) = series_map.values()
    observations: dict[str, Any] = series_body.get("observations") or {}

    time_values = payload["structure"]["dimensions"]["observation"][0]["values"]

    result: dict[date, float] = {}
    for index_str, obs in observations.items():
        if not obs or obs[0] is None:
            continue  # published index with a null value, e.g. a provisional gap
        day = date.fromisoformat(time_values[int(index_str)]["id"])
        result[day] = float(obs[0])
    return result


def _normalize_ccy(code: str) -> str:
    code = code.strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError(f"Not a valid ISO 4217 currency code: {code!r}")
    return code


def _rate_on_or_before(series: dict[date, float], on: date, currency: str) -> float:
    eligible = {d: rate for d, rate in series.items() if d <= on}
    if not eligible:
        raise FxError(
            f"No ECB reference rate for {currency} on or before {on.isoformat()} "
            f"within the {_CARRY_FORWARD_LOOKBACK_DAYS}-day lookback window."
        )
    latest = max(eligible)
    return eligible[latest]


class FxRates:
    """Convert amounts between currencies using ECB reference rates, IAS 21-style.

    One instance is meant to be shared across a batch conversion run (many
    filings, many concepts) so the in-memory series cache actually pays off --
    the common case is many `convert_instant` calls for the same entity and
    period end (one per balance-sheet concept), which collapse to one HTTP
    fetch per currency.
    """

    def __init__(self, http: HttpClient, *, cache_dir: Path | None = None) -> None:
        self._http = http
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        # Keyed by the exact (currency, start, end) window requested. Not an
        # interval index -- it only dedupes identical windows, which is the
        # case that matters (same period end / same fiscal period asked about
        # repeatedly), not arbitrary overlap.
        self._series_cache: dict[tuple[str, date, date], dict[date, float]] = {}

    async def rate_eur_per(self, currency: str, on: date) -> float:
        """EUR-base rate (units of `currency` per 1 EUR) on or before `on`.

        Carries the most recent prior published rate forward across
        weekends/holidays, per the module docstring.
        """
        currency = _normalize_ccy(currency)
        if currency == "EUR":
            return 1.0  # EUR-per-EUR is definitionally 1; never worth a request

        start = on - timedelta(days=_CARRY_FORWARD_LOOKBACK_DAYS)
        series = await self._series(currency, start, on)
        return _rate_on_or_before(series, on, currency)

    async def _avg_rate_eur_per(self, currency: str, start: date, end: date) -> float:
        """Mean EUR-base rate over the business days actually inside [start, end].

        No carry-forward and no lookback widening here: a duration average is
        already an approximation of "the rate over this span", so silently
        reaching outside the span to manufacture an average would be its own
        version of the bug this module exists to prevent.
        """
        currency = _normalize_ccy(currency)
        if currency == "EUR":
            return 1.0

        series = await self._series(currency, start, end)
        values = [rate for day, rate in series.items() if start <= day <= end]
        if not values:
            raise FxError(
                f"No ECB reference rate for {currency} anywhere in "
                f"{start.isoformat()}..{end.isoformat()} -- cannot compute a "
                "period-average rate. Refusing to fall back to a spot rate."
            )
        return mean(values)

    async def convert_instant(
        self, amount: float, *, from_ccy: str, to_ccy: str, on: date
    ) -> float:
        """IAS 21 for balance-sheet items: period-END spot rate."""
        from_ccy, to_ccy = _normalize_ccy(from_ccy), _normalize_ccy(to_ccy)
        if from_ccy == to_ccy:
            return amount  # no-op short-circuit -- must not touch the network

        eur_amount = amount
        if from_ccy != "EUR":
            eur_amount = amount / await self.rate_eur_per(from_ccy, on)
        if to_ccy == "EUR":
            return eur_amount
        return eur_amount * await self.rate_eur_per(to_ccy, on)

    async def convert_duration(
        self, amount: float, *, from_ccy: str, to_ccy: str, start: date, end: date
    ) -> float:
        """IAS 21 for income/cash-flow items: AVERAGE rate over [start, end]."""
        from_ccy, to_ccy = _normalize_ccy(from_ccy), _normalize_ccy(to_ccy)
        if from_ccy == to_ccy:
            return amount  # no-op short-circuit -- must not touch the network
        if start > end:
            raise ValueError(f"convert_duration window is inverted: start={start} end={end}")

        eur_amount = amount
        if from_ccy != "EUR":
            eur_amount = amount / await self._avg_rate_eur_per(from_ccy, start, end)
        if to_ccy == "EUR":
            return eur_amount
        return eur_amount * await self._avg_rate_eur_per(to_ccy, start, end)

    async def _series(self, currency: str, start: date, end: date) -> dict[date, float]:
        key = (currency, start, end)
        cached = self._series_cache.get(key)
        if cached is not None:
            return cached

        from_disk = self._read_disk_cache(currency, start, end)
        if from_disk is not None:
            self._series_cache[key] = from_disk
            return from_disk

        payload = await self._fetch(currency, start, end)
        series = parse_ecb_series(payload)
        if not series:
            # Distinguishing "unsupported currency" from "supported currency,
            # empty window" isn't reliably possible from ECB's response shape
            # alone, but either way the honest move is to say so, not to
            # invent a rate.
            raise FxError(
                f"ECB returned no observations for {currency} in "
                f"{start.isoformat()}..{end.isoformat()}."
            )

        self._series_cache[key] = series
        self._write_disk_cache(currency, start, end, series)
        return series

    async def _fetch(self, currency: str, start: date, end: date) -> dict[str, Any]:
        url = f"{BASE_URL}D.{currency}.EUR.SP00.A"
        params = {
            "startPeriod": start.isoformat(),
            "endPeriod": end.isoformat(),
            "format": "jsondata",
        }
        # HttpClient already knows the ECB rate limit for this host (see
        # data/http.py DEFAULT_LIMITS); we just make the request.
        response = await self._http.get(
            url, params=params, headers={"Accept": "application/vnd.sdmx.data+json"}
        )
        if response.status_code == 404:
            # ECB's SDMX REST endpoint 404s both for a currency code it has
            # never heard of and for a syntactically valid query with zero
            # matching observations -- same clear message covers both.
            raise FxError(
                f"ECB has no EXR reference-rate data for {currency!r} in "
                f"{start.isoformat()}..{end.isoformat()} (unsupported currency "
                "code, or nothing published in this window)."
            )
        response.raise_for_status()
        return response.json()

    def _cache_path(self, currency: str, start: date, end: date) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{currency}_{start.isoformat()}_{end.isoformat()}.json"

    def _read_disk_cache(self, currency: str, start: date, end: date) -> dict[date, float] | None:
        path = self._cache_path(currency, start, end)
        if path is None or not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {date.fromisoformat(k): float(v) for k, v in raw.items()}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # A corrupt cache entry is a miss, not a crash -- but logged, so a
            # cache that quietly stops hitting doesn't waste an afternoon.
            log.warning("Ignoring unreadable FX cache entry %s: %s", path, exc)
            return None

    def _write_disk_cache(
        self, currency: str, start: date, end: date, series: dict[date, float]
    ) -> None:
        path = self._cache_path(currency, start, end)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {day.isoformat(): rate for day, rate in series.items()}
            # Write-then-rename: a crash mid-write must not leave a truncated
            # file that a later run reads back as a corrupt hit.
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload), encoding="utf-8")
            temp.replace(path)
        except OSError as exc:
            log.warning("Could not write FX cache entry %s: %s", path, exc)
