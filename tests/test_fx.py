from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import respx
from httpx import Response

from scout.data.http import HttpClient
from scout.fundamentals.fx import BASE_URL, FxError, FxRates, parse_ecb_series

USER_AGENT = "scout-test test@example.com"


def _url(currency: str) -> str:
    return f"{BASE_URL}D.{currency}.EUR.SP00.A"


def _sdmx_payload(dates_and_rates: list[tuple[str, float]]) -> dict:
    """Build a minimal but shape-faithful ECB EXR `jsondata` payload.

    Mirrors the real SDMX-JSON structure: observation indices are stringified
    positions into `structure.dimensions.observation[0].values`, which lists
    every reported TIME_PERIOD once, in order.
    """
    values = [{"id": d, "name": d} for d, _ in dates_and_rates]
    observations = {str(i): [rate, 0, None, None, 0] for i, (_, rate) in enumerate(dates_and_rates)}
    return {
        "header": {"id": "test", "test": True},
        "dataSets": [
            {
                "action": "Information",
                "series": {
                    "0:0:0:0:0": {
                        "attributes": [0, 0, 0, 0],
                        "observations": observations,
                    }
                },
            }
        ],
        "structure": {
            "name": "Exchange Rates",
            "dimensions": {
                "series": [],
                "observation": [
                    {
                        "id": "TIME_PERIOD",
                        "name": "Time period or range",
                        "role": "time",
                        "values": values,
                    }
                ],
            },
        },
    }


# --------------------------------------------------------------------------
# parse_ecb_series -- pure function, tested thoroughly on its own
# --------------------------------------------------------------------------


def test_parse_ecb_series_basic():
    payload = _sdmx_payload([("2026-01-02", 1.0895), ("2026-01-05", 1.0912)])

    series = parse_ecb_series(payload)

    assert series == {date(2026, 1, 2): 1.0895, date(2026, 1, 5): 1.0912}


def test_parse_ecb_series_skips_null_observation():
    """A provisional/withheld index still appears in `observations` but with a
    null value -- must be dropped, not turned into a 0.0 rate."""
    payload = _sdmx_payload([("2026-01-02", 1.0895), ("2026-01-05", 1.0912)])
    payload["dataSets"][0]["series"]["0:0:0:0:0"]["observations"]["1"] = [None, 0, None, None, 0]

    series = parse_ecb_series(payload)

    assert series == {date(2026, 1, 2): 1.0895}


def test_parse_ecb_series_sparse_index_gap():
    """A TIME_PERIOD value with no corresponding key in `observations` at all
    (a holiday inside a mixed window) must simply be absent from the result."""
    payload = _sdmx_payload([("2026-01-02", 1.0895), ("2026-01-03", 1.09), ("2026-01-06", 1.0912)])
    del payload["dataSets"][0]["series"]["0:0:0:0:0"]["observations"]["1"]

    series = parse_ecb_series(payload)

    assert series == {date(2026, 1, 2): 1.0895, date(2026, 1, 6): 1.0912}


def test_parse_ecb_series_empty_dataset():
    payload = {"dataSets": [], "structure": {"dimensions": {"observation": [{"values": []}]}}}

    assert parse_ecb_series(payload) == {}


def test_parse_ecb_series_no_series_in_dataset():
    payload = {
        "dataSets": [{"action": "Information", "series": {}}],
        "structure": {"dimensions": {"observation": [{"values": []}]}},
    }

    assert parse_ecb_series(payload) == {}


# --------------------------------------------------------------------------
# rate_eur_per
# --------------------------------------------------------------------------


@respx.mock
async def test_rate_eur_per_returns_exact_date():
    http = HttpClient(user_agent=USER_AGENT)
    fx = FxRates(http)
    payload = _sdmx_payload([("2026-01-02", 1.0895), ("2026-01-05", 1.0912)])
    respx.get(_url("USD")).mock(return_value=Response(200, json=payload))

    rate = await fx.rate_eur_per("USD", date(2026, 1, 5))

    assert rate == 1.0912
    await http.aclose()


@respx.mock
async def test_rate_eur_per_carries_forward_over_weekend():
    """2026-01-03 is a Saturday with no ECB publication; querying it must
    return Friday 2026-01-02's rate, not fail and not use Monday's."""
    http = HttpClient(user_agent=USER_AGENT)
    fx = FxRates(http)
    payload = _sdmx_payload([("2026-01-02", 1.0895), ("2026-01-05", 1.0912)])
    respx.get(_url("USD")).mock(return_value=Response(200, json=payload))

    rate = await fx.rate_eur_per("USD", date(2026, 1, 3))

    assert rate == 1.0895
    await http.aclose()


@respx.mock
async def test_rate_eur_per_eur_is_always_one_no_http_call():
    http = HttpClient(user_agent=USER_AGENT)
    fx = FxRates(http)

    rate = await fx.rate_eur_per("EUR", date(2026, 1, 5))

    assert rate == 1.0
    assert not respx.calls
    await http.aclose()


# --------------------------------------------------------------------------
# convert_instant
# --------------------------------------------------------------------------


@respx.mock
async def test_convert_instant_uses_end_date_rate():
    http = HttpClient(user_agent=USER_AGENT)
    fx = FxRates(http)
    # 100 USD / 1.10 USD-per-EUR = ~90.91 EUR
    payload = _sdmx_payload([("2026-01-30", 1.10)])
    respx.get(_url("USD")).mock(return_value=Response(200, json=payload))

    result = await fx.convert_instant(100.0, from_ccy="USD", to_ccy="EUR", on=date(2026, 1, 30))

    assert result == pytest.approx(90.9090909, rel=1e-6)
    await http.aclose()


async def test_convert_instant_same_currency_short_circuits_no_http():
    with respx.mock:
        http = HttpClient(user_agent=USER_AGENT)
        fx = FxRates(http)

        result = await fx.convert_instant(
            100.0, from_ccy="USD", to_ccy="usd", on=date(2026, 1, 30)
        )

        assert result == 100.0
        assert not respx.calls
        await http.aclose()


@respx.mock
async def test_convert_instant_eur_to_x_multiplies():
    http = HttpClient(user_agent=USER_AGENT)
    fx = FxRates(http)
    payload = _sdmx_payload([("2026-01-30", 1.10)])
    respx.get(_url("USD")).mock(return_value=Response(200, json=payload))

    result = await fx.convert_instant(100.0, from_ccy="EUR", to_ccy="USD", on=date(2026, 1, 30))

    assert result == pytest.approx(110.0)
    await http.aclose()


# --------------------------------------------------------------------------
# convert_duration
# --------------------------------------------------------------------------


@respx.mock
async def test_convert_duration_averages_across_window():
    http = HttpClient(user_agent=USER_AGENT)
    fx = FxRates(http)
    # Mean of 1.00 and 1.10 is 1.05 -- an obviously-checkable number.
    payload = _sdmx_payload([("2026-01-02", 1.00), ("2026-01-05", 1.10)])
    respx.get(_url("USD")).mock(return_value=Response(200, json=payload))

    result = await fx.convert_duration(
        105.0, from_ccy="USD", to_ccy="EUR", start=date(2026, 1, 1), end=date(2026, 1, 6)
    )

    assert result == pytest.approx(100.0)
    await http.aclose()


async def test_convert_duration_same_currency_short_circuits_no_http():
    with respx.mock:
        http = HttpClient(user_agent=USER_AGENT)
        fx = FxRates(http)

        result = await fx.convert_duration(
            50.0, from_ccy="JPY", to_ccy="JPY", start=date(2026, 1, 1), end=date(2026, 1, 31)
        )

        assert result == 50.0
        assert not respx.calls
        await http.aclose()


async def test_convert_duration_rejects_inverted_window():
    http = HttpClient(user_agent=USER_AGENT)
    fx = FxRates(http)

    with pytest.raises(ValueError, match="inverted"):
        await fx.convert_duration(
            10.0, from_ccy="USD", to_ccy="EUR", start=date(2026, 2, 1), end=date(2026, 1, 1)
        )
    await http.aclose()


@respx.mock
async def test_convert_duration_empty_window_raises_not_spot_fallback():
    http = HttpClient(user_agent=USER_AGENT)
    fx = FxRates(http)
    # 200 OK but genuinely nothing published in the requested span.
    payload = _sdmx_payload([])
    respx.get(_url("USD")).mock(return_value=Response(200, json=payload))

    with pytest.raises(FxError, match="USD"):
        await fx.convert_duration(
            10.0, from_ccy="USD", to_ccy="EUR", start=date(2026, 1, 1), end=date(2026, 1, 2)
        )
    await http.aclose()


@respx.mock
async def test_unsupported_currency_404_raises_clear_error():
    http = HttpClient(user_agent=USER_AGENT)
    fx = FxRates(http)
    respx.get(_url("ZZZ")).mock(return_value=Response(404, json={"message": "No results found."}))

    with pytest.raises(FxError, match="ZZZ"):
        await fx.rate_eur_per("ZZZ", date(2026, 1, 30))
    await http.aclose()


# --------------------------------------------------------------------------
# Cross rate: X -> Y through EUR, two series
# --------------------------------------------------------------------------


@respx.mock
async def test_cross_rate_goes_through_eur_using_two_series():
    http = HttpClient(user_agent=USER_AGENT)
    fx = FxRates(http)
    # USD 1.10 per EUR, JPY 160.0 per EUR on the same date.
    usd_payload = _sdmx_payload([("2026-01-30", 1.10)])
    jpy_payload = _sdmx_payload([("2026-01-30", 160.0)])
    usd_route = respx.get(_url("USD")).mock(return_value=Response(200, json=usd_payload))
    jpy_route = respx.get(_url("JPY")).mock(return_value=Response(200, json=jpy_payload))

    # 110 USD -> 100 EUR -> 16000 JPY
    result = await fx.convert_instant(110.0, from_ccy="USD", to_ccy="JPY", on=date(2026, 1, 30))

    assert result == pytest.approx(16000.0)
    assert usd_route.called
    assert jpy_route.called
    await http.aclose()


# --------------------------------------------------------------------------
# Series caching
# --------------------------------------------------------------------------


@respx.mock
async def test_same_window_conversion_hits_http_once():
    http = HttpClient(user_agent=USER_AGENT)
    fx = FxRates(http)
    payload = _sdmx_payload([("2026-01-02", 1.00), ("2026-01-05", 1.10)])
    route = respx.get(_url("USD")).mock(return_value=Response(200, json=payload))

    first = await fx.convert_duration(
        105.0, from_ccy="USD", to_ccy="EUR", start=date(2026, 1, 1), end=date(2026, 1, 6)
    )
    second = await fx.convert_duration(
        210.0, from_ccy="USD", to_ccy="EUR", start=date(2026, 1, 1), end=date(2026, 1, 6)
    )

    assert first == pytest.approx(100.0)
    assert second == pytest.approx(200.0)
    assert route.call_count == 1
    await http.aclose()


@respx.mock
async def test_disk_cache_persists_across_instances(tmp_path: Path):
    http = HttpClient(user_agent=USER_AGENT)
    payload = _sdmx_payload([("2026-01-30", 1.10)])
    route = respx.get(_url("USD")).mock(return_value=Response(200, json=payload))

    fx1 = FxRates(http, cache_dir=tmp_path)
    rate1 = await fx1.rate_eur_per("USD", date(2026, 1, 30))

    # A brand-new instance, same cache_dir, must reuse the on-disk series
    # instead of refetching.
    fx2 = FxRates(http, cache_dir=tmp_path)
    rate2 = await fx2.rate_eur_per("USD", date(2026, 1, 30))

    assert rate1 == rate2 == 1.10
    assert route.call_count == 1
    await http.aclose()


def test_invalid_currency_code_rejected():
    from scout.fundamentals.fx import _normalize_ccy

    with pytest.raises(ValueError, match="ISO 4217"):
        _normalize_ccy("US")


async def test_convert_instant_rejects_invalid_currency_code():
    """The bad-code check must fire before any HTTP call is attempted."""
    with respx.mock:
        http = HttpClient(user_agent=USER_AGENT)
        fx = FxRates(http)

        with pytest.raises(ValueError, match="ISO 4217"):
            await fx.convert_instant(10.0, from_ccy="US", to_ccy="EUR", on=date(2026, 1, 30))

        assert not respx.calls
        await http.aclose()
