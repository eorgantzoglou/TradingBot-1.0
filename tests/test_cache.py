"""Tests for the content-addressed replay cache.

The load-bearing test here is `test_key_changes_when_*`: a cache key that
ignores any input which can change the output is how you end up serving
confidently wrong results.
"""

from __future__ import annotations

import json

import pytest

from scout.harness.cache import CachedClient, ReplayCache, cache_key
from scout.harness.protocol import (
    Capabilities,
    LLMClient,
    Message,
    ModelResponse,
    OutputMode,
    Usage,
    fingerprint,
)

MESSAGES = [Message(role="user", content="Does this filer have a going-concern opinion?")]
SCHEMA = {"type": "object", "properties": {"answer": {"type": "boolean"}}}


def base_fingerprint(**overrides):
    args = {
        "provider": "anthropic",
        "model": "claude-opus-4-8",
        "messages": MESSAGES,
        "output_mode": OutputMode.NATIVE_SCHEMA,
        "json_schema": SCHEMA,
        "effort": "low",
        "temperature": 0.2,
        "max_tokens": 4096,
        "prompt_version": "extract-v3",
    }
    args.update(overrides)
    return fingerprint(**args)


@pytest.fixture
def cache(tmp_path) -> ReplayCache:
    return ReplayCache(tmp_path / "llm")


# --------------------------------------------------------------------------
# The key
# --------------------------------------------------------------------------


def test_key_is_stable_across_dict_ordering():
    left = cache_key({"a": 1, "b": {"x": 1, "y": 2}})
    right = cache_key({"b": {"y": 2, "x": 1}, "a": 1})
    assert left == right


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "claude-haiku-4-5"),
        ("provider", "openai_compat"),
        ("temperature", 0.9),
        ("max_tokens", 8192),
        ("effort", "high"),
        ("output_mode", OutputMode.JSON_OBJECT),
        ("prompt_version", "extract-v4"),
        ("json_schema", {"type": "object", "properties": {"answer": {"type": "string"}}}),
        ("messages", [Message(role="user", content="a different question")]),
    ],
)
def test_key_changes_when_any_output_affecting_input_changes(field, value):
    """Prompt text alone is not the key. Every one of these must move it."""
    assert cache_key(base_fingerprint()) != cache_key(base_fingerprint(**{field: value}))


def test_identical_calls_share_a_key():
    assert cache_key(base_fingerprint()) == cache_key(base_fingerprint())


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def response(text: str = "hello") -> ModelResponse:
    return ModelResponse(
        text=text,
        reasoning="some thinking",
        usage=Usage(input_tokens=10, output_tokens=5, cached_input_tokens=2),
        model="claude-opus-4-8",
        output_mode=OutputMode.NATIVE_SCHEMA,
        raw={"id": "msg_1"},
    )


def test_roundtrip_preserves_every_field(cache):
    cache.put("abc123", response())

    restored = cache.get("abc123")

    assert restored == response()


def test_entries_are_sharded_by_key_prefix(cache):
    cache.put("ab" + "0" * 62, response())
    assert (cache.root / "ab" / ("ab" + "0" * 62 + ".json")).exists()


def test_miss_then_hit_counts(cache):
    assert cache.get("missing") is None
    cache.put("present", response())
    cache.get("present")

    assert cache.stats.misses == 1
    assert cache.stats.hits == 1
    assert cache.stats.writes == 1


def test_corrupt_entry_reads_as_a_miss_not_a_crash(cache, caplog):
    path = cache.path_for("deadbeef")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json", encoding="utf-8")

    assert cache.get("deadbeef") is None
    assert cache.stats.misses == 1
    assert "unreadable cache entry" in caplog.text  # never silent


def test_read_only_cache_serves_but_does_not_write(tmp_path):
    writable = ReplayCache(tmp_path / "llm")
    writable.put("key1", response("original"))

    frozen = ReplayCache(tmp_path / "llm", read_only=True)
    frozen.put("key2", response("new"))

    assert frozen.get("key1").text == "original"
    assert frozen.get("key2") is None
    assert frozen.stats.writes == 0


# --------------------------------------------------------------------------
# CachedClient
# --------------------------------------------------------------------------


async def test_cached_client_delegates_on_miss_and_replays_on_hit(make_client, cache):
    inner = make_client(["first answer"], capabilities=Capabilities(json_object=True))
    client = CachedClient(inner, cache, prompt_version="v1")

    first = await client.complete(MESSAGES, output_mode=OutputMode.JSON_OBJECT)
    second = await client.complete(MESSAGES, output_mode=OutputMode.JSON_OBJECT)

    assert first == second
    assert len(inner.calls) == 1  # the second call never reached the provider
    assert client.stats.hits == 1
    assert client.stats.misses == 1


async def test_cached_client_preserves_identity(make_client, cache):
    capabilities = Capabilities(native_schema=True, prompt_cache=True)
    inner = make_client(["ok"], capabilities=capabilities, name="anthropic", model="claude-opus-4-8")
    client = CachedClient(inner, cache)

    assert client.name == "anthropic"
    assert client.model == "claude-opus-4-8"
    assert client.capabilities is capabilities
    assert isinstance(client, LLMClient)


async def test_changing_temperature_bypasses_the_cache(make_client, cache):
    inner = make_client(["cold", "warm"])
    client = CachedClient(inner, cache)

    cold = await client.complete(MESSAGES, temperature=0.0)
    warm = await client.complete(MESSAGES, temperature=0.9)

    assert cold.text == "cold"
    assert warm.text == "warm"
    assert len(inner.calls) == 2


async def test_changing_prompt_version_bypasses_the_cache(make_client, cache):
    inner = make_client(["v1 answer"])
    await CachedClient(inner, cache, prompt_version="v1").complete(MESSAGES)

    inner_v2 = make_client(["v2 answer"])
    result = await CachedClient(inner_v2, cache, prompt_version="v2").complete(MESSAGES)

    assert result.text == "v2 answer"


async def test_exceptions_are_never_cached(make_client, cache):
    inner = make_client([RuntimeError("provider is having a day"), "recovered"])
    client = CachedClient(inner, cache)

    with pytest.raises(RuntimeError):
        await client.complete(MESSAGES)

    assert client.stats.writes == 0
    assert (await client.complete(MESSAGES)).text == "recovered"


async def test_cache_file_records_the_fingerprint_for_debugging(make_client, cache):
    inner = make_client(["ok"])
    client = CachedClient(inner, cache, prompt_version="v1")
    await client.complete(MESSAGES, temperature=0.2)

    stored = next(cache.root.rglob("*.json"))
    payload = json.loads(stored.read_text(encoding="utf-8"))

    assert payload["fingerprint"]["prompt_version"] == "v1"
    assert payload["fingerprint"]["temperature"] == 0.2
