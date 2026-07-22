"""Content-addressed replay cache for model calls.

The highest-ROI 50 lines in the harness (PLAN.md section 4.4). It buys:

  * crash resumability -- a fan-out over 60 candidates that dies at 47 resumes
    for the price of the last 13;
  * offline iteration -- stages 4 and 5 can be rewritten and re-run all day
    without paying for stage 3 again;
  * deterministic prompt regression tests -- diff two runs and the only thing
    that changed is the thing you changed.

Layout: `<root>/<first two hex chars of key>/<key>.json`. The shard directory
exists because a run over a full universe puts tens of thousands of files in
one place, and both Windows Explorer and `ls` get unpleasant well before that.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scout.harness.protocol import (
    Capabilities,
    Effort,
    LLMClient,
    Message,
    ModelResponse,
    OutputMode,
    Usage,
    fingerprint,
)

log = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1


def cache_key(fingerprint_dict: dict[str, Any]) -> str:
    """sha256 over the canonical JSON form of a `protocol.fingerprint()` dict.

    THE detail this whole module turns on: the key covers the provider, the
    model, *every* sampling parameter, the JSON schema and the prompt version --
    not just the prompt text. Keying on prompt text alone is the standard way
    people end up serving subtly wrong cached results: you drop temperature from
    0.7 to 0, or fix a field in the schema, or edit the rubric prompt, and the
    cache keeps handing back answers generated under the old settings. Nothing
    errors; the numbers are just quietly from a different experiment.

    `fingerprint()` owns which fields those are, so this function must never
    filter or reorder them -- it only canonicalises (sorted keys, no
    whitespace) so that dict ordering cannot change the hash.
    """
    canonical = json.dumps(
        fingerprint_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def __str__(self) -> str:
        return (
            f"cache: {self.hits} hits, {self.misses} misses, "
            f"{self.writes} writes ({self.hit_rate:.0%} hit rate)"
        )


class ReplayCache:
    """JSON files on disk, addressed by call fingerprint.

    `read_only=True` is for reproducing a past run exactly: misses still go to
    the provider (so the run completes), but nothing new is written, so the
    cache stays a snapshot of the run being reproduced.
    """

    def __init__(self, root: Path | str, *, read_only: bool = False) -> None:
        self.root = Path(root)
        self.read_only = read_only
        self.stats = CacheStats()

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> ModelResponse | None:
        path = self.path_for(key)
        if not path.exists():
            self.stats.misses += 1
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            response = _decode(payload)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # A corrupt entry is a miss, not a crash -- but it is never silent,
            # because a cache that quietly stops hitting looks exactly like a
            # cache key bug and we would waste an afternoon on it.
            log.warning("Ignoring unreadable cache entry %s: %s", path, exc)
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        return response

    def put(
        self,
        key: str,
        response: ModelResponse,
        *,
        fingerprint_dict: dict[str, Any] | None = None,
    ) -> None:
        """Store a response. Only ever called with successful responses."""
        if self.read_only:
            return

        path = self.path_for(key)
        payload = _encode(response, fingerprint_dict)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a crash mid-write must not leave a truncated
            # file that later reads as a corrupt hit.
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            temp.replace(path)
        except OSError as exc:
            log.warning("Could not write cache entry %s: %s", path, exc)
            return

        self.stats.writes += 1


def _encode(response: ModelResponse, fingerprint_dict: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "cached_at": datetime.now(UTC).isoformat(),
        # Kept for debugging only. Never read back -- the filename is the key.
        "fingerprint": fingerprint_dict,
        "response": {
            "text": response.text,
            "reasoning": response.reasoning,
            "model": response.model,
            "output_mode": response.output_mode.value,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cached_input_tokens": response.usage.cached_input_tokens,
                "reasoning_tokens": response.usage.reasoning_tokens,
            },
            "raw": response.raw,
        },
    }


def _decode(payload: dict[str, Any]) -> ModelResponse:
    body = payload["response"]
    return ModelResponse(
        text=body["text"],
        reasoning=body.get("reasoning"),
        usage=Usage(**body["usage"]),
        model=body["model"],
        output_mode=OutputMode(body["output_mode"]),
        raw=body.get("raw") or {},
    )


class CachedClient:
    """An LLMClient that answers from disk when it can.

    Wraps any LLMClient and preserves `name`, `model` and `capabilities`, so
    everything above it -- especially `structured.py`'s ladder, which reads
    `capabilities` to choose a rung -- behaves identically cached or not.

    Only successful responses are stored. Caching an exception would turn one
    bad afternoon at a provider into a permanently poisoned run, and the whole
    point of the cache is that re-running is cheap.
    """

    def __init__(
        self,
        inner: LLMClient,
        cache: ReplayCache,
        *,
        prompt_version: str | None = None,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.prompt_version = prompt_version

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def capabilities(self) -> Capabilities:
        return self.inner.capabilities

    @property
    def stats(self) -> CacheStats:
        return self.cache.stats

    async def complete(
        self,
        messages: list[Message],
        *,
        output_mode: OutputMode = OutputMode.TEXT,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        effort: Effort | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache_prefix_upto: int | None = None,
    ) -> ModelResponse:
        fingerprint_dict = fingerprint(
            provider=self.inner.name,
            model=self.inner.model,
            messages=messages,
            output_mode=output_mode,
            json_schema=json_schema,
            effort=effort,
            temperature=temperature,
            max_tokens=max_tokens,
            schema_name=schema_name,
            prompt_version=self.prompt_version,
        )
        key = cache_key(fingerprint_dict)

        hit = self.cache.get(key)
        if hit is not None:
            return hit

        # A raised exception propagates from here uncached, on purpose.
        response = await self.inner.complete(
            messages,
            output_mode=output_mode,
            json_schema=json_schema,
            schema_name=schema_name,
            effort=effort,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_prefix_upto=cache_prefix_upto,
        )
        self.cache.put(key, response, fingerprint_dict=fingerprint_dict)
        return response
