"""Per-stage token and cost accounting.

Phase 6 asks whether the LLM layer is signal or cost (PLAN.md section 6), and
that question is unanswerable without a real dollar figure per stage. So every
`ModelResponse` gets recorded against the stage that produced it, priced, and
rolled up.

Two things this file refuses to do:

  * guess at an unknown model's price -- an invented number is worse than a
    zero, because a zero is obviously wrong and a guess is not;
  * ignore cached input -- Anthropic cache reads are a tenth of base input, and
    a fan-out of 60 candidates over one long shared rubric is close to the ideal
    prompt-caching case. Pricing cache reads at full rate would overstate the
    run several-fold and make the whole ledger fiction.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from scout.harness.protocol import ModelResponse, Usage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Price table
#
# LAST REVIEWED: 2026-07-22.
#
# Prices drift, introductory rates expire, and models are added roughly
# monthly. This table needs a re-read every quarter or so; treat anything more
# than a few months past the date above as indicative, not authoritative. A
# stale table is a silent problem -- the run still completes, the number is
# just wrong -- so the date is here to be checked, not decoration.
#
# USD per 1M tokens. `cached_input` is the price of a prompt-cache *read*.
# ---------------------------------------------------------------------------

# Anthropic multipliers, applied to base input price.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIERS = {"5m": 1.25, "1h": 2.0}


@dataclass(frozen=True, slots=True)
class Price:
    input: float
    output: float
    cached_input: float = 0.0

    def cache_write(self, ttl: str = "5m") -> float:
        """Price of writing this many tokens to the provider's prompt cache.

        Not used by `record()`: `Usage` has no cache-write field, because the
        two SDKs report it differently and the contract deliberately does not
        model it. Exposed so a caller doing explicit `cache_control` placement
        can price the write side of the break-even (two reads on a 5m TTL,
        three on 1h).
        """
        multiplier = CACHE_WRITE_MULTIPLIERS.get(ttl)
        if multiplier is None:
            raise ValueError(
                f"Unknown cache TTL {ttl!r}; expected one of {list(CACHE_WRITE_MULTIPLIERS)}."
            )
        return self.input * multiplier


def _anthropic(input_price: float, output_price: float) -> Price:
    return Price(
        input=input_price,
        output=output_price,
        cached_input=input_price * CACHE_READ_MULTIPLIER,
    )


# Keys are matched by LONGEST PREFIX, so a date-suffixed or vendor-prefixed id
# (`claude-haiku-4-5-20251001`, `gpt-5.1-mini-2026-01-30`) resolves to its
# family without an entry of its own. More specific keys must therefore be
# safe to list alongside less specific ones -- `gpt-5-mini` wins over `gpt-5`.
PRICES: dict[str, Price] = {
    # Anthropic
    "claude-opus-4-8": _anthropic(5.00, 25.00),
    "claude-opus-4-7": _anthropic(5.00, 25.00),
    "claude-opus-4-6": _anthropic(5.00, 25.00),
    "claude-sonnet-5": _anthropic(3.00, 15.00),  # intro $2/$10 ran to 2026-08-31
    "claude-sonnet-4-6": _anthropic(3.00, 15.00),
    "claude-haiku-4-5": _anthropic(1.00, 5.00),
    "claude-fable-5": _anthropic(10.00, 50.00),
    # OpenAI. Cached input is OpenAI's automatic prefix cache, priced at 0.1x
    # base for the GPT-5 family and 0.5x for 4o.
    "gpt-5.1-mini": Price(0.25, 2.00, cached_input=0.025),
    "gpt-5.1": Price(1.25, 10.00, cached_input=0.125),
    "gpt-5-mini": Price(0.25, 2.00, cached_input=0.025),
    "gpt-5-nano": Price(0.05, 0.40, cached_input=0.005),
    "gpt-5": Price(1.25, 10.00, cached_input=0.125),
    "gpt-4o-mini": Price(0.15, 0.60, cached_input=0.075),
    "gpt-4o": Price(2.50, 10.00, cached_input=1.25),
    # DeepSeek V4 (hosted). Prices from api-docs.deepseek.com, read 2026-07-23;
    # cached_input is DeepSeek's context-cache *hit* price (caching is automatic,
    # like OpenAI's). Distinct ids from the "deepseek-r1" local zero-entry below,
    # so a hosted v4 run is priced rather than counted free.
    "deepseek-v4-flash": Price(0.14, 0.28, cached_input=0.0028),
    "deepseek-v4-pro": Price(0.435, 0.87, cached_input=0.003625),
    # Anything served locally: electricity is not in scope. Present so a local
    # Qwen3 run reports 0.0 as a *known* zero rather than an unknown-model
    # warning on every single call.
    "local/": Price(0.0, 0.0),
    "ollama/": Price(0.0, 0.0),
    "qwen": Price(0.0, 0.0),
    "llama": Price(0.0, 0.0),
    "mistral": Price(0.0, 0.0),
    "deepseek-r1": Price(0.0, 0.0),
}

# Ids arrive with routing prefixes attached: OpenRouter uses `anthropic/...`,
# Bedrock `us.anthropic....`. Strip them before matching rather than doubling
# every table row.
_ID_PREFIXES = ("anthropic/", "openai/", "us.anthropic.", "eu.anthropic.", "anthropic.")


def normalize_model_id(model: str) -> str:
    cleaned = model.strip().lower()
    for prefix in _ID_PREFIXES:
        if cleaned.startswith(prefix):
            return cleaned[len(prefix) :]
    return cleaned


def price_for(model: str) -> Price | None:
    """Longest-prefix lookup. Returns None for an unknown model."""
    cleaned = normalize_model_id(model)
    best: str | None = None
    for key in PRICES:
        if cleaned.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return PRICES[best] if best else None


def cost_of(usage: Usage, price: Price) -> float:
    """Dollar cost of one call.

    Assumes `input_tokens` and `cached_input_tokens` are DISJOINT: the former is
    billed at full rate, the latter at the cache-read rate. That matches
    Anthropic's native shape, where `input_tokens` already excludes cache reads.
    OpenAI-compatible adapters must subtract `cached_tokens` from `prompt_tokens`
    before filling in `Usage`, or every cached call is billed twice.

    `reasoning_tokens` are not added: providers already count them inside
    `output_tokens`, and adding them again would inflate every thinking call.
    """
    return (
        usage.input_tokens * price.input
        + usage.cached_input_tokens * price.cached_input
        + usage.output_tokens * price.output
    ) / 1_000_000


@dataclass
class StageTotals:
    calls: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0

    def add(self, usage: Usage, cost_usd: float) -> None:
        self.calls += 1
        self.usage = self.usage + usage
        self.cost_usd += cost_usd


class CostLedger:
    """Token and dollar rollup, bucketed by pipeline stage.

    Usage:

        ledger = CostLedger()
        with ledger.stage("research"):
            ledger.record(response)
        print(ledger.render())

    `stage()` is for the sequential pipeline (harvest -> screen -> research ->
    review). Inside a fan-out, several coroutines share one ledger and a
    context-manager-scoped "current stage" would race between them, so pass
    `stage=` to `record()` explicitly there.
    """

    def __init__(self, *, default_stage: str = "unattributed") -> None:
        self.default_stage = default_stage
        self.stages: dict[str, StageTotals] = {}
        self.warnings: list[str] = []
        self._stack: list[str] = []
        self._unknown_models: set[str] = set()

    @contextmanager
    def stage(self, name: str) -> Iterator[StageTotals]:
        """Enter a stage. The bucket is created on entry, not on first record,
        so a stage that made no calls at all still appears in `render()` -- that
        is exactly what a fully-cached run looks like, and it is worth seeing."""
        self._stack.append(name)
        try:
            yield self.stages.setdefault(name, StageTotals())
        finally:
            self._stack.pop()

    @property
    def current_stage(self) -> str:
        return self._stack[-1] if self._stack else self.default_stage

    def record(self, response: ModelResponse, *, stage: str | None = None) -> float:
        """Price one response into a stage bucket, returning its cost."""
        return self.record_usage(response.usage, response.model, stage=stage)

    def record_usage(self, usage: Usage, model: str, *, stage: str | None = None) -> float:
        price = price_for(model)
        if price is None:
            price = Price(0.0, 0.0)
            self._warn_unknown(model)

        cost = cost_of(usage, price)
        bucket = self.stages.setdefault(stage or self.current_stage, StageTotals())
        bucket.add(usage, cost)
        return cost

    def _warn_unknown(self, model: str) -> None:
        """Unknown model costs 0.0 and says so. It never raises and never guesses.

        Raising would kill a research run over a bookkeeping gap, and guessing
        from a similar-looking id is how a $25/1M model gets billed as a
        $1/1M one.
        """
        if model in self._unknown_models:
            return
        self._unknown_models.add(model)
        message = f"No price for model {model!r}; counted as $0.00. Update PRICES in cost.py."
        self.warnings.append(message)
        log.warning("%s", message)

    def totals(self) -> StageTotals:
        combined = StageTotals()
        for bucket in self.stages.values():
            combined.calls += bucket.calls
            combined.usage = combined.usage + bucket.usage
            combined.cost_usd += bucket.cost_usd
        return combined

    def snapshot(self) -> dict[str, StageTotals]:
        """Copy of the per-stage rollup, safe to keep past further recording."""
        return {name: replace(bucket) for name, bucket in self.stages.items()}

    def render(self) -> str:
        """Small fixed-width table, newest stages last, TOTAL at the bottom."""
        header = f"{'stage':<20}{'calls':>7}{'input':>12}{'cached':>12}{'output':>12}{'cost USD':>12}"
        lines = [header, "-" * len(header)]

        for name, bucket in self.stages.items():
            lines.append(_render_row(name, bucket))

        total = self.totals()
        lines.append("-" * len(header))
        lines.append(_render_row("TOTAL", total))

        if self.warnings:
            lines.append("")
            lines.extend(f"warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def _render_row(name: str, bucket: StageTotals) -> str:
    usage = bucket.usage
    return (
        f"{name[:20]:<20}"
        f"{bucket.calls:>7}"
        f"{usage.input_tokens:>12,}"
        f"{usage.cached_input_tokens:>12,}"
        f"{usage.output_tokens:>12,}"
        f"{bucket.cost_usd:>12.4f}"
    )
