"""The provider-facing contract.

Everything above this layer (structured output, cache, cost, the research
pipeline) speaks only in these types. Everything below it is adapter-specific
mess -- four reasoning-field conventions, two JSON-schema subsets, per-provider
parameter names -- and is not allowed to leak upward.

Deliberately hand-written rather than delegated to LiteLLM: its normalization is
leaky in exactly the place we need it (see PLAN.md section 4.1), and this file is
about 100 lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]

# Normalized reasoning budget. Adapters translate this to whatever their
# provider actually calls it: OpenAI `reasoning_effort`, Anthropic
# `thinking.budget_tokens`, Ollama `think`, vLLM `chat_template_kwargs`.
# "none" is meaningful and load-bearing: for routine structured extraction,
# thinking is pure latency (measured 193x on a local Qwen3 in the previous
# incarnation of this project).
Effort = Literal["none", "low", "medium", "high"]


class OutputMode(StrEnum):
    """Rungs of the structured-output ladder, most to least constrained.

    `structured.py` picks the highest rung the adapter advertises and walks down
    on UnsupportedParameterError.
    """

    NATIVE_SCHEMA = "native_schema"  # grammar-constrained decoding against a JSON Schema
    STRICT_TOOL = "strict_tool"  # a single strict-mode tool call
    JSON_OBJECT = "json_object"  # "must be valid JSON", shape unenforced
    TEXT = "text"  # free text; we scrape a JSON block out of it


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts, normalized across providers.

    Two conventions that adapters MUST follow, because getting either wrong
    silently corrupts every cost number downstream:

      * `input_tokens` and `cached_input_tokens` are DISJOINT, not nested.
        Anthropic reports them that way natively; OpenAI-compatible servers
        report `prompt_tokens` as a total including `cached_tokens`, so those
        adapters must subtract before filling this in. Treating them as nested
        double-bills every cached call.
      * `reasoning_tokens` are already INSIDE `output_tokens` -- both providers
        report them that way. `cost.py` does not bill them a second time.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    # Tokens served from the provider's prompt cache. Priced ~0.1x on Anthropic,
    # so tracking this separately is the difference between a real cost number
    # and a fictional one.
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    reasoning: str | None
    usage: Usage
    model: str
    output_mode: OutputMode
    # Provider payload, kept for debugging and for the replay cache. Never
    # interpreted above this layer.
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What an adapter can actually do, so the ladder does not guess.

    These are per-(provider, model), not per-provider: a local server's
    capabilities depend entirely on which model is loaded and how it was
    launched.
    """

    native_schema: bool = False
    strict_tools: bool = False
    json_object: bool = True
    effort: bool = False
    prompt_cache: bool = False
    # Anthropic's JSON Schema subset is narrower than OpenAI's: no recursive
    # schemas, no minimum/maximum, no minLength/maxLength, no array constraints
    # beyond minItems 0|1, and additionalProperties must be false. Models in
    # research/models.py are written to the intersection, but adapters that need
    # to strip keywords advertise it here.
    restricted_schema_keywords: frozenset[str] = frozenset()


@runtime_checkable
class LLMClient(Protocol):
    """One call, normalized. Implementations live in `harness/adapters/`.

    Implementations MUST:
      - return `text` with any inline <think> blocks removed and their content
        moved to `reasoning`;
      - raise EmptyContentError (not return "") when the model produced only
        reasoning and no answer;
      - raise UnsupportedParameterError on HTTP 400/422 so the ladder can
        downgrade, and ProviderError on anything else.
    """

    name: str
    model: str
    capabilities: Capabilities

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
        """Run one completion.

        `cache_prefix_upto` is an INDEX into `messages`, not a count: the cache
        breakpoint is placed at the end of `messages[cache_prefix_upto]`, so
        messages 0..cache_prefix_upto inclusive form the cached prefix.
        Adapters supporting explicit prompt caching (Anthropic) act on it;
        adapters with implicit caching (OpenAI, above ~1024 tokens of stable
        prefix) ignore it -- but callers should order prompts
        stable-prefix-first, varying-content-last regardless, because that is
        what makes implicit caching hit at all.
        """
        ...


def fingerprint(
    *,
    provider: str,
    model: str,
    messages: list[Message],
    output_mode: OutputMode,
    json_schema: dict[str, Any] | None,
    effort: Effort | None,
    temperature: float | None,
    max_tokens: int | None,
    schema_name: str | None = None,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    """Canonical description of a call, for the replay cache key.

    Every input that can change the output belongs here. Keying on the prompt
    text alone is the standard way to end up serving subtly wrong cached
    results -- a temperature change or a schema edit must produce a new key.
    """
    return {
        "provider": provider,
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "output_mode": output_mode.value,
        "json_schema": json_schema,
        "effort": effort,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # In STRICT_TOOL mode the tool name reaches the model, so it can change
        # the output even when the schema body is identical.
        "schema_name": schema_name,
        "prompt_version": prompt_version,
    }
